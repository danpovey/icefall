#!/usr/bin/env python3

import argparse
import logging
import math
from typing import List

import k2
import kaldifeat
import torch
import torchaudio
from conformer import Conformer
from torch.nn.utils.rnn import pad_sequence

from icefall.decode import (
    get_lattice,
    one_best_decoding,
    rescore_with_attention_decoder,
    rescore_with_whole_lattice,
)
from icefall.utils import AttributeDict, get_texts


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint. "
        "The checkpoint is assumed to be saved by "
        "icefall.checkpoint.save_checkpoint().",
    )

    parser.add_argument(
        "--words-file",
        type=str,
        required=True,
        help="Path to words.txt",
    )

    parser.add_argument(
        "--HLG", type=str, required=True, help="Path to HLG.pt."
    )

    parser.add_argument(
        "--method",
        type=str,
        default="1best",
        help="""Decoding method.
        Possible values are:
        (1) 1best - Use the best path as decoding output. Only
            the transformer encoder output is used for decoding.
            We call it HLG decoding.
        (2) whole-lattice-rescoring - Use an LM to rescore the
            decoding lattice and then use 1best to decode the
            rescored lattice.
            We call it HLG decoding + n-gram LM rescoring.
        (3) attention-decoder - Extract n paths from he rescored
            lattice and use the transformer attention decoder for
            rescoring.
            We call it HLG decoding + n-gram LM rescoring + attention
            decoder rescoring.
        """,
    )

    parser.add_argument(
        "--G",
        type=str,
        help="""An LM for rescoring.
        Used only when method is
        whole-lattice-rescoring or attention-decoder.
        It's usually a 4-gram LM.
        """,
    )

    parser.add_argument(
        "--num-paths",
        type=int,
        default=100,
        help="""
        Used only when method is attention-decoder.
        It specifies the size of n-best list.""",
    )

    parser.add_argument(
        "--ngram-lm-scale",
        type=float,
        default=1.3,
        help="""
        Used only when method is whole-lattice-rescoring and attention-decoder.
        It specifies the scale for n-gram LM scores.
        (Note: You need to tune it on a dataset.)
        """,
    )

    parser.add_argument(
        "--attention-decoder-scale",
        type=float,
        default=1.2,
        help="""
        Used only when method is attention-decoder.
        It specifies the scale for attention decoder scores.
        (Note: You need to tune it on a dataset.)
        """,
    )

    parser.add_argument(
        "--lattice-score-scale",
        type=float,
        default=0.5,
        help="""
        Used only when method is attention-decoder.
        It specifies the scale for lattice.scores when
        extracting n-best lists. A smaller value results in
        more unique number of paths with the risk of missing
        the best path.
        """,
    )

    parser.add_argument(
        "--sos-id",
        type=float,
        default=1,
        help="""
        Used only when method is attention-decoder.
        It specifies ID for the SOS token.
        """,
    )

    parser.add_argument(
        "--eos-id",
        type=float,
        default=1,
        help="""
        Used only when method is attention-decoder.
        It specifies ID for the EOS token.
        """,
    )

    parser.add_argument(
        "sound_files",
        type=str,
        nargs="+",
        help="The input sound file(s) to transcribe. "
        "Supported formats are those supported by torchaudio.load(). "
        "For example, wav and flac are supported. "
        "The sample rate has to be 16kHz.",
    )

    return parser


def get_params() -> AttributeDict:
    params = AttributeDict(
        {
            "feature_dim": 80,
            "nhead": 8,
            "num_classes": 5000,
            "sample_rate": 16000,
            "attention_dim": 512,
            "subsampling_factor": 4,
            "num_decoder_layers": 6,
            "vgg_frontend": False,
            "is_espnet_structure": True,
            "mmi_loss": False,
            "use_feat_batchnorm": True,
            "search_beam": 20,
            "output_beam": 8,
            "min_active_states": 30,
            "max_active_states": 10000,
            "use_double_scores": True,
        }
    )
    return params


def read_sound_files(
    filenames: List[str], expected_sample_rate: float
) -> List[torch.Tensor]:
    """Read a list of sound files into a list 1-D float32 torch tensors.
    Args:
      filenames:
        A list of sound filenames.
      expected_sample_rate:
        The expected sample rate of the sound files.
    Returns:
      Return a list of 1-D float32 torch tensors.
    """
    ans = []
    for f in filenames:
        wave, sample_rate = torchaudio.load(f)
        assert sample_rate == expected_sample_rate, (
            f"expected sample rate: {expected_sample_rate}. "
            f"Given: {sample_rate}"
        )
        # We use only the first channel
        ans.append(wave[0])
    return ans


def main():
    parser = get_parser()
    args = parser.parse_args()

    params = get_params()
    params.update(vars(args))
    logging.info(f"{params}")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"device: {device}")

    logging.info("Creating model")
    model = Conformer(
        num_features=params.feature_dim,
        nhead=params.nhead,
        d_model=params.attention_dim,
        num_classes=params.num_classes,
        subsampling_factor=params.subsampling_factor,
        num_decoder_layers=params.num_decoder_layers,
        vgg_frontend=params.vgg_frontend,
        is_espnet_structure=params.is_espnet_structure,
        mmi_loss=params.mmi_loss,
        use_feat_batchnorm=params.use_feat_batchnorm,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    logging.info(f"Loading HLG from {params.HLG}")
    HLG = k2.Fsa.from_dict(torch.load(params.HLG, map_location="cpu"))
    HLG = HLG.to(device)
    if not hasattr(HLG, "lm_scores"):
        # For whole-lattice-rescoring and attention-decoder
        HLG.lm_scores = HLG.scores.clone()

    if params.method in ["whole-lattice-rescoring", "attention-decoder"]:
        logging.info(f"Loading G from {params.G}")
        G = k2.Fsa.from_dict(torch.load(params.G, map_location="cpu"))
        G = G.to(device)
        # Add epsilon self-loops to G as we will compose
        # it with the whole lattice later
        G = k2.add_epsilon_self_loops(G)
        G = k2.arc_sort(G)
        G.lm_scores = G.scores.clone()

    logging.info("Constructing Fbank computer")
    opts = kaldifeat.FbankOptions()
    opts.device = device
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = params.sample_rate
    opts.mel_opts.num_bins = params.feature_dim

    fbank = kaldifeat.Fbank(opts)

    logging.info(f"Reading sound files: {params.sound_files}")
    waves = read_sound_files(
        filenames=params.sound_files, expected_sample_rate=params.sample_rate
    )
    waves = [w.to(device) for w in waves]

    logging.info(f"Decoding started")
    features = fbank(waves)

    features = pad_sequence(
        features, batch_first=True, padding_value=math.log(1e-10)
    )

    # Note: We don't use key padding mask for attention during decoding
    with torch.no_grad():
        nnet_output, memory, memory_key_padding_mask = model(features)

    batch_size = nnet_output.shape[0]
    supervision_segments = torch.tensor(
        [[i, 0, nnet_output.shape[1]] for i in range(batch_size)],
        dtype=torch.int32,
    )

    lattice = get_lattice(
        nnet_output=nnet_output,
        HLG=HLG,
        supervision_segments=supervision_segments,
        search_beam=params.search_beam,
        output_beam=params.output_beam,
        min_active_states=params.min_active_states,
        max_active_states=params.max_active_states,
        subsampling_factor=params.subsampling_factor,
    )

    if params.method == "1best":
        logging.info("Use HLG decoding")
        best_path = one_best_decoding(
            lattice=lattice, use_double_scores=params.use_double_scores
        )
    elif params.method == "whole-lattice-rescoring":
        logging.info("Use HLG decoding + LM rescoring")
        best_path_dict = rescore_with_whole_lattice(
            lattice=lattice,
            G_with_epsilon_loops=G,
            lm_scale_list=[params.ngram_lm_scale],
        )
        best_path = next(iter(best_path_dict.values()))
    elif params.method == "attention-decoder":
        logging.info("Use HLG + LM rescoring + attention decoder rescoring")
        rescored_lattice = rescore_with_whole_lattice(
            lattice=lattice, G_with_epsilon_loops=G, lm_scale_list=None
        )
        best_path_dict = rescore_with_attention_decoder(
            lattice=rescored_lattice,
            num_paths=params.num_paths,
            model=model,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
            sos_id=params.sos_id,
            eos_id=params.eos_id,
            scale=params.lattice_score_scale,
            ngram_lm_scale=params.ngram_lm_scale,
            attention_scale=params.attention_decoder_scale,
        )
        best_path = next(iter(best_path_dict.values()))

    hyps = get_texts(best_path)
    word_sym_table = k2.SymbolTable.from_file(params.words_file)
    hyps = [[word_sym_table[i] for i in ids] for ids in hyps]

    s = "\n"
    for filename, hyp in zip(params.sound_files, hyps):
        words = " ".join(hyp)
        s += f"{filename}:\n{words}\n\n"
    logging.info(s)

    logging.info(f"Decoding Done")


if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
