from typing import List

import k2
import torch

from icefall.lexicon import Lexicon


class CtcTrainingGraphCompiler(object):
    def __init__(
        self, lexicon: Lexicon, device: torch.device, oov: str = "<UNK>",
    ):
        """
        Args:
          lexicon:
            It is built from `data/lang/lexicon.txt`.
          device:
            The device to use for operations compiling transcripts to FSAs.
          oov:
            Out of vocabulary word. When a word in the transcript
            does not exist in the lexicon, it is replaced with `oov`.
        """
        L_inv = lexicon.L_inv.to(device)
        assert L_inv.requires_grad is False

        assert oov in lexicon.word_table

        self.L_inv = k2.arc_sort(L_inv)
        self.oov_id = lexicon.word_table[oov]
        self.word_table = lexicon.word_table

        max_token_id = max(lexicon.tokens)
        ctc_topo = k2.ctc_topo(max_token_id, modified=False)

        self.ctc_topo = ctc_topo.to(device)
        self.device = device

    def compile(self, texts: List[str]) -> k2.Fsa:
        """Build decoding graphs by composing ctc_topo with
        given transcripts.

        Args:
          texts:
            A list of strings. Each string contains a sentence for an utterance.
            A sentence consists of spaces separated words. An example `texts`
            looks like:

                ['hello icefall', 'CTC training with k2']

        Returns:
          An FsaVec, the composition result of `self.ctc_topo` and the
          transcript FSA.
        """
        transcript_fsa = self.convert_transcript_to_fsa(texts)

        # NOTE: k2.compose runs on CUDA only when treat_epsilons_specially
        # is False, so we add epsilon self-loops here
        fsa_with_self_loops = k2.remove_epsilon_and_add_self_loops(
            transcript_fsa
        )

        fsa_with_self_loops = k2.arc_sort(fsa_with_self_loops)

        decoding_graph = k2.compose(
            self.ctc_topo, fsa_with_self_loops, treat_epsilons_specially=False
        )

        assert decoding_graph.requires_grad is False

        return decoding_graph

    def convert_transcript_to_fsa(self, texts: List[str]) -> k2.Fsa:
        """Convert a list of transcript texts to an FsaVec.

        Args:
          texts:
            A list of strings. Each string contains a sentence for an utterance.
            A sentence consists of spaces separated words. An example `texts`
            looks like:

                ['hello icefall', 'CTC training with k2']

        Returns:
          Return an FsaVec, whose `shape[0]` equals to `len(texts)`.
        """
        word_ids_list = []
        for text in texts:
            word_ids = []
            for word in text.split(" "):
                if word in self.word_table:
                    word_ids.append(self.word_table[word])
                else:
                    word_ids.append(self.oov_id)
            word_ids_list.append(word_ids)

        word_fsa = k2.linear_fsa(word_ids_list, self.device)

        word_fsa_with_self_loops = k2.add_epsilon_self_loops(word_fsa)

        fsa = k2.intersect(
            self.L_inv, word_fsa_with_self_loops, treat_epsilons_specially=False
        )
        # fsa has word ID as labels and token ID as aux_labels, so
        # we need to invert it
        ans_fsa = fsa.invert_()
        return k2.arc_sort(ans_fsa)
