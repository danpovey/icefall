import torch
from torch import Tensor
from torch import nn
from typing import Tuple, Optional



def indexes_with_dups(batch_size: int, num_to_duplicate: int, **kwargs) -> Tensor:
    """
    Returns a Tensor containing indexes like [ 0 1 1 2 3 3 4 4 5 ], where, randomly,
    a certain proportion of the indexes are repeated twice and the remaining indexes
    are repeated once.

    Args:
      batch_size: the original batch size
      num_to_duplicate: the number of elements of the batch to duplicate
      kwargs: provided so you can add "device=something.device"
    Returns:
      a Tensor containing something like [ 0 1 1 2 3 3 4 4 5 ], with some indexes
      randomly duplicated.  At least one batch element will be duplicated.

      The caller can then do something like:
        x = torch.index_select(x, dim=0, index=ret)
      where ret is the return value of this function
    """
    return torch.cat((torch.arange(batch_size, **kwargs),
               torch.randperm(batch_size, **kwargs)[:num_to_duplicate]),
              dim=0).sort(dim=0)[0]

    # note: implementation should use randperm(), we don't want the number of indexes
    # duplicated to be random, or this will increase maximum memory used.



def loss_prediction_loss(loss: Tensor, pred_loss: Tensor, dup_indexes: Tensor) -> Tensor:
    """
    Computes a loss value that captures the accuracy of our prediction of loss
    values differences between different duplicates of a particular batch
    element.

    Args:
       loss: a Tensor of shape (aug_batch_size,), representing for instance
         the main loss used for training, where aug_batch_size is the length of
         the Tensor returned by indexes_with_dups().
         CAUTION: you should probably detach this before passing it in.
      pred_loss: a Tensor of shape (aug_batch_size,), likely returned from
         the model itself.
      dup_indexes: a Tensor of indexes like [ 0 1 1 2 3 3 4 5 ] where some
         nonempty subset of indexes are duplicated.

    Returns:
          a scalar >= 0 and probably less than 1, is the ratio between
          the sum-square inaccuracy of prediction of the differences between the loss values
          within the batch, and the sum-square of the differences in loss values themselves.
    """
    aug_batch_size, = loss.shape
    assert loss.shape == dup_indexes.shape

    repeat_indexes1 = torch.where(dup_indexes[:-1] == dup_indexes[1:])[0]
    repeat_indexes2 = repeat_indexes1 + 1
    print("dup_indexes = ", dup_indexes)
    print("ri1 = ", repeat_indexes1)

    def get_sumsq_diffs(x):
        return ((x[repeat_indexes2] - x[repeat_indexes1]) ** 2).sum()

    print("loss - pred_loss = ", loss - pred_loss)
    print("pred_loss = ", pred_loss)
    print("loss = ", loss)
    diff_loss = get_sumsq_diffs(loss - pred_loss) / get_sumsq_diffs(loss)
    return diff_loss


def deduplicate(loss: Tensor, dup_indexes: Tensor) -> Tensor:

    batch_size = dup_indexes[-1].item() + 1  # orig batch size

    # index_counts: e.g. [ 1, 1, 2, 1, 2, 1, 1, 1 ],
    # num times each original index was repeated
    index_counts = torch.zeros_like(dup_indexes[:batch_size])

    index_counts.index_add_(dim=0, index=dup_indexes,
                            source=torch.ones_like(dup_indexes))

    loss_dedup = torch.zeros_like(loss[:batch_size])
    loss_dedup.index_add_(dim=0, index=dup_indexes, source=loss)
    loss_dedup = loss_dedup / index_counts
    return loss_dedup



class ReduceFrames(nn.Module):
      """
      Reduce the number of frames by selectively discarding some frames.  Also returns an
      estimate of the final loss value that you are training with, e.g. transducer loss or
      whatever it is (this will eventually be used to predict differences in the final loss
      value in cases where we are processing duplicates of the same sequence).

      Args:
         num_channels: the number of channels in the input
         max_length_proportion: e.g. 0.75; the maximum proportion of the input frames
                 that we are 'allowed' to keep (remaining frames will be truncated,
                 even if they were chosen.
         score_stddev:  Target for normalizing the standard deviation of predicted
                delta-loss values, before we use them to choose which frames to
                keep.
      """
      def __init__(self, num_channels: int,
                   max_length_proportion: float,
                   score_stddev: float = 3.0):
          super().__init__()
          # to_loss projects from the input num-channels to two values that represent
          # the predicted loss in the case where we either (keep separate, or aggregate)
          # this pair of frames.
          self.to_loss = nn.Linear(num_channels, 2)
          # is_separate is a kind of indicator tensor that lets later modeling
          # stages know that we kept the two things separate.
          self.is_separate = nn.Parameter(0.01 * torch.zeros(num_channels)) #nn.Parameter(0.01 * torch.randn(num_channels))
          self.max_length_proportion = max_length_proportion
          self.score_stddev = score_stddev

      def forward(self,
                  x: Tensor,
                  lengths: Tensor,
                  ) -> Tuple[Tensor, Tensor, Tensor]:
          """
          Args:
            x: of shape (num_frames, batch_size, num_channels), the input features
        lengths: of shape (batch_size,), containing the actual number of frames for each sequence,
                not counting padding.  All elements should be <= num_frames.

          Returns: (y, new_lengths, seq_losses),
              y: of shape (new_num_frames, batch_size, num_channels)
              new_lengths: of shape (batch_size,)
              seq_losses: of shape (batch_size,)
          seq_losses are estimated loss values.  These are returned for purposes of training
          the part of the model which predicts the loss values as a function of which
          frames we keep vs. drop.  It will contribute to an extra loss term.
          """
          pred_losses = self.to_loss(x[0::2])  # the 0::2 takes every other frame.
          # pred_losses: (num_frames, batch_size, 2)
          #  1st one is pred loss if we keep the 2 frames separate, 2nd is if we aggregate them.
          (num_frames, batch_size, num_channels) = x.shape
          assert num_frames % 2 == 0
          # this is the max number of frames that we allow.. because of truncation, we expect that the
          # model will learn not to exceed this.
          new_num_frames = int(1 + num_frames * self.max_length_proportion)

          def normalize_stddev(x):
              x_std = (torch.mean(x ** 2, dim=0, keepdim=True) + 1.0e-20) ** 0.5
              # the - sign is because we want to change the sign so that the choice with
              # the higher predicted loss will be less likely to be chosen.
              return x * -self.score_stddev / x_std

          # get rid of any offset when normalizing the stddev; we only care
          # about the difference between the two scores (this affects the
          # computation of rms value)
          scores = normalize_stddev(pred_losses - torch.mean(pred_losses, dim=-1, keepdim=True))
          print("scores = ", scores)
          probs = torch.softmax(scores, dim=-1)
          print("probs = ", probs)
          # now scores: (num_frames, batch_size, 2)
          separate_probs = probs[..., 0:1]
          print("separate_probs = ", separate_probs[..., 0].t())
          kept = separate_probs > torch.rand_like(separate_probs)
          print("kept = ", kept[..., 0].t())
          # kept is a Bool array of shape (num_frames, batch_size, 1),
          # with True if we are going to keep separate the two

          seq_losses = torch.gather(pred_losses, dim=2,
                                    index=torch.logical_not(kept).to(torch.int64))
          seq_losses = seq_losses[..., 0].sum(dim=0)
          # seq_losses: (batch_size,)

          assert kept.shape == (num_frames // 2, batch_size, 1)

          lengths_mask = (torch.arange(num_frames, dtype=torch.int64, device=x.device).reshape(num_frames, 1, 1) <
                          lengths.reshape(1, batch_size, 1))
          ones = lengths_mask
          # ones will contain 1 in positions that are not masked due to being out-of-range w.r.t. the
          # sequence lengths; and 0 otherwise.
          # ones shape:  (num_frames, batch_size, 1)

          # For odd-numbered frames where 'kept' is false, set 'ones' to zero.
          ones = ones.to(torch.int64)
          ones[0::2] *= kept.to(torch.int64)

          indexes = torch.cumsum(ones, dim=0)
          new_lengths = indexes[-1, :, 0].clamp(max=new_num_frames)
          indexes = indexes - ones
          # 'indexes' now gives the destination frame of each source frame.
          # Caution: some of them may be out of range, i.e. >= new_num_frames.

          ones_float = ones.to(x.dtype)
          print("ones_float = ", ones_float[..., 0].t())
          scales = ones_float.reshape(num_frames // 2, 2, batch_size, 1).to(x.dtype).mean(
              dim=1, keepdim=True).expand(num_frames // 2, 2, batch_size, 1).reshape(
                  num_frames, batch_size, 1)
          # now scales will be 1.0 for pairs of frames where kept == True, and 0.5 for
          # pairs of frames where kept == False
          scales[indexes >= new_num_frames] = 0.0
          print("scales = ", scales[..., 0].t())

          # set the scales to zero for source frames whose destination-frame is out
          # of bounds.  Then we can clamp the indexes to new_num_frames-1 and not worry
          # about invalid data being included.
          indexes.clamp_(max=new_num_frames-1)
          print("new_num_frames = ", new_num_frames)
          print("indexes = ", indexes[..., 0].t())

          y = torch.zeros(new_num_frames, batch_size, num_channels,
                          device=x.device, dtype=x.dtype)
          y.scatter_add_(dim=0,
                         index=indexes.expand(num_frames, batch_size, num_channels),
                         src=x * scales)

          # is_separate will be 1.0 for "new" frame indexes that correspond to positions
          # where we kept frames separate, and 0.0 for positions where we merged them.
          is_separate = torch.zeros(new_num_frames, batch_size, 1,
                                    device=x.device, dtype=x.dtype)

          kept_dup = kept.reshape(num_frames // 2, 1, batch_size, 1).expand(
              num_frames // 2, 2, batch_size, 1).reshape(num_frames, batch_size, 1).to(torch.float)
          kept_dup = (kept_dup * lengths_mask).to(torch.float)

          is_separate.scatter_add_(dim=0, index=indexes,
                                   #reduce="min", include_self=False,
                                   src=kept_dup)
          print("is_separate = ", is_separate[..., 0].t())
          # add an indicator of whether we kept frames separate.
          y = y + is_separate * self.is_separate
          return y, new_lengths, seq_losses



#https://pytorch.org/docs/stable/generated/torch.nan_to_num.html

def _test_reduce_frames():

    num_channels = 10
    m = ReduceFrames(num_channels, 0.7)

    num_frames = 16
    batch_size = 2
    x = -4000 + (1000 * torch.arange(num_frames).reshape(num_frames, 1, 1) +
         (100 * torch.arange(batch_size)).reshape(1, batch_size, 1) +
         (10 * torch.arange(num_channels)))
    lengths = torch.tensor( [ 16, 10 ], dtype=torch.int64)
    x = x.to(torch.float)
    print("x = ", x)
    y, new_lengths, seq_losses  = m(x, lengths)
    print("y = ", y)
    print("new_lengths = ", new_lengths)
    print("seq_losses = ", seq_losses)


def _test_dup_indexes():
    ind = indexes_with_dups(20, 10)
    print("indexes_with_dups = ", ind)

    losses = torch.rand(ind.numel())
    pred_losses = 0.2 * torch.rand(ind.numel())  + 0.8 * losses

    lploss = loss_prediction_loss(losses, pred_losses, ind)

    print("lploss = ", lploss)

    losses_dedup = deduplicate(losses, ind)
    print("losses_dedup = ", losses_dedup)





if __name__ == '__main__':
    _test_reduce_frames()
    _test_dup_indexes()
