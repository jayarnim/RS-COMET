import torch
import torch.nn as nn


class HistoryIDXViewer(nn.Module):
    def __init__(
        self,
        histories: torch.Tensor,
        padding_idx: int,
    ):
        super().__init__()

        # global attr
        self.register_buffer("histories", histories)
        self.padding_idx = padding_idx

    def forward(
        self, 
        anchor_idx: torch.Tensor,
        target_idx: torch.Tensor, 
    ):
        # history idx of anchor
        hist_idx_slice = self.histories[anchor_idx]

        # generate mask
        kwargs = dict(
            hist_idx=hist_idx_slice,
            target_idx=target_idx, 
        )
        mask = self._mask_generator(**kwargs)

        # replace target idx -> padding idx
        kwargs = dict(
            input=hist_idx_slice,
            mask=~mask,
            value=self.padding_idx,
        )
        hist_idx_slice_padded = torch.masked_fill(**kwargs)

        return hist_idx_slice_padded

    def _mask_generator(self, hist_idx, target_idx):
        # mask to current target from history
        marking_target_idx = hist_idx==target_idx.unsqueeze(1)
        # mask to padding
        marking_padding_idx = hist_idx == self.padding_idx
        # final mask
        mask = ~(marking_target_idx | marking_padding_idx)
        return mask