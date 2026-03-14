import torch
import torch.nn as nn


class ElementwiseSum(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(
        self, 
        emb: torch.Tensor, 
        pooled: torch.Tensor,
    ):
        return emb + pooled