import torch
import torch.nn as nn


class MatrixFactorizationLayer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(
        self, 
        user_emb: torch.Tensor, 
        item_emb: torch.Tensor,
    ):
        return user_emb * item_emb