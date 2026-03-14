import torch
import torch.nn.functional as F


def bpr(
    pos: torch.Tensor, 
    neg: torch.Tensor,
):
    diff = pos - neg
    return -F.logsigmoid(diff).mean()