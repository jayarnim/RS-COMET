import torch.nn as nn
from .optimizer.registry import OPTIMIZER_REGISTRY


def optimizer_builder(
    model: nn.Module,
    cfg: dict,
):
    kwargs = dict(
        params=model.parameters(), 
        lr=cfg.lr, 
        weight_decay=cfg.weight_decay,
    )
    cls = OPTIMIZER_REGISTRY[cfg.optimizer]
    return cls(**kwargs)