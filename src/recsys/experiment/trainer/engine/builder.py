import torch.nn as nn
from .engine.registry import ENGINE_REGISTRY
from .optimizer.builder import optimizer_builder
from .loss.builder import loss_fn_builder


def engine_builder(
    model: nn.Module,
    cfg,
):
    kwargs = dict(
        model=model,
        cfg=cfg,
    )
    optimizer = optimizer_builder(**kwargs)

    kwargs = dict(
        cfg=cfg,
    )
    criterion = loss_fn_builder(**kwargs)

    kwargs = dict(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
    )
    cls = ENGINE_REGISTRY[cfg.strategy]
    return cls(**kwargs)