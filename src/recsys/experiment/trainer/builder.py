import torch
import torch.nn as nn
from .trainer import Trainer
from .engine.builder import engine_builder
from .monitor.builder import monitor_builder


# device setting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def trainer_builder(
    model: nn.Module,
    cfg: dict,
):
    kwargs = dict(
        model=model,
        cfg=cfg,
    )
    engine = engine_builder(**kwargs)

    kwargs = dict(
        model=model,
        cfg=cfg,
    )
    monitor = monitor_builder(**kwargs)

    kwargs = dict(
        model=model,
        engine=engine,
        monitor=monitor,
        num_epochs=cfg.num_epochs,
    )
    return Trainer(**kwargs)