import torch
from .histories import histories_generator


def histories_builder(
    interactions: torch.Tensor, 
    cfg,
):
    histories = dict()

    kwargs = dict(
        interactions=interactions,
        selector=cfg.histories["user"]["selector"],
        max_hist=cfg.histories["user"]["max_hist"],
    )
    histories["user"] = histories_generator(**kwargs)

    kwargs = dict(
        interactions=interactions.T,
        selector=cfg.histories["item"]["selector"],
        max_hist=cfg.histories["item"]["max_hist"],
    )
    histories["item"] = histories_generator(**kwargs)

    return histories