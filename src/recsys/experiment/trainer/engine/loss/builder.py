from .loss.registry import LOSS_FN_REGISTRY


def loss_fn_builder(cfg):
    return LOSS_FN_REGISTRY[cfg.strategy][cfg.loss]