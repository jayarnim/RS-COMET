from .metric.registry import METRIC_REGISTRY


def metric_builder(cfg):
    return METRIC_REGISTRY[cfg.metric]