import torch.nn as nn
from .monitor import Monitor
from .predictor import Predictor
from .metrics_computer import MetricsComputer
from .early_stopper import EarlyStopper
from .metric.builder import metric_builder


def monitor_builder(
    model: nn.Module,
    cfg,
):
    kwargs = dict(
        model=model,
        schema=cfg.schema,
    )
    predictor = Predictor(**kwargs)

    kwargs = dict(
        cfg=cfg,
    )
    criterion = metric_builder(**kwargs)

    kwargs = dict(
        criterion=criterion,
        k=cfg.k,
        schema=cfg.schema,
    )
    metrics_computer = MetricsComputer(**kwargs)

    kwargs = dict(
        delta=cfg.delta,
        patience=cfg.patience,
        warmup=cfg.warmup,
    )
    early_stopper = EarlyStopper(**kwargs)

    kwargs = dict(
        model=model,
        predictor=predictor,
        metrics_computer=metrics_computer,
        early_stopper=early_stopper,
    )
    return Monitor(**kwargs)