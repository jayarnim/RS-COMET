import torch.nn as nn
from .evaluator import Evaluator
from .predictor import Predictor
from .metrics_computer import MetricsComputer


def evaluator_builder(
    model: nn.Module, 
    cfg,
):
    kwargs = dict(
        model=model,
        schema=cfg.schema,
    )
    predictor = Predictor(**kwargs)

    kwargs = dict(
        k=cfg.k,
        schema=cfg.schema,
    )
    metrics_computer = MetricsComputer(**kwargs)
    
    kwargs = dict(
        model=model,
        predictor=predictor,
        metrics_computer=metrics_computer,
    )
    return Evaluator(**kwargs)