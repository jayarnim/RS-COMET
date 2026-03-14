import torch
import torch.nn as nn
from .predictor import Predictor
from .metrics_computer import MetricsComputer


# device setting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Evaluator:
    def __init__(
        self,
        model: nn.Module,
        predictor: Predictor,
        metrics_computer: MetricsComputer,
    ):
        self.model = model.to(DEVICE)
        self.predictor = predictor
        self.metrics_computer = metrics_computer

    def __call__(
        self,
        tst_loader: torch.utils.data.dataloader.DataLoader,
    ):
        kwargs = dict(
            tst_loader=tst_loader,
        )
        result = self.predictor(**kwargs)

        kwargs = dict(
            result=result,
        )
        metrics_sheet = self.metrics_computer(**kwargs)

        return result, metrics_sheet