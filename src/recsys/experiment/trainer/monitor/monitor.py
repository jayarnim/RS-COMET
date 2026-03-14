import copy
import torch
import torch.nn as nn
from .predictor import Predictor
from .metrics_computer import MetricsComputer
from .early_stopper import EarlyStopper


# device setting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Monitor:
    def __init__(
        self,
        model: nn.Module,
        predictor: Predictor,
        metrics_computer: MetricsComputer,
        early_stopper: EarlyStopper,
    ):
        self.model = model.to(DEVICE)
        self.predictor = predictor
        self.metrics_computer = metrics_computer
        self.early_stopper = early_stopper

    def __call__(
        self,
        dev_loader: torch.utils.data.dataloader.DataLoader,
        epoch: int,
    ):
        kwargs = dict(
            dev_loader=dev_loader,
            epoch=epoch,
        )
        result = self.predictor(**kwargs)
        
        kwargs = dict(
            result=result,
        )
        score = self.metrics_computer(**kwargs)

        kwargs = dict(
            current_score=score, 
            current_epoch=epoch,
            current_state=copy.deepcopy(self.model.state_dict()),
        )
        self.early_stopper(**kwargs)

        return score

    @property
    def should_stop(self):
        return self.early_stopper.should_stop

    @property
    def counter(self):
        return self.early_stopper.counter

    @property
    def best_epoch(self):
        return self.early_stopper.best_epoch

    @property
    def best_score(self):
        return self.early_stopper.best_score

    @property
    def best_state(self):
        return self.early_stopper.best_state