import collections


class EarlyStopper:
    def __init__(
        self,
        delta: float,
        patience: int,
        warmup: int,
    ):
        self.delta = delta
        self.patience = patience
        self.warmup = warmup

        self._set_up_components()

    def __call__(
        self,
        current_score: float,
        current_epoch: int,
        current_state: collections.OrderedDict,
    ):
        IMPROVED = current_score > self.best_score + self.delta
        
        if (current_epoch+1) <= self.warmup:
            self.should_stop = False
            self.counter = 0
        
        else:
            if IMPROVED:
                self.best_score = current_score
                self.best_epoch = current_epoch + 1
                self.best_state = current_state
                self.counter = 0
            else:
                self.counter += 1

        if self.counter > self.patience:
            self.should_stop = True

    def _set_up_components(self):
        self.best_epoch = 0
        self.best_score = -float("inf")
        self.best_state = None
        self.counter = 0
        self.should_stop = False