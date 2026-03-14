from IPython.display import clear_output
import torch
import torch.nn as nn


# device setting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(
        self, 
        model: nn.Module,
        engine,
        monitor,
        num_epochs,
    ):
        self.model = model.to(DEVICE)
        self.engine = engine
        self.monitor = monitor
        self.num_epochs = num_epochs

    def fit(
        self, 
        trn_loader: torch.utils.data.dataloader.DataLoader, 
        val_loader: torch.utils.data.dataloader.DataLoader, 
        dev_loader: torch.utils.data.dataloader.DataLoader, 
    ):
        kwargs = dict(
            trn_loader=trn_loader, 
            val_loader=val_loader, 
            dev_loader=dev_loader, 
        )
        trn_log_list, val_log_list, dev_log_list = self._progressor(**kwargs)

        clear_output(wait=False)

        kwargs = dict(
            trn_log_list=trn_log_list, 
            val_log_list=val_log_list, 
            dev_log_list=dev_log_list,
        )
        return self._finalizer(**kwargs)

    def _progressor(self, trn_loader, val_loader, dev_loader):
        trn_log_list = []
        val_log_list = []
        dev_log_list = []

        for epoch in range(self.num_epochs):
            # trn, val
            kwargs = dict(
                trn_loader=trn_loader, 
                val_loader=val_loader, 
                epoch=epoch,
            )
            trn_loss, val_loss = self._run_engine(**kwargs)

            # dev
            kwargs = dict(
                dev_loader=dev_loader,
                epoch=epoch,
            )
            dev_score = self._run_monitor(**kwargs)

            # accumulate
            trn_log_list.append(trn_loss)
            val_log_list.append(val_loss)
            dev_log_list.append(dev_score)

            # early stopping
            if self.monitor.should_stop==True:
                break

            # log reset
            if (epoch + 1) % 50 == 0:
                clear_output(wait=False)

        return trn_log_list, val_log_list, dev_log_list

    def _finalizer(self, trn_log_list, val_log_list, dev_log_list):
        if self.monitor.best_state is not None:
            self.model.load_state_dict(self.monitor.best_state)

        print(
            "DEVELOPMENT",
            f"\tBEST SCORE: {self.monitor.best_score:.4f}",
            f"\tBEST EPOCH: {self.monitor.best_epoch}",
            sep="\n",
        )

        return dict(
            trn=trn_log_list,
            val=val_log_list,
            dev=dev_log_list,
        )

    def _run_engine(self, trn_loader, val_loader, epoch):
        kwargs = dict(
            trn_loader=trn_loader, 
            val_loader=val_loader, 
            epoch=epoch,
        )
        trn_loss, val_loss = self.engine(**kwargs)

        print(
            f"TRN LOSS: {trn_loss:.4f}",
            f"VAL LOSS: {val_loss:.4f}",
            sep='\n'
        )

        return trn_loss, val_loss

    def _run_monitor(self, dev_loader, epoch):
        kwargs = dict(
            dev_loader=dev_loader, 
            epoch=epoch,
        )
        dev_score = self.monitor(**kwargs)

        print(
            f"CURRENT METRIC: {dev_score:.4f}",
            f"BEST METRIC: {self.monitor.best_score:.4f}",
            f"COUNTER: {self.monitor.counter}",
            sep='\t',
        )

        return dev_score