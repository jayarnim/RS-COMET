from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn


# device setting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Predictor:
    def __init__(
        self, 
        model: nn.Module,
        schema,
    ):
        self.model = model.to(DEVICE)
        self.schema = schema

    @torch.no_grad()
    def __call__(
        self,
        tst_loader: torch.utils.data.dataloader.DataLoader,
    ):
        # evaluation
        self.model.eval()

        # to save result
        user_idx_list = []
        item_idx_list = []
        label_list = []
        prediction_list = []

        iter_obj = tqdm(
            iterable=tst_loader, 
            desc=f"TST"
        )

        for user_idx, item_idx, label in iter_obj:
            # to gpu
            kwargs = dict(
                user_idx=user_idx.to(DEVICE),
                item_idx=item_idx.to(DEVICE),
            )

            # predict
            logit = self.model.predict(**kwargs)

            # to cpu & save
            user_idx_list.extend(user_idx.cpu().tolist())
            item_idx_list.extend(item_idx.cpu().tolist())
            label_list.extend(label.cpu().tolist())
            prediction_list.extend(logit.cpu().tolist())

        # list -> df
        result = pd.DataFrame(
            {
                self.schema.col_user: user_idx_list,
                self.schema.col_item: item_idx_list,
                self.schema.col_rating: label_list,
                self.schema.col_prediction: prediction_list,
            }
        )

        return result