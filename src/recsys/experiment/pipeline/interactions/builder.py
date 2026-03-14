import torch
import pandas as pd


def interactions_builder(
    df: pd.DataFrame, 
    cfg,
):
    kwargs = dict(
        size=(cfg.num_users+1, cfg.num_items+1),
        dtype=torch.int32,
    )
    user_item_matrix = torch.zeros(**kwargs)

    kwargs = dict(
        data=df[cfg.schema.col_user].values, 
        dtype=torch.long,
    )
    user_indices = torch.tensor(**kwargs)
    
    kwargs = dict(
        data=df[cfg.schema.col_item].values, 
        dtype=torch.long,
    )
    item_indices = torch.tensor(**kwargs)

    user_item_matrix[user_indices, item_indices] = 1

    return user_item_matrix