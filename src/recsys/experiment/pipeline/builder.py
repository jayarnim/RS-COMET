import pandas as pd
from .dataloader.builder import dataloader_builder
from .histories.builder import histories_builder
from .interactions.builder import interactions_builder


def pipeline_builder(
    df: pd.DataFrame,
    cfg,
):
    kwargs = dict(
        df=df,
        cfg=cfg,
    )
    dataloaders = dataloader_builder(**kwargs)

    df_trn = dataloaders["trn"].dataset.df

    kwargs = dict(
        df=df_trn,
        cfg=cfg,
    )
    interactions = interactions_builder(**kwargs)

    kwargs = dict(
        interactions=interactions,
        cfg=cfg,
    )
    histories = histories_builder(**kwargs)

    return dataloaders, interactions, histories