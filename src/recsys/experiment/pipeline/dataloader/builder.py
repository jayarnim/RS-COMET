import pandas as pd
from ....msr.python_splitters import python_stratified_split
from .dataloader.registry import DATALOADER_REGISTRY


def _stratified_splitter(df, cfg):
    split_name = list(cfg.split_ratio.keys())
    split_val = list(cfg.split_ratio.values())

    # for dev data set
    dev = (
        df
        .groupby(cfg.schema.col_user)
        .sample(n=1, random_state=cfg.seed)
        .sort_values(by=cfg.schema.col_user)
        .reset_index(drop=True)
    )

    # for trn, val, tst data set
    trn_val_tst = (
        df[~df[[cfg.schema.col_user, cfg.schema.col_item]]
        .apply(tuple, axis=1)
        .isin(set(dev[[cfg.schema.col_user, cfg.schema.col_item]]
        .apply(tuple, axis=1)))]
        .reset_index(drop=True)
    )

    # trn_val_tst -> [trn, val, tst]
    kwargs = dict(
        data=trn_val_tst,
        ratio=split_val,
        min_rating=cfg.min_rating,
        filter_by=cfg.filter_by,
        seed=cfg.seed,
        col_user=cfg.schema.col_user,
        col_item=cfg.schema.col_item,
    )
    split_list = python_stratified_split(**kwargs)

    split_dict = dict(zip(split_name, split_list))
    split_dict["dev"] = dev

    return split_dict


def _neg_candidates_generator(df, cfg):
    user_list = sorted(df[cfg.schema.col_user].unique())
    item_list = sorted(df[cfg.schema.col_item].unique())

    pos_per_user = {
        user: set(df.loc[df[cfg.schema.col_user]==user, cfg.schema.col_item].tolist())
        for user in user_list
    }

    neg_per_user = {
        user: list(set(item_list) - pos_per_user[user])
        for user in user_list
    }

    return neg_per_user


def _dataloader_generator(split_dict, neg_candidates, cfg):
    dataloader_dict = dict()

    for name, val in split_dict.items():
        cls = (
            DATALOADER_REGISTRY[cfg.strategy]
            if name in ["trn", "val"]
            else DATALOADER_REGISTRY["pointwise"]
        )

        NEG_RATIO = (
            cfg.neg_ratio["optimize"]
            if name in ["trn", "val"]
            else cfg.neg_ratio["measure"]
        )

        kwargs = dict(
            df=val, 
            neg_candidates=neg_candidates,
            neg_ratio=NEG_RATIO, 
            batch_size=cfg.batch_size, 
            shuffle=cfg.shuffle,
            seed=cfg.seed,
            schema=cfg.schema,
        )
        dataloader = cls(**kwargs)
        dataloader_dict[name] = dataloader

    return dataloader_dict


def dataloader_builder(
    df: pd.DataFrame,
    cfg,
):    
    # split original data
    kwargs = dict(
        df=df,
        cfg=cfg,
    )
    split_dict = _stratified_splitter(**kwargs)

    kwargs = dict(
        df=df,
        cfg=cfg,
    )
    neg_candidates = _neg_candidates_generator(**kwargs)

    # generate data loaders
    kwargs = dict(
        split_dict=split_dict, 
        neg_candidates=neg_candidates,
        cfg=cfg,
    )
    dataloader_dict = _dataloader_generator(**kwargs)

    return dataloader_dict