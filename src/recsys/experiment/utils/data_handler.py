from typing import Optional
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocessor(
    df: pd.DataFrame,
    cfg,
    col_user: str,
    col_item: str,
    col_rating: Optional[str]=None,
    col_timestamp: Optional[str]=None,
):
    ORIGIN_COL_LIST = [col_user, col_item]
    RENAME_COL_LIST = [cfg.col_user, cfg.col_item]

    if col_rating is not None:
        ORIGIN_COL_LIST.append(col_rating)
        RENAME_COL_LIST.append(cfg.col_rating)

    if col_timestamp is not None:
        ORIGIN_COL_LIST.append(col_timestamp)
        RENAME_COL_LIST.append(cfg.col_timestamp)

    RENAMES = dict(zip(ORIGIN_COL_LIST, RENAME_COL_LIST))
    
    df = df[ORIGIN_COL_LIST]
    df = df.rename(columns=RENAMES)

    encoder = dict(
        user=LabelEncoder(),
        item=LabelEncoder(),
    )
    df[cfg.col_user] = encoder["user"].fit_transform(df[cfg.col_user])
    df[cfg.col_item] = encoder["item"].fit_transform(df[cfg.col_item])

    return df, encoder


def description(
    df: pd.DataFrame, 
    cfg,
    percentaile: float=0.9,
):
    user_counts = df[cfg.col_user].value_counts()
    item_counts = df[cfg.col_item].value_counts()

    N_USERS = df[cfg.col_user].nunique()
    N_ITEMS = df[cfg.col_item].nunique()
    TOTAL_INTERACTION = len(df)
    DENSITY = df.shape[0] / (N_USERS * N_ITEMS)
    MAX_USER_INTERACTION = user_counts.max()
    MAX_ITEM_INTERACTION = item_counts.max()
    TOP_PERCENTAILE_USER_INTERACTION = user_counts.quantile(percentaile)
    TOP_PERCENTAILE_ITEM_INTERACTION = item_counts.quantile(percentaile)

    print(
        f"number of user: {N_USERS}",
        f"number of item: {N_ITEMS}",
        f"total interaction: {TOTAL_INTERACTION}",
        f"interaction density: {DENSITY * 100:.4f} %",
        f"max interaction of user: {MAX_USER_INTERACTION}",
        f"max interaction of item: {MAX_ITEM_INTERACTION}",
        f"top {(1-percentaile) * 100:.1f} % interaction of user: {TOP_PERCENTAILE_USER_INTERACTION:.1f}",
        f"top {(1-percentaile) * 100:.1f} % interaction of item: {TOP_PERCENTAILE_ITEM_INTERACTION:.1f}",
        f"mean interaction of user: {TOTAL_INTERACTION // N_USERS}",
        f"mean interaction of item: {TOTAL_INTERACTION // N_ITEMS}",
        sep="\n",
    )
