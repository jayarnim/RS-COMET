from ..config.experiment import (
    SchemaCfg,
    ExperimentCfg,
    PipelineCfg,
    TrainerCfg,
    EvaluatorCfg,
)
from ...msr.const import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_LABEL_COL,
    DEFAULT_TIMESTAMP_COL,
    DEFAULT_PREDICTION_COL,
)


def experiment(cfg):
    return ExperimentCfg(
        model=cfg["model"]["name"],
        data=cfg["data"]["name"],
        strategy=cfg["trainer"]["strategy"],
        seed=cfg["seed"],
    )


def schema(*args):
    return SchemaCfg(
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        col_rating=DEFAULT_RATING_COL,
        col_label=DEFAULT_LABEL_COL,
        col_timestamp=DEFAULT_TIMESTAMP_COL,
        col_prediction=DEFAULT_PREDICTION_COL,
    )


def pipeline(cfg):
    return PipelineCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        split_ratio=cfg["split"]["ratio"],
        min_rating=cfg["split"]["min_rating"],
        filter_by=cfg["split"]["filter_by"],
        neg_ratio=cfg["negatives"]["ratio"],
        strategy=cfg["trainer"]["strategy"],
        batch_size=cfg["dataloader"]["batch_size"],
        shuffle=cfg["dataloader"]["shuffle"],
        histories=cfg["histories"],
        seed=cfg["seed"],
        schema=schema(),
    )


def trainer(cfg):
    return TrainerCfg(
        strategy=cfg["trainer"]["strategy"],
        loss=cfg["trainer"]["loss"],
        num_epochs=cfg["trainer"]["num_epochs"],
        optimizer=cfg["optimizer"]["name"],
        lr=cfg["optimizer"]["lr"],
        weight_decay=cfg["optimizer"]["weight_decay"],
        metric=cfg["monitor"]["metric"],
        k=cfg["monitor"]["k"],
        delta=cfg["monitor"]["delta"],
        patience=cfg["monitor"]["patience"],
        warmup=cfg["monitor"]["warmup"],
        schema=schema(),
    )


def evaluator(cfg):
    return EvaluatorCfg(
        k=cfg["evaluator"]["k"],
        schema=schema(),
    )