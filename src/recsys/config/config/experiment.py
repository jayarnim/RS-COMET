from dataclasses import dataclass
from ...msr.const import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_LABEL_COL,
    DEFAULT_TIMESTAMP_COL,
    DEFAULT_PREDICTION_COL,
)


@dataclass
class ExperimentCfg:
    model: str
    data: str
    strategy: str
    seed: int


@dataclass
class SchemaCfg:
    col_user: str=DEFAULT_USER_COL
    col_item: str=DEFAULT_ITEM_COL
    col_rating: str=DEFAULT_RATING_COL
    col_label: str=DEFAULT_LABEL_COL
    col_timestamp: str=DEFAULT_TIMESTAMP_COL
    col_prediction: str=DEFAULT_PREDICTION_COL


@dataclass
class PipelineCfg:
    split_ratio: dict[str, int]
    min_rating: int
    filter_by: str
    neg_ratio: dict[str, int]
    strategy: str
    batch_size: int
    shuffle: bool
    histories: dict[str, dict]
    seed: int
    num_users: int
    num_items: int
    schema: SchemaCfg


@dataclass
class TrainerCfg:
    strategy: str
    loss: str
    num_epochs: int
    optimizer: str
    lr: float
    weight_decay: float
    metric: str
    k: int
    delta: float
    patience: int
    warmup: int
    schema: SchemaCfg


@dataclass
class EvaluatorCfg:
    k: list
    schema: SchemaCfg