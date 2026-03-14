from dataclasses import dataclass


@dataclass
class COMETCfg:
    num_users: int
    num_items: int
    embedding_dim: int
    channels: int
    kernel_size: list
    hidden_dim: list
    dropout: float