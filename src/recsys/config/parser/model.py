from ..config.model import (
    COMETCfg,
)


def auto(cfg):
    model = cfg["model"]["name"]
    if model=="comet":
        return comet(cfg)
    else:
        raise ValueError("invalid model name in .yaml config")


def comet(cfg):
    return COMETCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        embedding_dim=cfg["model"]["embedding_dim"],
        channels=cfg["model"]["channels"],
        kernel_size=cfg["model"]["kernel_size"],
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout=cfg["model"]["dropout"],
    )

