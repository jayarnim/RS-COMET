import torch
import torch.nn as nn


# device setting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save(
    obj: nn.Module, 
    path: str,
):
    checkpoint = {
        "state_dict": obj.state_dict(),
        "init_args": obj.init_args,
    }
    torch.save(checkpoint, path)


def load(
    cls: nn.Module, 
    path: str, 
):
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    model = cls(**checkpoint["init_args"])
    model.load_state_dict(checkpoint["state_dict"])
    return model
