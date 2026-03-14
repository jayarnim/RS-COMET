import torch.optim as optim


OPTIMIZER_REGISTRY = {
    "adamw": optim.AdamW,
    "adam": optim.Adam,
    "adagrad": optim.Adagrad,
}