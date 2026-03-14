import random
import numpy as np
import torch


def reset(cfg):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    print(f"ALL SEEDS RESET: {cfg.seed}")
