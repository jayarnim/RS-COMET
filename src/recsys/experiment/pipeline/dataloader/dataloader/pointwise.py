import random
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class NegativeSampler:
    def __init__(
        self,
        candidates: dict, 
        ratio: int,
        seed: int,
    ):
        self.candidates = candidates
        self.ratio = ratio
        self.rng = random.Random(seed)

    def __call__(self, user):
        return self.rng.choice(self.candidates[user])


class CustomizedDataset(Dataset):
    def __init__(
        self, 
        df: pd.DataFrame,
        sampler, 
        neg_ratio: int,
        schema,
    ):
        super().__init__()
        self.df = df
        self.sampler = sampler
        self.neg_ratio = neg_ratio
        self.schema = schema

        self._set_up_components()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        user, pos = self.pairs[idx // (1 + self.neg_ratio)]

        decision = (idx % (1 + self.neg_ratio) == 0)

        if decision==True:
            return user, pos, 1
        else:
            neg = self.sampler(user)
            return user, neg, 0

    def _set_up_components(self):
        zip_obj = zip(self.df[self.schema.col_user], self.df[self.schema.col_item])
        self.pairs = list(zip_obj)
        self.num_samples = len(self.pairs) * (1 + self.neg_ratio)


def _pointwise_collate_fn(batch):
    user_list, item_list, label_list = zip(*batch)
    
    user_batch = torch.tensor(user_list, dtype=torch.long)
    item_batch = torch.tensor(item_list, dtype=torch.long)
    label_batch = torch.tensor(label_list, dtype=torch.float32)
    
    return user_batch, item_batch, label_batch


def pointwise_dataloader(
    df: pd.DataFrame,
    neg_candidates: dict,
    neg_ratio: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
    schema,
):
    kwargs = dict(
        candidates=neg_candidates, 
        ratio=neg_ratio,
        seed=seed,
    )
    sampler = NegativeSampler(**kwargs)

    kwargs = dict(
        df=df,
        sampler=sampler, 
        neg_ratio=neg_ratio,
        schema=schema,    
    )
    dataset = CustomizedDataset(**kwargs)

    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_pointwise_collate_fn,
    )
    dataloader = DataLoader(**kwargs)

    return dataloader