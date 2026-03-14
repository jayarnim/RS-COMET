import torch
from torch.nn.utils.rnn import pad_sequence
from .selector.builder import selector_builder


def histories_generator(
    interactions: torch.Tensor, 
    selector: str,
    max_hist: int,
):
    # drop padding idx
    interactions_unpadded = interactions[:-1, :-1]

    # padding idx
    num_anchor, num_target = interactions_unpadded.shape

    # generate selector
    kwargs = dict(
        name=selector,
    )
    selector = selector_builder(**kwargs)

    # select hist per anchor    
    kwargs = dict(
        interactions=interactions_unpadded,
        max_hist=max_hist,
    )
    hist_indices = selector(**kwargs)

    # padding
    kwargs = dict(
        sequences=hist_indices, 
        batch_first=True, 
        padding_value=num_target,
    )
    hist_indices_padded = pad_sequence(**kwargs)

    return hist_indices_padded