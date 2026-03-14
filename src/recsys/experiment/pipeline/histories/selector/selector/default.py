import torch


def default_selector(
    interactions: torch.Tensor,
    **kwargs,
):
    rows, cols = interactions.nonzero(as_tuple=True)

    hist_indices = [[] for _ in range(len(interactions))]
    for r, c in zip(rows.tolist(), cols.tolist()):
        hist_indices[r].append(c)

    hist_indices = [
        torch.tensor(indices, dtype=torch.long)
        for indices in hist_indices
    ]

    return hist_indices