import torch


def freq_selector(
    interactions: torch.Tensor,
    max_hist: int,
):
    # padding idx
    num_anchor, num_target = interactions.shape

    freq_target = interactions.sum(dim=0)

    # select top-k indices
    topk_indices = []
    for row in range(len(interactions)):
        hist_count = int(interactions[row].sum().item())
        # padding only
        if hist_count == 0:
            indices = torch.tensor([num_target], dtype=torch.long)
            topk_indices.append(indices.to(torch.long))
        # all
        elif hist_count <= max_hist:
            indices = interactions[row].nonzero(as_tuple=True)[0]
            topk_indices.append(indices.to(torch.long))
        # top-k selection
        else:
            hist_idx = interactions[row].nonzero(as_tuple=True)[0]
            scores = freq_target[hist_idx]
            vals, indices = torch.topk(scores, k=max_hist)
            topk_indices.append(indices.to(torch.long))

    return topk_indices