import torch
from sklearn.feature_extraction.text import TfidfTransformer


def tfidf_selector(
    interactions: torch.Tensor,
    max_hist: int,
):
    # padding idx
    num_anchor, num_target = interactions.shape

    # compute tfidf
    tfidf = TfidfTransformer(norm=None)
    tfidf_matrix = tfidf.fit_transform(interactions)

    # ndarray -> tensor
    kwargs = dict(
        data=tfidf_matrix.toarray(),
        dtype=torch.float32,
    )
    tfidf_matrix_dense = torch.tensor(**kwargs)

    # select top-k indices
    topk_indices = []
    for row in range(len(tfidf_matrix_dense)):
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
            vals, indices = torch.topk(tfidf_matrix_dense[row], k=max_hist)
            topk_indices.append(indices.to(torch.long))

    return topk_indices