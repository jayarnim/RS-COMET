from ......msr.python_evaluation import (
    hit_ratio_at_k,
    precision_at_k, 
    recall_at_k,
    map_at_k, 
    ndcg_at_k, 
)


METRIC_REGISTRY = {
    "hit_ratio": hit_ratio_at_k,
    "precision": precision_at_k,
    "recall": recall_at_k,
    "map": map_at_k,
    "ndcg": ndcg_at_k,
}