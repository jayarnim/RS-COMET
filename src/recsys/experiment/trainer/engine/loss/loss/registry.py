from .pointwise import bce
from .pairwise import bpr
from .listwise import climf


POINTWISE_REGISTRY = {
    "bce": bce,
}

PAIRWISE_REGISTRY = {
    "bpr": bpr,
}

LISTWISE_REGISTRY = {
    "climf": climf,
}

LOSS_FN_REGISTRY = {
    "pointwise": POINTWISE_REGISTRY,
    "pairwise": PAIRWISE_REGISTRY,
    "listwise": LISTWISE_REGISTRY,
}