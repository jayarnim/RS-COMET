from .pointwise import PointwiseEngine
from .pairwise import PairwiseEngine
from .listwise import ListwiseEngine


ENGINE_REGISTRY = {
    "pointwise": PointwiseEngine,
    "pairwise": PairwiseEngine,
    "listwise": ListwiseEngine,
}