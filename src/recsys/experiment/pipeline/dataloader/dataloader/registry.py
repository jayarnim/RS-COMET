from .pointwise import pointwise_dataloader
from .pairwise import pairwise_dataloader
from .listwise import listwise_dataloader


DATALOADER_REGISTRY = {
    "pointwise": pointwise_dataloader,
    "pairwise": pairwise_dataloader,
    "listwise": listwise_dataloader,
}