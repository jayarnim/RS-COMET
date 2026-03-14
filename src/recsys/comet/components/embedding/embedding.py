import torch
import torch.nn as nn
from .viewer import HistoryIDXViewer


class IDXEmbeddingWithHistory(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        histories: dict[str, torch.Tensor],
    ):
        super().__init__()

        # global attr
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.histories = histories

        # generate layers
        self._set_up_components()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        kwargs = dict(
            anchor_idx=user_idx,
            target_idx=item_idx,
        )
        user_hist_idx = self.viewer["user"](**kwargs)

        kwargs = dict(
            anchor_idx=item_idx,
            target_idx=user_idx,
        )
        item_hist_idx = self.viewer["item"](**kwargs)

        user_emb_slice = dict(
            anchor=self.user(user_idx),
            history=self.item(user_hist_idx),
        )

        item_emb_slice = dict(
            anchor=self.item(item_idx),
            history=self.user(item_hist_idx),
        )

        return user_emb_slice, item_emb_slice

    def _set_up_components(self):
        self._create_components()
        self._create_embeddings()
        self._init_embeddings()

    def _create_components(self):
        kwargs = dict(
            histories=self.histories["user"],
            padding_idx=self.num_items,
        )
        viewer_user = HistoryIDXViewer(**kwargs)
        
        kwargs = dict(
            histories=self.histories["item"],
            padding_idx=self.num_users,
        )
        viewer_item = HistoryIDXViewer(**kwargs)

        components = dict(
            user=viewer_user,
            item=viewer_item,
        )
        self.viewer = nn.ModuleDict(components)

    def _create_embeddings(self):
        kwargs = dict(
            num_embeddings=self.num_users+1, 
            embedding_dim=self.embedding_dim,
            padding_idx=self.num_users,
        )
        self.user = nn.Embedding(**kwargs)

        kwargs = dict(
            num_embeddings=self.num_items+1, 
            embedding_dim=self.embedding_dim,
            padding_idx=self.num_items,
        )
        self.item = nn.Embedding(**kwargs)

    def _init_embeddings(self):
        embeddings = [
            self.user,
            self.item,
        ]

        for emb in embeddings:
            kwargs = dict(
                tensor=emb.weight, 
                mean=0.0, 
                std=0.01,
            )
            nn.init.normal_(**kwargs)