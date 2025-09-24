import torch
import torch.nn as nn
from . import rl


class Module(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int,
        hidden: list,
        dropout: float,
        channels: int,
        kernel_width: list,
        stride: int,
        user_hist: torch.Tensor,
        item_hist: torch.Tensor,
    ):
        super().__init__()
        # attr dictionary for load
        self.init_args = locals().copy()
        del self.init_args["self"]
        del self.init_args["__class__"]

        # global attr
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.hidden = hidden
        self.dropout = dropout
        self.channels = channels
        self.kernel_width = kernel_width
        self.stride = stride
        self.register_buffer(
            name="user_hist", 
            tensor=user_hist,
        )
        self.register_buffer(
            name="item_hist", 
            tensor=item_hist,
        )

        # generate layers
        self._init_layers()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        user_idx: (B,)
        item_idx: (B,)
        """
        return self.score(user_idx, item_idx)

    def predict(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        user_idx: (B,)
        item_idx: (B,)
        """
        with torch.no_grad():
            logit = self.score(user_idx, item_idx)
            pred = torch.sigmoid(logit)
        return pred

    def score(self, user_idx, item_idx):
        pred_vector = self.gmf(user_idx, item_idx)
        logit = self.logit_layer(pred_vector).squeeze(-1)
        return logit

    def gmf(self, user_idx, item_idx):
        rep_user = self.user(user_idx, item_idx)
        rep_item = self.item(user_idx, item_idx)
        pred_vector = rep_user * rep_item
        return pred_vector

    def user(self, user_idx, item_idx):
        user_id_embed = self.user_embed(user_idx)
        
        kwargs = dict(
            target_idx=user_idx, 
            counterpart_idx=item_idx, 
            target_hist_idx=self.user_hist, 
            counterpart_padding_value=self.n_items,
        )
        hist_idx_slice = self._hist_idx_slicer(**kwargs)
        hist_embed_slice = self.item_embed(hist_idx_slice)
        user_hist_embed = self.user_rl(hist_embed_slice)

        rep_user = user_id_embed + user_hist_embed

        return rep_user

    def item(self, user_idx, item_idx):
        item_id_embed = self.item_embed(item_idx)
        
        kwargs = dict(
            target_idx=item_idx, 
            counterpart_idx=user_idx, 
            target_hist_idx=self.item_hist, 
            counterpart_padding_value=self.n_users,
        )
        hist_idx_slice = self._hist_idx_slicer(**kwargs)
        hist_embed_slice = self.user_embed(hist_idx_slice)
        item_hist_embed = self.item_rl(hist_embed_slice)

        rep_item = item_id_embed + item_hist_embed

        return rep_item

    def _hist_idx_slicer(self, target_idx, counterpart_idx, target_hist_idx, counterpart_padding_value):
        target_hist_idx_slice = target_hist_idx[target_idx]
        mask = target_hist_idx_slice == counterpart_idx.unsqueeze(1)
        target_hist_idx_slice_padded = target_hist_idx_slice.masked_fill(mask, counterpart_padding_value)
        return target_hist_idx_slice_padded

    def _init_layers(self):
        kwargs = dict(
            num_embeddings=self.n_users+1, 
            embedding_dim=self.n_factors,
            padding_idx=self.n_users,
        )
        self.user_embed = nn.Embedding(**kwargs)

        kwargs = dict(
            num_embeddings=self.n_items+1, 
            embedding_dim=self.n_factors,
            padding_idx=self.n_items,
        )
        self.item_embed = nn.Embedding(**kwargs)

        nn.init.normal_(self.user_embed.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embed.weight, mean=0.0, std=0.01)

        kwargs = dict(
            n_factors=self.n_factors,
            hidden=self.hidden,
            dropout=self.dropout,
            channels=self.channels,
            kernel_width=self.kernel_width,
            stride=self.stride,
        )
        self.user_rl = rl.Module(**kwargs, kernel_length=self.user_hist.size(1))
        self.item_rl = rl.Module(**kwargs, kernel_length=self.item_hist.size(1))

        kwargs = dict(
            in_features=self.n_factors,
            out_features=1,
        )
        self.logit_layer = nn.Linear(**kwargs)