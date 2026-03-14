import torch
import torch.nn as nn
from .components.embedding.builder import embedding_builder
from .components.pooling import PoolingLayer
from .components.combination import ElementwiseSum
from .components.matching import MatrixFactorizationLayer
from .components.prediction import ProjectionLayer


class Module(nn.Module):
    def __init__(
        self,
        histories: dict[str, torch.Tensor],
        cfg,
    ):
        """
        Comet: Convolutional dimension interaction for collaborative filtering (Lin et al., 2023)
        -----
        Implements the base structure of COnvolutional diMEnsion inTeraction (COMET),
        MF & id embedding based latent factor model,
        applying CNN to aggregate histories.

        Args:
            num_users (int):
                total number of users in the dataset, U.
            num_items (int):
                total number of items in the dataset, I.
            embedding_dim (int):
                dimensionality of user and item latent representation vectors, K.
            channels (int): 
                number of convolutional feature maps (output channels) used in the CNN layers.
            kernel_size (list):
                widths of the convolution kernels applied to the user and item embedding dimensions.
                each element in the list defines the kernel size for one convolutional layer.
            hidden_dim (int):
                layer dimensions for the MLP-based hist. feature aggregator.
                (e.g., [64, 32, 16, 8])
            dropout (float):
                dropout rate applied to MLP layers for regularization.
            histories (dict[str, torch.Tensor]):
                interaction histories.
                    - `user`: item history for each user.
                    (shape: [U, max_history_length])
                    - `item`: user history for each item. 
                    (shape: [I, max_history_length])
        """
        super().__init__()

        # attr dictionary for load
        self.init_args = locals().copy()
        del self.init_args["self"]
        del self.init_args["__class__"]

        # global attr
        self.num_users = cfg.num_users
        self.num_items = cfg.num_items
        self.embedding_dim = cfg.embedding_dim
        self.channels = cfg.channels
        self.kernel_size = cfg.kernel_size
        self.hidden_dim = cfg.hidden_dim
        self.dropout = cfg.dropout
        self.histories = histories

        # generate layers
        self._set_up_components()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        user_emb, item_emb = self.embedding(user_idx, item_idx)

        user_pooled = self.pooling["user"](user_emb["history"])
        item_pooled = self.pooling["item"](item_emb["history"])

        user_combined = self.combination["user"](user_emb["anchor"], user_pooled)
        item_combined = self.combination["item"](item_emb["anchor"], item_pooled)

        X_pred = self.matching(user_combined, item_combined)

        return X_pred

    def estimate(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        Training Method
        -----

        Args:
            user_idx (torch.Tensor): target user idx (shape: [B,])
            item_idx (torch.Tensor): target item idx (shape: [B,])
        
        Returns:
            logit (torch.Tensor): (u,i) pair extracted logit (shape: [B,])
        """
        X_pred = self.forward(user_idx, item_idx)
        logit = self.prediction(X_pred)
        return logit

    @torch.no_grad()
    def predict(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        Evaluation Method
        -----

        Args:
            user_idx (torch.Tensor): target user idx (shape: [B,])
            item_idx (torch.Tensor): target item idx (shape: [B,])

        Returns:
            logit (torch.Tensor): (u,i) pair extracted logit (shape: [B,])
        """
        X_pred = self.forward(user_idx, item_idx)
        logit = self.prediction(X_pred)
        return logit

    def _set_up_components(self):
        self._create_components()

    def _create_components(self):
        NUM_USERS, MAX_USER_HIST = self.histories["user"].shape
        NUM_ITEMS, MAX_ITEM_HIST = self.histories["item"].shape

        kwargs = dict(
            num_users=self.num_users,
            num_items=self.num_items,
            embedding_dim=self.embedding_dim,
            histories=self.histories,
        )
        self.embedding = embedding_builder(**kwargs)

        kwargs = dict(
            input_dim=self.embedding_dim,
            channels=self.channels,
            kernel_width=self.kernel_size,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        )
        pooling_user = PoolingLayer(
            kernel_length=MAX_USER_HIST,
            **kwargs,
        )
        pooling_item = PoolingLayer(
            kernel_length=MAX_ITEM_HIST,
            **kwargs,
        )
        components = dict(
            user=pooling_user,
            item=pooling_item,
        )
        self.pooling = nn.ModuleDict(components)

        components = dict(
            user=ElementwiseSum(),
            item=ElementwiseSum(),
        )
        self.combination = nn.ModuleDict(components)
        
        self.matching = MatrixFactorizationLayer()

        kwargs = dict(
            dim=self.hidden_dim[-1],
        )
        self.prediction = ProjectionLayer(**kwargs)