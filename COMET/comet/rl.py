import numpy as np
import torch
import torch.nn as nn


class Module(nn.Module):
    def __init__(
        self,
        n_factors: int,
        hidden: list,
        dropout: float,
        channels: int,
        kernel_length: int,
        kernel_width: list,
        stride: int,
    ):
        super().__init__()
        # attr dictionary for load
        self.init_args = locals().copy()
        del self.init_args["self"]
        del self.init_args["__class__"]

        # global attr
        self.n_factors = n_factors
        self.hidden = hidden
        self.dropout = dropout
        self.channels = channels
        self.kernel_length = kernel_length
        self.kernel_width = kernel_width
        self.stride = stride

        # debugging args error
        self._assert_arg_error()

        # generate layers
        self._init_layers()

    def forward(
        self, 
        hist_map: torch.Tensor, 
    ):
        """
        hist_matrix: (B,H,D)
        """
        agg_vector = self.agg(hist_map)
        rep_vector = self.interaction(agg_vector)
        return rep_vector

    def interaction(self, agg_vector):
        rep_vector = self.mlp_layers(agg_vector)
        return rep_vector

    def agg(self, hist_map):
        hist_map_exp = hist_map.unsqueeze(1)
        features = [conv(hist_map_exp).flatten(1) for conv in self.conv_layers]
        agg_vector = torch.cat(tensors=features, dim=1)
        return agg_vector

    def _init_layers(self):
        kwargs = dict(
            channels=self.channels,
            kernel_length=self.kernel_length,
            kernel_width=self.kernel_width,
            stride=self.stride,
        )
        self.conv_layers = nn.ModuleList(
            list(self._generate_conv_layers(**kwargs)),
        )

        self.mlp_layers = nn.Sequential(
            *list(self._generate_mlp_layers(self.hidden))
        )

    def _generate_conv_layers(self, channels, kernel_length, kernel_width, stride):
        for window in kernel_width:
            yield nn.Conv2d(
                in_channels=1, 
                out_channels=channels, 
                kernel_size=(kernel_length, window), 
                stride=stride,
            )

    def _generate_mlp_layers(self, hidden):
        idx = 1
        while idx < len(hidden):
            yield nn.Linear(hidden[idx-1], hidden[idx])
            yield nn.LayerNorm(hidden[idx])
            yield nn.ReLU()
            yield nn.Dropout(self.dropout)
            idx += 1

    def _assert_arg_error(self):
        AGG_DIM = self.channels * sum([(self.n_factors - window) // self.stride + 1 for window in self.kernel_width])
        CONDITION = (self.hidden[0] == AGG_DIM)
        ERROR_MESSAGE = f"First MLP layer must match input size: {AGG_DIM}"
        assert CONDITION, ERROR_MESSAGE

        CONDITION = (self.hidden[-1] == self.n_factors)
        ERROR_MESSAGE = f"Last MLP layer must match input size: {self.n_factors}"
        assert CONDITION, ERROR_MESSAGE