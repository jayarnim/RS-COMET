import torch
import torch.nn as nn


class PoolingLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        channels: int,
        kernel_length: int,
        kernel_width: list,
        hidden_dim: list,
        dropout: float,
    ):
        super().__init__()

        # global attr
        self.input_dim = input_dim
        self.channels = channels
        self.kernel_length = kernel_length
        self.kernel_width = kernel_width
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # debugging args error
        self._assert_arg_error()

        # generate layers
        self._set_up_components()

    def forward(
        self, 
        hist_emb: torch.Tensor, 
    ):
        hist_feat = self.extraction(hist_emb)
        hist_rep = self.representation(hist_feat)
        return hist_rep

    def representation(self, hist_feat):
        return self.mlp(hist_feat)

    def extraction(self, hist_emb):
        hist_emb_exp = hist_emb.unsqueeze(1)
        hist_feat = [conv(hist_emb_exp).flatten(1) for conv in self.cnn]
        return torch.cat(tensors=hist_feat, dim=1)

    def _set_up_components(self):
        self._create_layers()

    def _create_layers(self):
        components = list(self._yield_conv_block(self.kernel_width))
        self.cnn = nn.ModuleList(components)

        components = list(self._yield_linear_block(self.hidden_dim))
        self.mlp = nn.Sequential(*components)

    def _yield_conv_block(self, kernel_width):
        IN_CHANNELS = 1
        OUT_CHANNELS = self.channels
        KERNEL_LENGTH = self.kernel_length
        STRIDE = 1
        PADDING = 0

        for KERNEL_WIDTH in kernel_width:
            kwargs = dict(
                in_channels=IN_CHANNELS, 
                out_channels=OUT_CHANNELS, 
                kernel_size=(KERNEL_LENGTH, KERNEL_WIDTH), 
                stride=STRIDE,
                padding=PADDING,
            )
            yield nn.Sequential(
                nn.Conv2d(**kwargs), 
                nn.BatchNorm2d(OUT_CHANNELS),
                nn.ReLU(), 
                nn.Dropout(self.dropout),
            )

    def _yield_linear_block(self, hidden_dim):
        SATIAL_SIZE = self.input_dim
        PADDING = 0
        STRIDE = 1
        IN_FEATRUES = self.channels * sum([
            (SATIAL_SIZE + 2*PADDING - KERNEL_WIDTH)//STRIDE + 1
            for KERNEL_WIDTH in self.kernel_width
        ])
        
        for OUT_FEATURES in hidden_dim:
            yield nn.Sequential(
                nn.Linear(IN_FEATRUES, OUT_FEATURES),
                nn.LayerNorm(OUT_FEATURES),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            )
            IN_FEATRUES = OUT_FEATURES

    def _assert_arg_error(self):
        CONDITION = self.hidden_dim[-1]==self.input_dim
        ERROR_MESSAGE = f"the last unit of mlp must match the embedding dimension: {self.input_dim}"
        assert CONDITION, ERROR_MESSAGE