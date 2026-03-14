import torch
import torch.nn as nn


class ProjectionLayer(nn.Module):
    def __init__(
        self,
        dim: int,
    ):
        super().__init__()

        # global attr
        self.dim = dim

        # generate layers
        self._set_up_components()

    def forward(
        self, 
        X: torch.Tensor, 
    ):
        return self.linear(X).squeeze(-1)

    def _set_up_components(self):
        self._create_layers()

    def _create_layers(self):
        kwargs = dict(
            in_features=self.dim,
            out_features=1,
        )
        self.linear = nn.Linear(**kwargs)