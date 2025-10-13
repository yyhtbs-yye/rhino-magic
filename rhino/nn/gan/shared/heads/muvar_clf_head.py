import math

import torch
import torch.nn as nn
from timm.layers import Mlp

class MuvarClfHead(nn.Module):
    """
    Predicts (mu, logvar) for continuous codes.
      returns: {'mu': (B, D), 'logvar': (B, D)}
    """
    def __init__(self, in_channels, param_dim,
                 hidden_scale=1,        # must be a perfect square (1, 4, 9, ...)
                 hidden=None,                 # int or iterable of ints, or None
                 logvar_min=-10.0,
                 logvar_max=10.0,
    ):
        super().__init__()

        # Validate hidden_scale is a perfect square
        sqrt_hidden_scale = math.sqrt(hidden_scale)
        assert abs(sqrt_hidden_scale - round(sqrt_hidden_scale)) < 1e-3, \
            "hidden_scale must be a perfect square, like 1, 4, 9, 16, ..."
        s = int(round(sqrt_hidden_scale))

        # (B, C, H, W) -> (B, C * s * s)
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(s), nn.Flatten(1))

        pooled_dim = in_channels * (s * s)
        out_dim = 2 * param_dim

        # timm.layers.Mlp is 2-layer (in -> hidden -> out) with GELU

        if hidden is None:
            self.mlp = nn.Linear(in_channels, out_dim)
        else:
            self.mlp = Mlp(
                in_features=pooled_dim,
                hidden_features=hidden,
                out_features=out_dim,
                drop=0.0,
                bias=True,
            )

        self.param_dim = param_dim
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

    def forward(self, h: torch.Tensor):
        # h: (B, C, H, W)
        v = self.pool(h)                 # (B, C * s * s)
        params = self.mlp(v)             # (B, 2 * D)
        mu, logvar = torch.chunk(params, 2, dim=1)  # each (B, D)
        if self.logvar_min is not None or self.logvar_max is not None:
            logvar = logvar.clamp(min=self.logvar_min, max=self.logvar_max)
        return {'mu': mu, 'logvar': logvar}
