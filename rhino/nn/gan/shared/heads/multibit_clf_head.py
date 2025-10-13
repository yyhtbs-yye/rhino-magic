import math

import torch
import torch.nn as nn
from timm.layers import Mlp

class MultiBitClfHead(nn.Module):
    """
    Predicts per-bit logits for independent binary attributes.
      returns: {'logits': (B, num_bits)}
    """
    def __init__(self, in_channels, num_bits, 
                hidden_scale=1,        # must be a perfect square (1, 4, 9, ...)
                hidden=None):
        super().__init__()
        # Validate hidden_scale is a perfect square
        sqrt_hidden_scale = math.sqrt(hidden_scale)
        assert abs(sqrt_hidden_scale - round(sqrt_hidden_scale)) < 1e-3, \
            "hidden_scale must be a perfect square, like 1, 4, 9, 16, ..."
        s = int(round(sqrt_hidden_scale))

        # (B, C, H, W) -> (B, C * s * s)
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(s), nn.Flatten(1))

        pooled_dim = in_channels * (s * s)
        out_dim = num_bits

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

    def forward(self, h):
        v = self.pool(h)              # (B, C)
        logits = self.mlp(v)          # (B, Bbits)
        return logits

