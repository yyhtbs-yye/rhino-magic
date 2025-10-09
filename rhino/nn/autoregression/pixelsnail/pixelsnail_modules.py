
import torch
import torch.nn as nn
import torch.nn.functional as F

from rhino.nn.autoregression.shared.masked_conv2d import MaskedConv2d
from rhino.nn.autoregression.shared.causal_self_attention2d import CausalSelfAttention2D

from rhino.nn.shared.channel_layer_norm import ChannelLayerNorm

class PixelSNAILBlock(nn.Module):
    """
    A simplified PixelSNAIL block that combines:
      - Pre-norm
      - Causal self-attention over spatial positions
      - Gated masked convolutional pathway (gated 3x3)
    Both sub-paths are added residually to the input.

    Input/Output: (B, C, H, W)
    """
    def __init__(
        self,
        channels: int,
        *,
        n_types: int,
        kernel_size: int = 3,
        heads: int = 8,
        attn_dim: int | None = None,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = ChannelLayerNorm(channels)
        self.attn = CausalSelfAttention2D(
            channels=channels, heads=heads, attn_dim=attn_dim,
            attn_dropout=attn_dropout, resid_dropout=dropout
        )

        # Gated masked conv branch (Mask B so the center can use <= channel order)
        self.norm2 = ChannelLayerNorm(channels)
        self.conv_in = MaskedConv2d('B', channels, 2 * channels, kernel_size, n_types=n_types)
        self.conv_out = nn.Conv2d(channels, channels, kernel_size=1)

        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention residual
        a = self.attn(self.norm1(x))
        x = x + a

        # Gated convolutional residual (GLU)
        h = self.conv_in(self.norm2(x))
        h1, h2 = torch.chunk(h, 2, dim=1) # Split channels C -> C/2, C/2
        h = torch.tanh(h1) * torch.sigmoid(h2)
        h = self.conv_out(h)
        h = self.drop(h)
        x = x + h
        return x
