import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# -------------------------------
# Masked Conv2D (standard PixelCNN mask)
# -------------------------------
class MaskedConv2d(nn.Conv2d):
    """
    'A' mask: forbids center pixel.
    'B' mask: allows center pixel.
    Causal over spatial dims only (height, width).
    Channel-AR is handled by reshaping W->W*C outside this layer.
    """
    def __init__(self, mask_type: str, in_channels: int, out_channels: int, kernel_size: int, **kwargs):
        assert mask_type in {"A", "B"}
        padding = kwargs.pop("padding", kernel_size // 2)
        super().__init__(in_channels, out_channels, kernel_size, padding=padding, bias=True, **kwargs)
        self.register_buffer("mask", torch.ones_like(self.weight))
        k = self.kernel_size[0]
        assert self.kernel_size[0] == self.kernel_size[1], "Use odd square kernels (e.g., 3, 5)."

        # Build spatial mask
        self.mask[..., k // 2, k // 2 + (mask_type == "B"):] = 0  # right of center
        self.mask[..., k // 2 + 1:, :] = 0                        # rows below

        if mask_type == "A":
            self.mask[..., k // 2, k // 2] = 0  # forbid center for first layer

    def forward(self, x):
        w = self.weight * self.mask            # stays in the graph
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

# -------------------------------
# Residual Block with MaskedConv
# -------------------------------
class PixelCNNResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReplicationPad2d(0),  # no-op, keeps TorchScript happy if needed
            nn.GroupNorm(num_groups=1, num_channels=channels),
            nn.ReLU(inplace=True),
            MaskedConv2d("B", channels, channels, kernel_size=kernel_size),
            nn.Dropout(dropout),
            nn.GroupNorm(num_groups=1, num_channels=channels),
            nn.ReLU(inplace=True),
            MaskedConv2d("B", channels, channels, kernel_size=1),
        )

    def forward(self, x):
        return x + self.net(x)