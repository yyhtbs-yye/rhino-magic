import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelNorm(nn.Module):
    """Pixelwise feature vector normalization used in the original PGGAN paper."""
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(torch.mean(x * x, dim=1, keepdim=True) + self.eps)