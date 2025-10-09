
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Channel-wise LayerNorm helper
# -----------------------------
class ChannelLayerNorm(nn.Module):
    """
    LayerNorm over channels for 2D feature maps.
    Expects input of shape (B, C, H, W).
    """
    def __init__(self, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, H, W, C) -> LN -> (B, C, H, W)
        x_perm = x.permute(0, 2, 3, 1)
        x_norm = self.ln(x_perm)
        return x_norm.permute(0, 3, 1, 2)