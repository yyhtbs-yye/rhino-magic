import torch
import torch.nn as nn

from rhino.nn.gan.shared.blocks.equalized_lr_conv_module import EqualizedLRModule

class Noise2ImgProjector(EqualizedLRModule):
    """
    (B, z_dim) or (B, z_dim, 1, 1) -> (B, C, H, W) via MLP + reshape.
    out_shape should be a tuple (C, H, W).
    """

    ELR_ENABLED = True
    ELR_INIT = "pggan"

    def __init__(self, z_dim, out_shape, hidden_dim=None, 
                 activation=nn.ReLU(),
                 normalization=nn.GroupNorm()):
        super().__init__()
        self.C, self.H, self.W = out_shape
        out_flat = self.C * self.H * self.W

        if hidden_dim is None:
            self.mlp = nn.Linear(z_dim, out_flat)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(z_dim, hidden_dim),
                activation,
                normalization,
                nn.Linear(hidden_dim, out_flat),
            )

    def forward(self, z):
        if z.ndim == 4:  # (B, z_dim, 1, 1) -> (B, z_dim)
            z = z.flatten(1)
        x = self.mlp(z)
        return x.view(z.size(0), self.C, self.H, self.W)
