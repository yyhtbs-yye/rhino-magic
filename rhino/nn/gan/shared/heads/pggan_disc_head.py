import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from rhino.nn.gan.shared.blocks.equalized_lr_conv_module import EqualizedLRModule

class PGGANDiscHead(EqualizedLRModule):
    """
    Final discriminator head operating on the base-resolution features.

    Typical PGGAN-style:
      - (optional) Minibatch-StdDev channel
      - 3x3 conv + LeakyReLU
      - Flatten + Linear -> logits

    Args:
        in_channels:     feature channels entering the final head.
        hidden_channels: channels after the 3x3 conv (default: in_channels).
        out_features:    size of the output (default: 1 for real/fake logit).
        use_mbstd:       append a minibatch-stddev channel before the final conv.
        mbstd_group:     group size for minibatch-stddev (clamped to batch size).
        lrelu_slope:     LeakyReLU slope.
    """
    ELR_ENABLED = True
    ELR_INIT = "pggan"

    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_features=1,
        use_mbstd=True,
        mbstd_group=4,
        lrelu_slope=0.2,
        bias=True,
        eps=1e-8,
    ):
        super().__init__()
        self.use_mbstd = use_mbstd
        self.mbstd_group = mbstd_group
        self.eps = eps
        self.act = nn.LeakyReLU(lrelu_slope, inplace=True)

        c_in = in_channels + (1 if use_mbstd else 0)
        c_hidden = hidden_channels if hidden_channels is not None else in_channels

        self.conv = nn.Conv2d(c_in, c_hidden, kernel_size=3, padding=1, bias=bias)
        # Linear is applied on flattened HxW; keep flexible spatial size.
        self.fc = nn.Linear(c_hidden, out_features, bias=bias)

    def _minibatch_stddev(self, x):
        """
        Adds one channel containing the per-sample minibatch stddev (averaged over C,H,W).
        Shape in:  (N, C, H, W)
        Shape out: (N, C+1, H, W)
        """
        N, C, H, W = x.shape
        if N == 1 or not self.use_mbstd:
            return torch.cat([x, x.new_zeros(N, 1, H, W)], dim=1)

        G = min(self.mbstd_group, N)
        if N % G != 0:
            G = 1  # fall back to whole batch if not divisible

        y = x.view(G, N // G, C, H, W)
        y = y - y.mean(dim=0, keepdim=True)
        y = torch.sqrt((y ** 2).mean(dim=0) + self.eps)   # (N//G, C, H, W)
        y = y.mean(dim=(1, 2, 3), keepdim=True)           # (N//G, 1, 1, 1)
        y = y.repeat(G, 1, H, W)                          # (N, 1, H, W)
        return torch.cat([x, y], dim=1)

    def forward(self, x):
        """
        x: (B, C, H, W) at the base resolution (e.g., 4x4 or 8x8).
        """
        if self.use_mbstd:
            x = self._minibatch_stddev(x)

        x = self.act(self.conv(x))           # (B, hidden, H, W)
        # Global mean pooling over spatial dims before the FC, keeps it res-agnostic
        x = x.mean(dim=(2, 3))               # (B, hidden)
        out = self.fc(x)                     # (B, out_features)
        return out
