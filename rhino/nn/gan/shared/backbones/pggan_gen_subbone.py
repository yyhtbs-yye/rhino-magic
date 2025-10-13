import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from rhino.nn.gan.shared.blocks.equalized_lr_conv_module import EqualizedLRModule
from rhino.nn.gan.shared.ops.norms import PixelNorm

class PGGANGenSubbone(EqualizedLRModule):
    ELR_ENABLED = True
    ELR_INIT = "pggan"

    def __init__(self, in_channels, out_channels, 
                 up_kernel=4, up_stride=2, up_padding=1, up_output_padding=0,
                 num_post_convs=1, lrelu_slope=0.2,
                 use_pixelnorm=True, use_skip=False):
        
        super().__init__()
        self.use_pixelnorm = use_pixelnorm
        self.pixelnorm = PixelNorm() if use_pixelnorm else nn.Identity()
        self.act = nn.LeakyReLU(lrelu_slope, inplace=True)

        self.in_channels = in_channels
        self.out_channels = out_channels

        # --- First step: ConvTranspose2d (upsample) ---
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=up_kernel, stride=up_stride,
            padding=up_padding, output_padding=up_output_padding, bias=True
        )

        # --- Post 3x3 convs (stay at out_channels) ---
        self.post_convs = nn.ModuleList(
            [nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
             for _ in range(max(0, num_post_convs))]
        )

        # --- Optional skip path (identity if shapes match, else 1x1 or ConvT) ---
        self.use_skip = use_skip
        if use_skip:
            # Skip path mirrors the resize to align shapes/channels
            self.skip = nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=up_kernel, stride=up_stride,
                padding=up_padding, output_padding=up_output_padding, bias=True
            )
            self.skip_scale = 1.0 / math.sqrt(2.0)
        else:
            self.skip = None

    def forward(self, x):
        y = x
        # first op: upsample
        y = self.up(y)
        y = self.act(y)
        y = self.pixelnorm(y)

        # post 3x3 convs
        for conv in self.post_convs:
            y = self.act(conv(y))
            y = self.pixelnorm(y)

        # optional residual
        if self.use_skip:
            s = self.skip(x)
            y = (y + s) * self.skip_scale

        return y
