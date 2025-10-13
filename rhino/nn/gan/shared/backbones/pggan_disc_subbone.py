import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from rhino.nn.gan.shared.blocks.equalized_lr_conv_module import EqualizedLRModule

# ---------------------------
# Discriminator Backbone Block
# ---------------------------
class PGGANDiscSubbone(EqualizedLRModule):
    """
    Progressive discriminator block WITH internal downsampling via Conv2d (stride=2).

    Typical flow:
        x --[Conv (in->out, stride=2)]--> act --> [Conv3x3]*N --> act

    Args:
        in_channels:   input channels.
        out_channels:  output channels after the downsample step.
        down_kernel/stride/padding: Conv2d params for the downsample step.
        num_post_convs: number of 3x3 convs at the current (downsampled) resolution.
        lrelu_slope:   LeakyReLU slope.
        use_skip:      optional residual skip path (Conv2d stride=2) with sqrt(1/2) scaling.
    """
    ELR_ENABLED = True
    ELR_INIT = "pggan"

    def __init__(
        self,
        in_channels,
        out_channels,
        down_kernel=4,
        down_stride=2,
        down_padding=1,
        num_post_convs=1,
        lrelu_slope=0.2,
        use_skip=False,
    ):
        super().__init__()
        self.act = nn.LeakyReLU(lrelu_slope, inplace=True)

        self.in_channels = in_channels
        self.out_channels = out_channels

        # --- First op: strided Conv2d downsample ---
        self.down = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=down_kernel,
            stride=down_stride,
            padding=down_padding,
            bias=True,
        )

        # --- Post 3x3 convs (stay at out_channels) ---
        self.post_convs = nn.ModuleList(
            [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
                for _ in range(max(0, num_post_convs))
            ]
        )

        # --- Optional residual skip path (mirrors downsample) ---
        self.use_skip = use_skip
        if use_skip:
            self.skip = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=down_kernel,
                stride=down_stride,
                padding=down_padding,
                bias=True,
            )
            self.skip_scale = 1.0 / math.sqrt(2.0)
        else:
            self.skip = None

    def forward(self, x):
        y = self.act(self.down(x))
        for conv in self.post_convs:
            y = self.act(conv(y))

        if self.use_skip:
            s = self.skip(x)
            y = (y + s) * self.skip_scale

        return y


# ------------------------
# Per-stage FromRGB "Head"
# ------------------------
class BaseDiscHeadBlock(EqualizedLRModule):
    """
    Per-stage head that maps image (RGB) -> features (FromRGB).

    Use one of these at each resolution stage in the discriminator.
    """
    ELR_ENABLED = True
    ELR_INIT = "pggan"

    def __init__(self, in_channels=3, out_channels=64, bias=True, lrelu_slope=0.2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.act = nn.LeakyReLU(lrelu_slope, inplace=True)

    def forward(self, x):
        return self.act(self.conv(x))

