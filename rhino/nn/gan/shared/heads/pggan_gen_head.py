import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from rhino.nn.gan.shared.blocks.equalized_lr_conv_module import EqualizedLRModule

class PGGANGenHead(EqualizedLRModule):
    """
    Per-stage head that maps features -> image (ToRGB).
    Final tanh is applied by the top-level generator after fade-in mixing.
    """
    ELR_ENABLED = True
    ELR_INIT = "pggan"

    def __init__(self, in_channels, out_channels=3, bias=True):
        super().__init__()
        self.net = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.net(x)






