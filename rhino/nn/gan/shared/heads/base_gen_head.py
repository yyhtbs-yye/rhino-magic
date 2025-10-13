import math
import torch
import torch.nn as nn

from rhino.baked_nn.bcv.conv_module import ConvModule

class BaseGenHead(nn.Module):
    """
    Input:  (B, IN_CH, H, W)
    Output: (B, 3, H, W)
    """
    def __init__(self, in_channels, 
                 out_channels=3):
        super().__init__()
        self.final = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.final(x)
