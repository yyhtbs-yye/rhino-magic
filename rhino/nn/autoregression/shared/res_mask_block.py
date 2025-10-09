import torch
import torch.nn as nn
import torch.nn.functional as F

from rhino.nn.autoregression.shared.ops.masked_conv2d import MaskedConv2d

class ResMaskedBlock(nn.Module):
    def __init__(self, C, k=7, n_types=3, p=0.0):
        super().__init__()
        self.conv1 = MaskedConv2d('B', C, C, k, n_types=n_types)
        self.act1  = nn.ReLU(inplace=True)
        self.conv2 = MaskedConv2d('B', C, C, k, n_types=n_types)
        self.drop  = nn.Dropout2d(p) if p > 0 else nn.Identity()
    def forward(self, x):
        h = self.act1(self.conv1(x))
        h = self.drop(self.conv2(h))
        return F.relu(x + h)