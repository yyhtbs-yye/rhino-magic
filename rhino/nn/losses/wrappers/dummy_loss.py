import torch
import torch.nn as nn

class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, **args):
        return torch.rand(1)
