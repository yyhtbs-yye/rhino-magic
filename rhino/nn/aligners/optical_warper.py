import torch.nn as nn

from rhino.nn.utils.warping import flow_warp

# Warper = lambda x,y:flow_warp(x, y)

class Warper(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feat_supp, flow):
        return flow_warp(feat_supp, flow)