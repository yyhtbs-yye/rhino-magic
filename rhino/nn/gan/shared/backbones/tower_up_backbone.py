import torch
import torch.nn as nn
from rhino.baked_nn.bcv.conv_module import ConvModule

class TowerUpBackbone(nn.Module):
    """Downsampling stack (inverse of the upsampling tower)."""
    def __init__(self, base_channels, num_downsamples, norm_cfg, act_cfg):
        super().__init__()
        layers = []
        curr_channels = base_channels
        for _ in range(num_downsamples - 1):
            layers.append(
                ConvModule(
                    curr_channels, curr_channels * 2,
                    kernel_size=4, stride=2, padding=1,
                    conv_cfg=dict(type='Conv2d'),
                    norm_cfg=norm_cfg, act_cfg=act_cfg
                )
            )
            curr_channels *= 2
        self.out_channels = curr_channels
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
