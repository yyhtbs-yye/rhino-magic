import torch
import torch.nn as nn
from rhino.baked_nn.bcv.conv_module import ConvModule

class TowerDownBackbone(nn.Module):
    """Upsampling stack (excluding final output layer)."""
    def __init__(self, base_channels, num_upsamples, norm_cfg, act_cfg):
        super().__init__()
        layers = []
        curr_channels = base_channels
        for _ in range(num_upsamples - 1):
            layers.append(
                ConvModule(
                    curr_channels, curr_channels // 2,
                    kernel_size=4, stride=2, padding=1,
                    conv_cfg=dict(type='ConvTranspose2d'),
                    norm_cfg=norm_cfg, act_cfg=act_cfg
                )
            )
            curr_channels //= 2
        self.out_channels = curr_channels
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
