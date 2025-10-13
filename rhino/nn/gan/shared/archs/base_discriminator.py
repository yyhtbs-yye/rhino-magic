# single_stage_discriminator.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from trainer.utils.build_components import build_module

class BaseDiscriminator(nn.Module):

    def __init__(self, in_channels, stem_config, backbone_configs, final_config):
        super().__init__()
        self.in_channels = in_channels

        # Optional fromRGB / stem (Identity if None)
        self.stem = build_module(stem_config)
        self.backbone = build_module(backbone_configs)
        self.final = build_module(final_config)

    def forward(self, img):
        """
        img: (B, in_channels, H, W)
        """
        x = self.stem(img)
        x = self.backbone(x)
        out = self.final(x)
        return out
