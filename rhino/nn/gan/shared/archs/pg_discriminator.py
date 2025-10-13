import torch
import torch.nn as nn
import torch.nn.functional as F

from trainer.utils.build_components import build_module

class PGDiscriminator(nn.Module):

    def __init__(
        self, in_channels, h_channels, num_layers,
        stem_configs, downsample_config, backbone_configs, final_config):
        super().__init__()
        self.in_channels = in_channels
        self.h_channels = h_channels

        # Progressive status
        self.curr_stage = 0
        self.alpha = 1.0

        # Build components
        self.stems = nn.ModuleList([build_module(stem_configs[i]) for i in range(num_layers)])
        self.backbone = nn.ModuleList([build_module(backbone_configs[i]) for i in range(num_layers)])
        self.downsample = build_module(downsample_config)
        # final head usually depends on h_channels; ensure its config encodes that.
        self.final = build_module(final_config)

    @property
    def num_stages(self):
        return len(self.backbone)

    def set_stage(self, stage_index, alpha=1.0):
        self.curr_stage = stage_index
        self.alpha = alpha

    def forward(self, img, stage=None, alpha=None):
        """
        img: (B, in_channels, H, W) where H,W correspond to the resolution of `stage`.
        """
        if stage is None:
            stage = self.curr_stage
        if alpha is None:
            alpha = self.alpha

        # Transition phase: blend high-res path with downsampled previous path
        if stage > 0 and alpha < 1.0:
            # High-res path: fromRGB at current stage, then down through current block
            x_high = self.stems[stage](img)
            x_high = self.backbone[stage](x_high)  # now at (stage-1) resolution/features

            # Low-res skip path: downsample image once, then previous stage fromRGB
            img_down = self.downsample(img)
            x_low = self.stems[stage - 1](img_down)

            # Blend at (stage-1) feature level
            x = alpha * x_high + (1.0 - alpha) * x_low

            # Continue through remaining lower-stage blocks
            for s in range(stage - 1, -1, -1):
                x = self.backbone[s](x)

        else:
            # Stable phase: go straight down from current stage to base
            x = self.stems[stage](img)
            for s in range(stage, -1, -1):
                x = self.backbone[s](x)

        # Final prediction head (e.g., to scalar logits)
        out = self.final(x)
        return out
