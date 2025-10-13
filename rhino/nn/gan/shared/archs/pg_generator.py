import torch
import torch.nn as nn
import torch.nn.functional as F

from trainer.utils.build_components import build_module

class PGGenerator(nn.Module):

    def __init__(self, z_dim, h_channels, num_layers,
                 project_config, upsample_config, backbone_configs, head_configs):
        super().__init__()
        self.z_dim = z_dim
        self.h_channels = h_channels

        # Progressive status
        self.curr_stage = 0
        self.alpha = 1.0

        # Build components
        self.project = build_module(project_config)
        self.upsample = build_module(upsample_config)
        self.backbone = nn.ModuleList([build_module(backbone_configs[i]) for i in range(num_layers)])
        self.heads = nn.ModuleList([build_module(head_configs[i]) for i in range(num_layers)])

    @property
    def num_stages(self):
        return len(self.backbone)

    def set_stage(self, stage_index, alpha=1.0):
        self.curr_stage = stage_index
        self.alpha = alpha

    def forward(self, z, stage=None, alpha=None):
        if stage is None:
            stage = self.curr_stage
        if alpha is None:
            alpha = self.alpha

        # z -> base feature map (expects project to output spatial map)
        x = self.project(z)

        feats = []
        for s in range(stage+1):
            x = self.backbone[s](x)
            feats.append(x)

        curr_out = self.heads[stage](feats[stage])

        if stage > 0 and alpha < 1.0:
            prev_out = self.heads[stage - 1](feats[stage - 1])
            prev_out = self.upsample(prev_out)
            img = alpha * curr_out + (1.0 - alpha) * prev_out
        else:
            img = curr_out

        return torch.tanh(img)
