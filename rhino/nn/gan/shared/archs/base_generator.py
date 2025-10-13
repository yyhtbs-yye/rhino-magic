# single_stage_generator.py

import torch
import torch.nn as nn

from trainer.utils.build_components import build_module

class BaseGenerator(nn.Module):

    def __init__(self, z_dim, project_config, backbone_configs, head_config):
        super().__init__()
        self.z_dim = z_dim

        # z -> spatial feature map
        self.project = build_module(project_config)
        self.backbone = build_module(backbone_configs)
        self.head = build_module(head_config)

    def forward(self, z):
        """
        z: (B, z_dim) or (B, z_dim, 1, 1), depending on project_config.
        """
        x = self.project(z)
        x = self.backbone(x)
        x = self.head(x)

        return torch.tanh(x)
