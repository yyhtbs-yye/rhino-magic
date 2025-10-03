from typing import Optional

import torch.nn as nn
from rhino.baked_nn.gan.shared.patch_disc import PatchDiscriminator

class BiPatchDiscriminators(nn.Module):
    def __init__(self,
                 x_channels: int,
                 y_channels: int,
                 base_channels: int = 64,
                 num_conv: int = 3,
                 norm_cfg=dict(type='IN'),
                 init_cfg: Optional[dict] = dict(type='normal', gain=0.02)):
        super().__init__()
        self.D_Y = PatchDiscriminator(
            in_channels=y_channels,
            base_channels=base_channels,
            num_conv = num_conv,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
        )
        self.D_X = PatchDiscriminator(
            in_channels=x_channels,
            base_channels=base_channels,
            num_conv = num_conv,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
        )

    def forward(self, tensor, mode: str = 'y'):
        if mode == 'y':
            return self.D_Y(tensor)
        elif mode == 'x':
            return self.D_X(tensor)
        elif mode == 'yx':
            return self.D_Y(tensor[0]), self.D_X(tensor[1])
        else:
            raise ValueError(f"Mode '{mode}' not recognized in BiResnetDiscriminators")