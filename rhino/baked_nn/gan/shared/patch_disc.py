# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch.nn as nn
from rhino.baked_nn.bcv.helpers import build_conv_layer
from rhino.baked_nn.bcv.conv_module import ConvModule
from torch import Tensor

class PatchDiscriminator(nn.Module):
    """A PatchGAN discriminator.

    Args:
        in_channels (int): Number of channels in input images.
        base_channels (int): Number of channels at the first conv layer.
            Default: 64.
        num_conv (int): Number of stacked intermediate convs (excluding input
            and output conv). Default: 3.
        norm_cfg (dict): Config dict to build norm layer. Default:
            `dict(type='BN')`.
        init_cfg (dict): Config dict for initialization.
            `type`: The name of our initialization method. Default: 'normal'.
            `gain`: Scaling factor for normal, xavier and orthogonal.
            Default: 0.02.
    """

    def __init__(self,
                 in_channels: int,
                 base_channels: int = 64,
                 num_conv: int = 3,
                 norm_cfg: dict = dict(type='GN'),
                 init_cfg: Optional[dict] = dict(type='normal', gain=0.02)):

        super().__init__()
        
        assert isinstance(norm_cfg, dict), ("'norm_cfg' should be dict, but"
                                            f'got {type(norm_cfg)}')
        assert 'type' in norm_cfg, "'norm_cfg' must have key 'type'"
        # We use norm layers in the patch discriminator.
        # Only for IN, use bias since it does not have affine parameters.
        use_bias = norm_cfg['type'] == 'IN'

        kernel_size = 4
        padding = 1

        # input layer
        sequence = [
            ConvModule(
                in_channels=in_channels,
                out_channels=base_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                bias=True,
                norm_cfg=None,
                act_cfg=dict(type='LeakyReLU', negative_slope=0.2))
        ]

        # stacked intermediate layers,
        # gradually increasing the number of filters
        multiple_now = 1
        multiple_prev = 1
        for n in range(1, num_conv):
            multiple_prev = multiple_now
            multiple_now = min(2**n, 8)
            sequence += [
                ConvModule(
                    in_channels=base_channels * multiple_prev,
                    out_channels=base_channels * multiple_now,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=padding,
                    bias=use_bias,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='LeakyReLU', negative_slope=0.2))
            ]
        multiple_prev = multiple_now
        multiple_now = min(2**num_conv, 8)
        sequence += [
            ConvModule(
                in_channels=base_channels * multiple_prev,
                out_channels=base_channels * multiple_now,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=use_bias,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='LeakyReLU', negative_slope=0.2))
        ]

        # output one-channel prediction map
        sequence += [
            build_conv_layer(
                dict(type='Conv2d'),
                base_channels * multiple_now,
                1,
                kernel_size=kernel_size,
                stride=1,
                padding=padding)
        ]

        self.model = nn.Sequential(*sequence)

        # store init config
        self.init_type = 'normal' if init_cfg is None else init_cfg.get('type', 'normal')
        self.init_gain = 0.02 if init_cfg is None else float(init_cfg.get('gain', 0.02))

        self.init_weights()

    def init_weights(self):
        """Initialize module weights according to self.init_type / self.init_gain."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.init_type == 'normal':
                    nn.init.normal_(m.weight, 0.0, self.init_gain)
                elif self.init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight, gain=self.init_gain)
                elif self.init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                elif self.init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight, gain=self.init_gain)
                else:
                    # fall back to normal
                    nn.init.normal_(m.weight, 0.0, self.init_gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if getattr(m, 'weight', None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        return self.model(x)
