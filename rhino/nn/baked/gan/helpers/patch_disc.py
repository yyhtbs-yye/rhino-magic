# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch.nn as nn
from rhino.nn.baked.bcv.conv_module import ConvModule, build_conv_layer
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
    """

    def __init__(self,
                 in_channels: int,
                 base_channels: int = 64,
                 num_conv: int = 3,
                 norm_cfg: dict = dict(type='BN')):
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

        self.init_weights()

    def forward(self, x: Tensor) -> Tensor:
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        return self.model(x)

    def init_weights(self):
        """Initialize weights following the common pix2pix scheme.

        - Conv / ConvTranspose / Linear: N(0, 0.02), bias = 0
        - BatchNorm / InstanceNorm (if affine): weight ~ N(1, 0.02), bias = 0
        """
        def _init(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.normal_(m.weight, 0.0, 0.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.InstanceNorm2d, nn.GroupNorm)):
                # InstanceNorm may have affine=False; check before init
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.normal_(m.weight, 1.0, 0.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(_init)
