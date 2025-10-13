# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rhino.baked_nn.bcv.helpers import build_norm_layer, build_activation_layer, build_upsample_layer

from rhino.baked_nn.bcv.helpers import pixel_norm

class PGGANNoiseTo2DFeat(nn.Module):

    def __init__(self,
                 noise_size,
                 out_channels,
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
                 norm_cfg=dict(type='PN'),
                 normalize_latent=True,
                 order=('linear', 'act', 'norm')):
        super().__init__()
        self.noise_size = noise_size
        self.out_channels = out_channels
        self.normalize_latent = normalize_latent
        self.with_activation = act_cfg is not None
        self.with_norm = norm_cfg is not None
        self.order = order
        assert len(order) == 3 and set(order) == set(['linear', 'act', 'norm'])

        # w/o bias, because the bias is added after reshaping the tensor to
        # 2D feature
        self.linear = EqualizedLRLinearModule(
            noise_size,
            out_channels * 16,
            equalized_lr_cfg=dict(gain=np.sqrt(2) / 4),
            bias=False)

        if self.with_activation:
            self.activation = build_activation_layer(act_cfg)

        # add bias for reshaped 2D feature.
        self.register_parameter(
            'bias', nn.Parameter(torch.zeros(1, out_channels, 1, 1)))

        if self.with_norm:
            _, self.norm = build_norm_layer(norm_cfg, out_channels)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input noise tensor with shape (n, c).

        Returns:
            Tensor: Forward results with shape (n, c, 4, 4).
        """
        assert x.ndim == 2
        if self.normalize_latent:
            x = pixel_norm(x)
        for order in self.order:
            if order == 'linear':
                x = self.linear(x)
                # [n, c, 4, 4]
                x = torch.reshape(x, (-1, self.out_channels, 4, 4))
                x = x + self.bias
            elif order == 'act' and self.with_activation:
                x = self.activation(x)
            elif order == 'norm' and self.with_norm:
                x = self.norm(x)

        return x

class PGGANDecisionHead(nn.Module):

    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 bias=True,
                 equalized_lr_cfg=dict(gain=1),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
                 out_act=None):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.with_activation = act_cfg is not None
        self.with_out_activation = out_act is not None

        # setup linear layers
        # dirty code for supporting default mode in PGGAN
        if equalized_lr_cfg:
            equalized_lr_cfg_ = dict(gain=2**0.5)
        else:
            equalized_lr_cfg_ = None
        self.linear0 = EqualizedLRLinearModule(
            self.in_channels,
            self.mid_channels,
            bias=bias,
            equalized_lr_cfg=equalized_lr_cfg_)
        self.linear1 = EqualizedLRLinearModule(
            self.mid_channels,
            self.out_channels,
            bias=bias,
            equalized_lr_cfg=equalized_lr_cfg)

        # setup activation layers
        if self.with_activation:
            self.activation = build_activation_layer(act_cfg)

        if self.with_out_activation:
            self.out_activation = build_activation_layer(out_act)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        if x.ndim > 2:
            x = torch.reshape(x, (x.shape[0], -1))

        x = self.linear0(x)
        if self.with_activation:
            x = self.activation(x)

        x = self.linear1(x)
        if self.with_out_activation:
            x = self.out_activation(x)

        return x

class MiniBatchStddevLayer(nn.Module):
    """Minibatch standard deviation.

    Args:
        group_size (int, optional): The size of groups in batch dimension.
            Defaults to 4.
        eps (float, optional):  Epsilon value to avoid computation error.
            Defaults to 1e-8.
    """

    def __init__(self, group_size=4, eps=1e-8):
        super().__init__()
        self.group_size = group_size
        self.eps = eps

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        # batch size should be smaller than or equal to group size. Otherwise,
        # batch size should be divisible by the group size.
        assert x.shape[
            0] <= self.group_size or x.shape[0] % self.group_size == 0, (
                'Batch size be smaller than or equal '
                'to group size. Otherwise,'
                ' batch size should be divisible by the group size.'
                f'But got batch size {x.shape[0]},'
                f' group size {self.group_size}')
        n, c, h, w = x.shape
        group_size = min(n, self.group_size)
        # [G, M, C, H, W]
        y = torch.reshape(x, (group_size, -1, c, h, w))
        # [G, M, C, H, W]
        y = y - y.mean(dim=0, keepdim=True)
        # In pt>=1.7, you can just use `.square()` function.
        # [M, C, H, W]
        y = y.pow(2).mean(dim=0, keepdim=False)
        y = torch.sqrt(y + self.eps)
        # [M, 1, 1, 1]
        y = y.mean(dim=(1, 2, 3), keepdim=True)
        y = y.repeat(group_size, 1, h, w)
        return torch.cat([x, y], dim=1)