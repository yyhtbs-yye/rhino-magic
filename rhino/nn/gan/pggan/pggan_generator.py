# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import math
import torch
import torch.nn as nn
from .pggan_modules import EqualizedLRConvModule, EqualizedLRConvUpModule, PGGANNoiseTo2DFeat

from rhino.baked_nn.bcv.helpers import build_upsample_layer

class PGGANGenerator(nn.Module):

    _default_fused_upconv_cfg = dict(
        conv_cfg=dict(type='ConvTranspose2d'),
        kernel_size=3,
        stride=2,
        padding=1,
        bias=True,
        act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
        norm_cfg=dict(type='PN'),
        order=('conv', 'act', 'norm'))
    
    _default_conv_module_cfg = dict(
        conv_cfg=None,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
        act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
        norm_cfg=dict(type='PN'),
        order=('conv', 'act', 'norm'))

    _default_upsample_cfg = dict(type='nearest', scale_factor=2)

    def __init__(self,
                 noise_size,
                 out_scale,
                 label_size=0,
                 base_channels=8192,
                 channel_decay=1.,
                 max_channels=512,
                 fused_upconv=True,
                 conv_module_cfg=None,
                 fused_upconv_cfg=None,
                 upsample_cfg=None):
        super().__init__()
        self.noise_size = noise_size if noise_size else min(
            base_channels, max_channels)
        self.out_scale = out_scale
        self.out_log2_scale = int(math.log2(out_scale))
        # sanity check for the output scale
        assert out_scale == 2**self.out_log2_scale and out_scale >= 4
        self.label_size = label_size
        self.base_channels = base_channels
        self.channel_decay = channel_decay
        self.max_channels = max_channels
        self.fused_upconv = fused_upconv

        # set conv cfg
        self.conv_module_cfg = deepcopy(self._default_conv_module_cfg)
        # update with customized config
        if conv_module_cfg:
            self.conv_module_cfg.update(conv_module_cfg)

        if self.fused_upconv:
            self.fused_upconv_cfg = deepcopy(self._default_fused_upconv_cfg)
            # update with customized config
            if fused_upconv_cfg:
                self.fused_upconv_cfg.update(fused_upconv_cfg)

        self.upsample_cfg = deepcopy(self._default_upsample_cfg)
        if upsample_cfg is not None:
            self.upsample_cfg.update(upsample_cfg)

        self.noise2feat = PGGANNoiseTo2DFeat(noise_size + label_size,
                                             self._num_out_channels(1))

        self.torgb_layers = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()
        for s in range(2, self.out_log2_scale + 1):
            in_ch = self._num_out_channels(
                s - 1) if s == 2 else self._num_out_channels(s - 2)
        
            # setup torgb layers
            self.torgb_layers.append(EqualizedLRConvModule(self._num_out_channels(s - 1), 3, kernel_size=1, stride=1, equalized_lr_cfg=dict(gain=1), 
                                     bias=True, norm_cfg=None, act_cfg=None))
            
            # setup upconv or conv blocks
            self.conv_blocks.extend(self._get_upconv_block(in_ch, s))

        # build upsample layer for residual path
        self.upsample_layer = build_upsample_layer(self.upsample_cfg)

    def _num_out_channels(self, log_scale: int):
        """Calculate the number of output channels based on logarithm of
        current scale.

        Args:
            log_scale (int): The logarithm of the current scale.

        Returns:
            int: The current number of output channels.
        """
        return min(int(self.base_channels / (2.0**(log_scale * self.channel_decay))), self.max_channels)

    def _get_upconv_block(self, in_channels, log_scale):
        """Get the conv block for upsampling.

        Args:
            in_channels (int): The number of input channels.
            log_scale (int): The logarithmic of the current scale.

        Returns:
            nn.Module: The conv block for upsampling.
        """
        modules = []
        # start 4x4 scale
        if log_scale == 2:
            modules.append(EqualizedLRConvModule(in_channels, self._num_out_channels(log_scale - 1), **self.conv_module_cfg))
        # 8x8 --> 1024x1024 scales
        else:
            if self.fused_upconv:
                cfg_ = dict(upsample=dict(type='fused_nn'))
                cfg_.update(self.fused_upconv_cfg)
            else:
                cfg_ = dict(upsample=self.upsample_cfg)
                cfg_.update(self.conv_module_cfg)
            # up + conv
            modules.append(EqualizedLRConvUpModule(in_channels, self._num_out_channels(log_scale - 1), **cfg_))
            # refine conv
            modules.append(EqualizedLRConvModule(self._num_out_channels(log_scale - 1), self._num_out_channels(log_scale - 1), **self.conv_module_cfg))

        return modules

    def forward(self, noise, transition_weight=1., curr_scale=-1):
        
        # noise vector to 2D feature
        x = self.noise2feat(noise)

        # build current computational graph
        curr_log2_scale = self.out_log2_scale if curr_scale < 0 else int(math.log2(curr_scale))

        # 4x4 scale
        x = self.conv_blocks[0](x)
        if curr_log2_scale <= 3:
            out_img = last_img = self.torgb_layers[0](x)

        # 8x8 and larger scales
        for s in range(3, curr_log2_scale + 1):
            x = self.conv_blocks[2 * s - 5](x)
            x = self.conv_blocks[2 * s - 4](x)
            if s + 1 == curr_log2_scale:
                last_img = self.torgb_layers[s - 2](x)
            elif s == curr_log2_scale:
                out_img = self.torgb_layers[s - 2](x)
                residual_img = self.upsample_layer(last_img)
                out_img = residual_img + transition_weight * (out_img - residual_img)

        return out_img