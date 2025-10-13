import math
import torch
import torch.nn as nn

from rhino.baked_nn.bcv.conv_module import ConvModule

class DCGANGenerator(nn.Module):

    def __init__(self, output_scale, out_channels=3, base_channels=1024,
                 input_scale=4, noise_size=100,
                 default_norm_cfg=dict(type='GN', num_groups=32),
                 default_act_cfg=dict(type='ReLU'),
                 ):
        super().__init__()
        self.output_scale = output_scale
        self.base_channels = base_channels
        self.input_scale = input_scale
        self.noise_size = noise_size
        self.init_cfg = None

        # the number of times for upsampling
        self.num_upsamples = int(math.log2(output_scale // input_scale))

        # output 4x4 feature map
        self.noise2feat = ConvModule(
            noise_size,
            base_channels,
            kernel_size=4,
            stride=1,
            padding=0,
            conv_cfg=dict(type='ConvTranspose2d'),
            norm_cfg=default_norm_cfg,
            act_cfg=default_act_cfg)

        # build up upsampling backbone (excluding the output layer)
        upsampling = []
        curr_channel = base_channels
        for _ in range(self.num_upsamples - 1):
            upsampling.append(
                ConvModule(
                    curr_channel,
                    curr_channel // 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    conv_cfg=dict(type='ConvTranspose2d'),
                    norm_cfg=default_norm_cfg,
                    act_cfg=default_act_cfg))

            curr_channel //= 2

        self.upsampling = nn.Sequential(*upsampling)

        # output layer
        self.output_layer = ConvModule(
            curr_channel,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            conv_cfg=dict(type='ConvTranspose2d'),
            norm_cfg=None,
            act_cfg=None)

        self.init_weights()

    # ----------------- DCGAN weight initialization -----------------
    def init_weights(self):
        """DCGAN init:
        - Conv/ConvTranspose/Linear weights ~ N(0, 0.02), bias = 0
        """
        def _init(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                if m.weight is not None:
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                if m.weight is not None:
                    nn.init.normal_(m.weight, mean=1.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(_init)
        
    def forward(self, noise: torch.Tensor):
        """
        Args:
            noise (torch.Tensor): Noise of shape (N, C) or (N, C, 1, 1).
            return_noise (bool): If True, return {'fake_img', 'noise_batch'}.
        """
        if not isinstance(noise, torch.Tensor):
            raise TypeError(f"`noise` must be a torch.Tensor, got {type(noise)}")

        # Normalize to (N, C, 1, 1)
        if noise.ndim == 2:
            # (N, C) -> (N, C, 1, 1)
            noise_batch = noise.unsqueeze(-1).unsqueeze(-1)
        elif noise.ndim == 4:
            noise_batch = noise
        else:
            raise ValueError(
                f"`noise` must have shape (N, C) or (N, C, 1, 1), got {tuple(noise.shape)}"
            )

        # Channel sanity check
        if noise_batch.size(1) != self.noise_size:
            raise ValueError(
                f"Expected noise channel size {self.noise_size}, got {noise_batch.size(1)}"
            )

        x = self.noise2feat(noise_batch)
        x = self.upsampling(x)
        x = self.output_layer(x)

        return x