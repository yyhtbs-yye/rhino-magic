import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rhino.baked_nn.bcv.conv_module import ConvModule

class InfoGANGenerator(nn.Module):
    """
    InfoGAN-style Generator (DCGAN backbone):
      - Input latent = [z, c_cat(one-hot), c_cont]
      - Same upsampling recipe as DCGANGenerator
    """
    def __init__(self, output_scale, out_channels=3, base_channels=1024,
                 input_scale=4, noise_size=100, cat_dim=0, cont_dim=0,
                 default_norm_cfg=dict(type='GN', num_groups=32),
                 default_act_cfg=dict(type='ReLU')):
        super().__init__()
        self.output_scale = output_scale
        self.base_channels = base_channels
        self.input_scale = input_scale
        self.noise_size = noise_size
        self.cat_dim = cat_dim
        self.cont_dim = cont_dim
        self.latent_size = noise_size + cat_dim + cont_dim
        self.init_cfg = None

        # number of upsample stages (excluding final output layer)
        self.num_upsamples = int(math.log2(output_scale // input_scale))

        # (N, latent, 1, 1) -> 4x4 features
        self.noise2feat = ConvModule(
            self.latent_size,
            base_channels,
            kernel_size=4,
            stride=1,
            padding=0,
            conv_cfg=dict(type='ConvTranspose2d'),
            norm_cfg=default_norm_cfg,
            act_cfg=default_act_cfg
        )

        # backbone upsampling (no output layer)
        upsampling = []
        curr_ch = base_channels
        for _ in range(self.num_upsamples - 1):
            upsampling.append(
                ConvModule(
                    curr_ch, curr_ch // 2,
                    kernel_size=4, stride=2, padding=1,
                    conv_cfg=dict(type='ConvTranspose2d'),
                    norm_cfg=default_norm_cfg,
                    act_cfg=default_act_cfg
                )
            )
            curr_ch //= 2
        self.upsampling = nn.Sequential(*upsampling)

        # output layer (no activation here; match DCGAN style)
        self.output_layer = ConvModule(
            curr_ch, out_channels,
            kernel_size=4, stride=2, padding=1,
            conv_cfg=dict(type='ConvTranspose2d'),
            norm_cfg=None, act_cfg=None
        )

        self.init_weights()

    # ---- DCGAN-style initialization (match your DCGAN files) ----
    def init_weights(self):
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

    def _concat_latent(self, z, c_cat=None, c_cont=None, one_hot_cat=False):
        """
        z: (N, noise_size) or (N, noise_size, 1, 1)
        c_cat: (N,) int labels OR (N, cat_dim)[one-hot]; if one_hot_cat=True and c_cat.ndim==1 -> one-hot encode
        c_cont: (N, cont_dim)
        Returns (N, latent_size, 1, 1)
        """
        if z.ndim == 4:
            z = z.squeeze(-1).squeeze(-1)
        if z.ndim != 2:
            raise ValueError(f"z must be (N, C) or (N, C, 1, 1); got {tuple(z.shape)}")

        chunks = [z]
        if self.cat_dim > 0:
            if c_cat is None:
                raise ValueError("cat_dim > 0 but c_cat is None")
            if c_cat.ndim == 1 or (one_hot_cat and c_cat.ndim == 1):
                c_cat = F.one_hot(c_cat.to(torch.int64), num_classes=self.cat_dim).float()
            if c_cat.ndim != 2 or c_cat.size(1) != self.cat_dim:
                raise ValueError(f"c_cat must be (N, {self.cat_dim}) one-hot or (N,) indices")
            chunks.append(c_cat)
        if self.cont_dim > 0:
            if c_cont is None:
                raise ValueError("cont_dim > 0 but c_cont is None")
            if c_cont.ndim != 2 or c_cont.size(1) != self.cont_dim:
                raise ValueError(f"c_cont must be (N, {self.cont_dim})")
            chunks.append(c_cont)

        latent = torch.cat(chunks, dim=1)
        return latent.unsqueeze(-1).unsqueeze(-1)  # (N, C, 1, 1)

    def forward(self, z, c_cat=None, c_cont=None, one_hot_cat=False):
        latent = self._concat_latent(z, c_cat, c_cont, one_hot_cat)
        x = self.noise2feat(latent)
        x = self.upsampling(x)
        x = self.output_layer(x)
        return x
