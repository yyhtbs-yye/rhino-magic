import math

import torch
import torch.nn as nn
from rhino.baked_nn.bcv.conv_module import ConvModule

class InfoGANDiscriminator(nn.Module):
    """
    InfoGAN-style Discriminator + Q head (DCGAN backbone):
      - Shared conv downsamples (like DCGANDiscriminator)
      - Adversarial head (real/fake map, like DCGAN)
      - Q-head predicts:
          * categorical code logits: (N, cat_dim)
          * continuous code mean/logvar: (N, cont_dim) each
    """
    def __init__(self, input_scale, output_scale,
                 in_channels=3, base_channels=128,
                 adv_out_channels=1, cat_dim=0, cont_dim=0,
                 default_norm_cfg=dict(type='GN'),
                 default_act_cfg=dict(type='LeakyReLU')):
        super().__init__()
        self.input_scale = input_scale
        self.output_scale = output_scale
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.adv_out_channels = adv_out_channels
        self.cat_dim = cat_dim
        self.cont_dim = cont_dim

        # number of downsamples like DCGAN
        self.num_downsamples = int(math.log2(input_scale // output_scale))

        # shared backbone
        downs = []
        for i in range(self.num_downsamples):
            norm_cfg_ = None if i == 0 else default_norm_cfg   # match DCGAN: no norm in first conv
            in_ch = in_channels if i == 0 else base_channels * (2 ** (i - 1))
            downs.append(
                ConvModule(
                    in_ch, base_channels * (2 ** i),
                    kernel_size=4, stride=2, padding=1,
                    conv_cfg=dict(type='Conv2d'),
                    norm_cfg=norm_cfg_,
                    act_cfg=default_act_cfg
                )
            )
        self.downsamples = nn.Sequential(*downs)
        shared_ch = base_channels * (2 ** (self.num_downsamples - 1))

        # --- Adversarial head (keeps spatial map like your DCGANDiscriminator) ---
        self.adv_head = ConvModule(
            shared_ch, adv_out_channels,
            kernel_size=3, stride=1, padding=1,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=None, act_cfg=None
        )

        # --- Q head ---
        # small conv to mix features, then global pool and 1x1 heads
        q_channels = max(128, base_channels)
        self.q_trunk = ConvModule(
            shared_ch, q_channels,
            kernel_size=3, stride=1, padding=1,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=default_norm_cfg, act_cfg=default_act_cfg
        )
        self.q_pool = nn.AdaptiveAvgPool2d(1)
        self.q_cat = nn.Conv2d(q_channels, cat_dim, kernel_size=1) if cat_dim > 0 else None
        self.q_cont_mu = nn.Conv2d(q_channels, cont_dim, kernel_size=1) if cont_dim > 0 else None
        self.q_cont_logvar = nn.Conv2d(q_channels, cont_dim, kernel_size=1) if cont_dim > 0 else None

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

    def forward(self, x):
        """
        Returns:
          {
            'adv': (N, adv_out_channels * output_scale * output_scale),
            'q_cat_logits': (N, cat_dim) [optional],
            'q_cont_mu': (N, cont_dim)   [optional],
            'q_cont_logvar': (N, cont_dim) [optional],
          }
        """
        n = x.size(0)
        feat = self.downsamples(x)

        # adversarial map (like DCGANDiscriminator) -> flatten
        adv_map = self.adv_head(feat)               # (N, adv_out_channels, S, S)
        adv = adv_map.view(n, -1)

        # Q predictions (global, not per-pixel)
        q_feat = self.q_trunk(feat)
        q_feat = self.q_pool(q_feat)                # (N, C, 1, 1)

        out = {'adv': adv}
        if self.q_cat is not None:
            out['q_cat_logits'] = self.q_cat(q_feat).view(n, self.cat_dim)
        if self.q_cont_mu is not None:
            out['q_cont_mu'] = self.q_cont_mu(q_feat).view(n, self.cont_dim)
        if self.q_cont_logvar is not None:
            out['q_cont_logvar'] = self.q_cont_logvar(q_feat).view(n, self.cont_dim)
        return out
