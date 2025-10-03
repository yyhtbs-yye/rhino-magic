import numpy as np
import torch.nn as nn
from rhino.baked_nn.bcv.conv_module import ConvModule

class DCGANDiscriminator(nn.Module):

    def __init__(self, input_scale, output_scale,
                 out_channels, in_channels=3, base_channels=128,
                 default_norm_cfg=dict(type='GN'),
                 default_act_cfg=dict(type='LeakyReLU')):
        
        super().__init__()
        self.input_scale = input_scale
        self.output_scale = output_scale
        self.out_channels = out_channels
        self.base_channels = base_channels

        # the number of times for downsampling
        self.num_downsamples = int(np.log2(input_scale // output_scale))

        # build up downsampling backbone (excluding the output layer)
        downsamples = []
        curr_channels = in_channels
        for i in range(self.num_downsamples):
            # remove norm for the first conv
            norm_cfg_ = None if i == 0 else default_norm_cfg
            in_ch = in_channels if i == 0 else base_channels * 2**(i - 1)

            downsamples.append(
                ConvModule(in_ch, base_channels * 2**i,
                    kernel_size=4, stride=2, padding=1,
                    conv_cfg=dict(type='Conv2d'), norm_cfg=norm_cfg_, act_cfg=default_act_cfg))
            
            curr_channels = base_channels * (2**i)

        self.downsamples = nn.Sequential(*downsamples)

        # define output layer
        self.output_layer = ConvModule(curr_channels, out_channels,
                                       kernel_size=3, stride=1, padding=1, 
                                       conv_cfg=dict(type='Conv2d'), norm_cfg=None, act_cfg=None)

        # ---- Add this line to initialize weights on creation ----
        self.init_weights()

    # ----------------- NEW: weight initialization -----------------
    def init_weights(self):
        """
        DCGAN-style initialization:
        - Conv/Linear weights ~ N(0, 0.02), bias = 0
        """
        def _weights_init_dcgan(m):
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

        self.apply(_weights_init_dcgan)

    def forward(self, x):

        n = x.shape[0]
        x = self.downsamples(x)
        x = self.output_layer(x)

        return x.view(n, -1)
