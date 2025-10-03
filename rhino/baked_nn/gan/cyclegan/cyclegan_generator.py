# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from rhino.baked_nn.bcv.conv_module import ConvModule

from .cyclegan_modules import ResidualBlockWithDropout

class ResnetGenerator(nn.Module):
    """Construct a Resnet-based generator that consists of residual blocks
    between a few downsampling/upsampling operations.

    Args:
        in_channels (int): Number of channels in input images.
        out_channels (int): Number of channels in output images.
        base_channels (int): Number of filters at the last conv layer.
            Default: 64.
        norm_cfg (dict): Config dict to build norm layer. Default:
            `dict(type='IN')`.
        use_dropout (bool): Whether to use dropout layers. Default: False.
        num_blocks (int): Number of residual blocks. Default: 9.
        padding_mode (str): The name of padding layer in conv layers:
            'reflect' | 'replicate' | 'zeros'. Default: 'reflect'.
        init_cfg (dict): Config dict for initialization.
            `type`: The name of our initialization method. Default: 'normal'.
            `gain`: Scaling factor for normal, xavier and orthogonal.
            Default: 0.02.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 base_channels=64,
                 norm_cfg=dict(type='IN'),
                 use_dropout=False,
                 num_blocks=9,
                 padding_mode='reflect'):
        
        super().__init__()
        assert num_blocks >= 0, ('Number of residual blocks must be '
                                 f'non-negative, but got {num_blocks}.')
        assert isinstance(norm_cfg, dict), ("'norm_cfg' should be dict, but"
                                            f'got {type(norm_cfg)}')
        assert 'type' in norm_cfg, "'norm_cfg' must have key 'type'"
        # We use norm layers in the resnet generator.
        # Only for IN, use bias to follow cyclegan's original implementation.
        use_bias = norm_cfg['type'] == 'IN'

        model = []
        model += [
            ConvModule(
                in_channels=in_channels,
                out_channels=base_channels,
                kernel_size=7,
                padding=3,
                bias=use_bias,
                norm_cfg=norm_cfg,
                padding_mode=padding_mode)
        ]

        num_down = 2
        # add downsampling layers
        for i in range(num_down):
            multiple = 2**i
            model += [
                ConvModule(
                    in_channels=base_channels * multiple,
                    out_channels=base_channels * multiple * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=use_bias,
                    norm_cfg=norm_cfg)
            ]

        # add residual blocks
        multiple = 2**num_down
        for i in range(num_blocks):
            model += [
                ResidualBlockWithDropout(
                    base_channels * multiple,
                    padding_mode=padding_mode,
                    norm_cfg=norm_cfg,
                    use_dropout=use_dropout)
            ]

        # add upsampling layers
        for i in range(num_down):
            multiple = 2**(num_down - i)
            model += [
                ConvModule(
                    in_channels=base_channels * multiple,
                    out_channels=base_channels * multiple // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=use_bias,
                    conv_cfg=dict(type='ConvTranspose2d', output_padding=1),
                    norm_cfg=norm_cfg)
            ]

        model += [
            ConvModule(
                in_channels=base_channels,
                out_channels=out_channels,
                kernel_size=7,
                padding=3,
                bias=True,
                norm_cfg=None,
                act_cfg=dict(type='Tanh'),
                padding_mode=padding_mode)
        ]

        self.model = nn.Sequential(*model)

        self.init_weights()

    def forward(self, x):

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


# --- new: bidirectional wrapper with X→Y and Y→X and cycle helpers ---
class BiResnetGenerators(nn.Module):
    """Two ResnetGenerators: one for X→Y and one for Y→X.

    Exposes:
      - x2y(x): generate Y from X
      - y2x(y): generate X from Y
      - xyx(x): cycle X→Y→X
      - yxy(y): cycle Y→X→Y
    """
    def __init__(self,
                 x_channels: int,
                 y_channels: int,
                 base_channels: int = 64,
                 norm_cfg=dict(type='IN'),
                 use_dropout: bool = False,
                 num_blocks: int = 9,
                 padding_mode: str = 'reflect'):
        super().__init__()
        # G_X2Y: maps X (x_channels) -> Y (y_channels)
        self.G_X2Y = ResnetGenerator(
            in_channels=x_channels,
            out_channels=y_channels,
            base_channels=base_channels,
            norm_cfg=norm_cfg,
            use_dropout=use_dropout,
            num_blocks=num_blocks,
            padding_mode=padding_mode,
        )
        # G_Y2X: maps Y (y_channels) -> X (x_channels)
        self.G_Y2X = ResnetGenerator(
            in_channels=y_channels,
            out_channels=x_channels,
            base_channels=base_channels,
            norm_cfg=norm_cfg,
            use_dropout=use_dropout,
            num_blocks=num_blocks,
            padding_mode=padding_mode,
        )

    # --- interfaces you asked for ---
    def x2y(self, x):
        """X → Y"""
        return self.G_X2Y(x)

    def y2x(self, y):
        """Y → X"""
        return self.G_Y2X(y)

    def xyx(self, x):
        """X → Y → X (cycle)"""
        return self.G_Y2X(self.G_X2Y(x))

    def yxy(self, y):
        """Y → X → Y (cycle)"""
        return self.G_X2Y(self.G_Y2X(y))

    def forward(self, tensor, mode: str = 'x2y', *, include_identity: bool = False):
        if mode == 'x2y':
            return self.x2y(tensor)
        elif mode == 'y2x':
            return self.y2x(tensor)
        elif mode == 'xyx':
            return self.xyx(tensor)
        elif mode == 'yxy':
            return self.yxy(tensor)
        elif mode == 'full':
            # Expect a pair (x, y)
            if not (isinstance(tensor, (tuple, list)) and len(tensor) == 2):
                raise ValueError("For mode='full', pass a (x, y) tuple/list.")
            x, y = tensor

            # One top-level forward call -> multiple internal submodule calls
            y_hat = self.G_X2Y(x)        # X -> Y
            x_hat = self.G_Y2X(y)        # Y -> X
            x_cyc = self.G_Y2X(y_hat)    # X -> Y -> X
            y_cyc = self.G_X2Y(x_hat)    # Y -> X -> Y

            if include_identity:
                id_y = self.G_X2Y(y)     # identity on Y
                id_x = self.G_Y2X(x)     # identity on X
                # Return identities as a convenience (keeps single DDP forward)
                return y_hat, x_hat, x_cyc, y_cyc, id_y, id_x

            # Return the four main outputs when include_identity=False
            return y_hat, x_hat, x_cyc, y_cyc
        else:
            raise ValueError(f"Unknown mode '{mode}'. Expected one of: "
                             f"'x2y', 'y2x', 'xyx', 'yxy', 'full'.")
