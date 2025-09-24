import torch
import torch.nn as nn
import torch.nn.functional as F

from .pix2pix_modules import UnetSkipConnectionBlock

class UnetGenerator(nn.Module):
    """Construct the Unet-based generator from the innermost layer to the
    outermost layer, which is a recursive process.

    Args:
        in_channels (int): Number of channels in input images.
        out_channels (int): Number of channels in output images.
        num_down (int): Number of downsamplings in Unet. If `num_down` is 8,
            the image with size 256x256 will become 1x1 at the bottleneck.
            Default: 8.
        base_channels (int): Number of channels at the last conv layer.
            Default: 64.
        norm_cfg (dict): Config dict to build norm layer. Default:
            `dict(type='BN')`.
        use_dropout (bool): Whether to use dropout layers. Default: False.
        init_cfg (dict): Config dict for initialization.
            `type`: The name of our initialization method. Default: 'normal'.
            `gain`: Scaling factor for normal, xavier and orthogonal.
            Default: 0.02.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_down=8,
                 base_channels=64,
                 norm_cfg=dict(type='GN'),
                 use_dropout=False):
        super().__init__()
        # We use norm layers in the unet generator.
        assert isinstance(norm_cfg, dict), ("'norm_cfg' should be dict, but"
                                            f'got {type(norm_cfg)}')
        assert 'type' in norm_cfg, "'norm_cfg' must have key 'type'"

        # add the innermost layer
        unet_block = UnetSkipConnectionBlock(
            base_channels * 8,
            base_channels * 8,
            in_channels=None,
            submodule=None,
            norm_cfg=norm_cfg,
            is_innermost=True)
        # add intermediate layers with base_channels * 8 filters
        for _ in range(num_down - 5):
            unet_block = UnetSkipConnectionBlock(
                base_channels * 8,
                base_channels * 8,
                in_channels=None,
                submodule=unet_block,
                norm_cfg=norm_cfg,
                use_dropout=use_dropout)
        # gradually reduce the number of filters
        # from base_channels * 8 to base_channels
        unet_block = UnetSkipConnectionBlock(
            base_channels * 4,
            base_channels * 8,
            in_channels=None,
            submodule=unet_block,
            norm_cfg=norm_cfg)
        unet_block = UnetSkipConnectionBlock(
            base_channels * 2,
            base_channels * 4,
            in_channels=None,
            submodule=unet_block,
            norm_cfg=norm_cfg)
        unet_block = UnetSkipConnectionBlock(
            base_channels,
            base_channels * 2,
            in_channels=None,
            submodule=unet_block,
            norm_cfg=norm_cfg)
        # add the outermost layer
        self.model = UnetSkipConnectionBlock(
            out_channels,
            base_channels,
            in_channels=in_channels,
            submodule=unet_block,
            is_outermost=True,
            norm_cfg=norm_cfg)

        self.init_weights()

    def forward(self, noise, cond):
        """Forward function.

        Args:
            noise (Tensor): Input tensor with shape (n, c, h, w).
            cond (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        x = torch.concat([noise, cond], dim=1)
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
