# ===== Helper types & builders for ConvModule (PyTorch >= 2.0) =====
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

# Internal base classes used for isinstance checks / typing
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm
from torch.nn.modules.conv import _ConvNd


# ---------------------------- Padding helpers ----------------------------

def _expand_padding_for_dim(padding: Union[int, Tuple[int, ...]], dim: int) -> Tuple[int, ...]:
    """
    Expand common padding specs to PyTorch's explicit (left/right[/top/bottom...]) form.

    For dim == 1: (l, r)
    For dim == 2: (l, r, t, b)
    For dim == 3: (l, r, t, b, f, bk)  # width, height, depth order expected by PyTorch
    """
    if isinstance(padding, int):
        if dim == 1:
            return (padding, padding)
        elif dim == 2:
            return (padding, padding, padding, padding)
        elif dim == 3:
            return (padding, padding, padding, padding, padding, padding)

    # Tuple cases
    assert isinstance(padding, tuple), "padding must be int or tuple"
    if dim == 1:
        if len(padding) == 1:
            return (padding[0], padding[0])
        elif len(padding) == 2:
            return (padding[0], padding[1])
        else:
            raise ValueError("1D padding expects int or tuple of length 1 or 2.")
    elif dim == 2:
        if len(padding) == 2:
            # (pad_w, pad_h) -> (l, r, t, b)
            pw, ph = padding
            return (pw, pw, ph, ph)
        elif len(padding) == 4:
            # already (l, r, t, b)
            return padding
        else:
            raise ValueError("2D padding expects int, (w, h) or (l, r, t, b).")
    elif dim == 3:
        if len(padding) == 3:
            # (pw, ph, pd) -> (l, r, t, b, f, bk)
            pw, ph, pd = padding
            return (pw, pw, ph, ph, pd, pd)
        elif len(padding) == 6:
            return padding
        else:
            raise ValueError("3D padding expects int, (w, h, d) or (l, r, t, b, f, bk).")
    else:
        raise ValueError("dim must be 1, 2, or 3.")


def build_padding_layer(pad_cfg: Dict[str, Any],
                        padding: Union[int, Tuple[int, ...]]) -> nn.Module:
    """
    Build an explicit padding layer when nn.Conv*'s built-in padding isn't used.

    Supported:
      - type='reflect'  -> nn.ReflectionPad{1d,2d,3d}
      - type='replicate'-> nn.ReplicationPad{1d,2d,3d}
      - type='constant' -> nn.ConstantPad{1d,2d,3d} (requires value in pad_cfg['value'])
      - type='zeros' or 'circular' -> return nn.Identity() (use Conv's native padding)
    """
    pad_type = pad_cfg.get('type', 'zeros').lower()

    # If the conv will handle padding (zeros/circular), just no-op here
    if pad_type in ('zeros', 'circular'):
        return nn.Identity()

    # Infer dimensionality from the padding tuple length (fallback to 2D)
    dim: int
    if isinstance(padding, int):
        dim = 2
    else:
        dim = {2: 1, 4: 2, 6: 3}.get(len(padding), 2)

    expanded = _expand_padding_for_dim(padding, dim)

    if pad_type == 'reflect':
        if dim == 1:
            return nn.ReflectionPad1d(expanded)
        elif dim == 2:
            return nn.ReflectionPad2d(expanded)
        else:
            return nn.ReflectionPad3d(expanded)
    elif pad_type == 'replicate':
        if dim == 1:
            return nn.ReplicationPad1d(expanded)
        elif dim == 2:
            return nn.ReplicationPad2d(expanded)
        else:
            return nn.ReplicationPad3d(expanded)
    elif pad_type == 'constant':
        value = pad_cfg.get('value', 0.0)
        if dim == 1:
            return nn.ConstantPad1d(expanded, value)
        elif dim == 2:
            return nn.ConstantPad2d(expanded, value)
        else:
            return nn.ConstantPad3d(expanded, value)
    else:
        raise ValueError(f"Unsupported padding type: {pad_type}")


# ---------------------------- Conv builder ----------------------------

_CONV_LAYERS = {
    # standard
    'Conv1d': nn.Conv1d,
    'Conv2d': nn.Conv2d,
    'Conv3d': nn.Conv3d,
    # transposed
    'ConvTranspose1d': nn.ConvTranspose1d,
    'ConvTranspose2d': nn.ConvTranspose2d,
    'ConvTranspose3d': nn.ConvTranspose3d,
}

def build_conv_layer(conv_cfg: Optional[Dict[str, Any]],
                     in_channels: int,
                     out_channels: int,
                     kernel_size: Union[int, Tuple[int, int, int]],
                     stride: Union[int, Tuple[int, int, int]] = 1,
                     padding: Union[int, Tuple[int, int, int]] = 0,
                     dilation: Union[int, Tuple[int, int, int]] = 1,
                     groups: int = 1,
                     bias: bool = True) -> _ConvNd:
    """
    Build a convolution layer from a tiny config dict like:
        conv_cfg = dict(type='Conv2d')  # defaults to Conv2d if None
    """
    if conv_cfg is None:
        conv_type = 'Conv2d'
        extra = {}
    else:
        conv_type = conv_cfg.get('type', 'Conv2d')
        extra = {k: v for k, v in conv_cfg.items() if k != 'type'}

    if conv_type not in _CONV_LAYERS:
        raise ValueError(f"Unsupported conv type: {conv_type}")

    Conv = _CONV_LAYERS[conv_type]
    layer = Conv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        **extra
    )
    return layer


# ---------------------------- Norm builder ----------------------------

_NORM_LAYERS = {
    # batch norms
    'BN1d': nn.BatchNorm1d,
    'BN2d': nn.BatchNorm2d,
    'BN3d': nn.BatchNorm3d,
    'SyncBN': nn.SyncBatchNorm,  # requires process group setup when used distributed
    # instance norms
    'IN1d': nn.InstanceNorm1d,
    'IN2d': nn.InstanceNorm2d,
    'IN3d': nn.InstanceNorm3d,
    # group / layer norms
    'GN': nn.GroupNorm,
    'LN': nn.LayerNorm,
}

def _infer_num_features_for_norm(norm_type: str,
                                 in_channels: int,
                                 out_channels: int,
                                 order: Tuple[str, str, str]) -> int:
    """
    For norms that need num_features (BN/IN/SyncBN), decide whether to use
    in_channels or out_channels depending on layer order.
    (The public ConvModule already does this, but this helper simplifies custom use.)
    """
    # If norm comes after conv -> num_features is out_channels, else in_channels
    return out_channels if order.index('norm') > order.index('conv') else in_channels


def build_norm_layer(norm_cfg: Dict[str, Any],
                     num_features: Optional[int] = None) -> Tuple[str, nn.Module]:
    """
    Build a norm layer. Returns (name, module) to match OpenMMLab's convention.

    Example norm_cfg:
      dict(type='BN', eps=1e-5, momentum=0.1)  # auto-resolves to BN2d unless explicitly BN1d/3d
      dict(type='SyncBN')
      dict(type='IN')         # InstanceNorm2d default
      dict(type='GN', num_groups=32)  # requires num_channels via num_features
      dict(type='LN')         # requires normalized_shape via num_features
    """
    assert isinstance(norm_cfg, dict) and 'type' in norm_cfg, "norm_cfg must have a 'type' key"
    cfg = norm_cfg.copy()
    norm_type = cfg.pop('type')

    # Shorthand aliases
    # 'BN'/'IN' will map to 2D by default unless overridden with BN1d/BN3d/IN1d/IN3d
    if norm_type == 'BN':
        norm_type = 'BN2d'
    if norm_type == 'IN':
        norm_type = 'IN2d'

    if norm_type not in _NORM_LAYERS:
        raise ValueError(f"Unsupported norm type: {norm_type}")

    Norm = _NORM_LAYERS[norm_type]

    # figure required args
    if Norm in (nn.GroupNorm,):
        assert num_features is not None, "GroupNorm requires num_features (channels)"
        num_groups = cfg.pop('num_groups', 32)
        layer = Norm(num_groups=num_groups, num_channels=num_features, **cfg)
        name = 'gn'
    elif Norm in (nn.LayerNorm,):
        assert num_features is not None, "LayerNorm requires normalized_shape (channels or shape)"
        layer = Norm(normalized_shape=num_features, **cfg)
        name = 'ln'
    else:
        # BatchNorm/SyncBN/InstanceNorm family: need num_features
        assert num_features is not None, f"{norm_type} requires num_features"
        layer = Norm(num_features, **cfg)
        # conventional names
        if 'Sync' in norm_type:
            name = 'syncbn'
        elif norm_type.startswith('BN'):
            name = 'bn'
        elif norm_type.startswith('IN'):
            name = 'in'
        else:
            name = 'norm'

    return name, layer


# ---------------------------- Activation builder ----------------------------

_ACT_LAYERS = {
    'ReLU': nn.ReLU,
    'LeakyReLU': nn.LeakyReLU,
    'PReLU': nn.PReLU,
    'ReLU6': nn.ReLU6,
    'ELU': nn.ELU,
    'SELU': nn.SELU,
    'Tanh': nn.Tanh,
    'Sigmoid': nn.Sigmoid,
    'GELU': nn.GELU,
    'Mish': nn.Mish,
    'SiLU': nn.SiLU,        # a.k.a. Swish
    'HSigmoid': nn.Hardsigmoid,  # OpenMMLab alias
    # Note: 'Swish' maps to nn.SiLU
}

def build_activation_layer(act_cfg: Dict[str, Any]) -> nn.Module:
    """
    Build an activation layer from a tiny config dict like:
      dict(type='ReLU', inplace=True)
      dict(type='LeakyReLU', negative_slope=0.1, inplace=True)
      dict(type='Swish')  # alias of SiLU
      dict(type='HSigmoid')
    """
    assert isinstance(act_cfg, dict) and 'type' in act_cfg, "act_cfg must have a 'type' key"
    cfg = act_cfg.copy()
    act_type = cfg.pop('type')

    if act_type == 'Swish':
        act_type = 'SiLU'

    # print(cfg.get('inplace'))
    if act_type not in _ACT_LAYERS:
        raise ValueError(f"Unsupported activation type: {act_type}")

    Act = _ACT_LAYERS[act_type]
    return Act(**cfg)

# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelShufflePack(nn.Module):
    """Pixel Shuffle upsample layer.

    Packs a Conv2d followed by F.pixel_shuffle to upsample.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scale_factor (int): Upsample ratio.
        upsample_kernel (int): Kernel size of the conv layer to expand channels.
    """

    def __init__(self, in_channels: int, out_channels: int, scale_factor: int,
                 upsample_kernel: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel

        self.upsample_conv = nn.Conv2d(
            in_channels,
            out_channels * scale_factor * scale_factor,
            kernel_size=upsample_kernel,
            padding=(upsample_kernel - 1) // 2
        )
        self.init_weights()

    def init_weights(self):
        # Replace mmengine.model.xavier_init with native PyTorch init
        nn.init.xavier_uniform_(self.upsample_conv.weight)
        if self.upsample_conv.bias is not None:
            nn.init.zeros_(self.upsample_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample_conv(x)
        x = F.pixel_shuffle(x, self.scale_factor)
        return x


# Simple "registry" replacement
_UPSAMPLE_LAYERS = {
    'nearest': nn.Upsample,       # will set mode='nearest' below
    'bilinear': nn.Upsample,      # will set mode='bilinear' below
    'pixel_shuffle': PixelShufflePack,
}


def build_upsample_layer(cfg: Dict, *args, **kwargs) -> nn.Module:
    """Build an upsample layer without mmengine.

    Args:
        cfg (dict): Must contain:
            - type (str): Layer type. One of {'nearest', 'bilinear', 'pixel_shuffle'}.
            - layer args: Args needed to instantiate the layer (e.g., in_channels, etc.)
                          For nn.Upsample types, provide `scale_factor` or `size` as usual.
        *args, **kwargs: Forwarded to the layer constructor.

    Returns:
        nn.Module: Instantiated upsample layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if 'type' not in cfg:
        raise KeyError(f'the cfg dict must contain the key "type", but got {cfg}')

    cfg_ = cfg.copy()
    layer_type = cfg_.pop('type')

    upsample_cls = _UPSAMPLE_LAYERS.get(layer_type, None)
    if upsample_cls is None:
        raise KeyError(f'Cannot find layer type "{layer_type}" in UPSAMPLE_LAYERS')

    # If using nn.Upsample, ensure the mode matches the requested type
    if upsample_cls is nn.Upsample:
        # Don't overwrite if user explicitly passed a different mode
        cfg_.setdefault('mode', layer_type)
        # Note: align_corners (for bilinear) can be passed via cfg_ if desired

    layer = upsample_cls(*args, **kwargs, **cfg_)
    return layer
