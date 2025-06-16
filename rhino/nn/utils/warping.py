# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F

# Global cache for grids
_GRID_CACHE = {}

def _get_cached_grid(h, w, device, dtype):
    """Get or create a cached coordinate grid."""
    cache_key = (h, w, device, dtype)
    
    if cache_key not in _GRID_CACHE:
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, h, device=device, dtype=dtype),
            torch.arange(0, w, device=device, dtype=dtype),
            indexing='ij')
        
        grid = torch.stack((grid_x, grid_y), 2)  # h, w, 2
        grid.requires_grad = False
        _GRID_CACHE[cache_key] = grid
    
    return _GRID_CACHE[cache_key]

def flow_warp(x, flow,
              interp_mode='bilinear',
              padding_mode='zeros',
              align_corners=True):
    """Warp an image or a feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
            a two-channel, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interp_mode (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.

    Returns:
        Tensor: Warped image or feature map.
    """
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                         f'flow ({flow.size()[1:3]}) are not the same.')
    _, _, h, w = x.size()
    
    # Get cached grid
    grid = _get_cached_grid(h, w, flow.device, x.dtype)
    
    grid_flow = grid + flow

    # if interp_mode == 'nearest4': # todo: bug, no gradient for flow model in this case!!! but the result is good
    #     vgrid_x_floor = 2.0 * torch.floor(grid_flow[:, :, :, 0]) / max(w - 1, 1) - 1.0
    #     vgrid_x_ceil = 2.0 * torch.ceil(grid_flow[:, :, :, 0]) / max(w - 1, 1) - 1.0
    #     vgrid_y_floor = 2.0 * torch.floor(grid_flow[:, :, :, 1]) / max(h - 1, 1) - 1.0
    #     vgrid_y_ceil = 2.0 * torch.ceil(grid_flow[:, :, :, 1]) / max(h - 1, 1) - 1.0

    #     output00 = F.grid_sample(x, torch.stack((vgrid_x_floor, vgrid_y_floor), dim=3), 
    #                              mode='nearest', 
    #                              padding_mode=padding_mode, 
    #                              align_corners=align_corners)
    #     output01 = F.grid_sample(x, torch.stack((vgrid_x_floor, vgrid_y_ceil), dim=3), 
    #                              mode='nearest', 
    #                              padding_mode=padding_mode, 
    #                              align_corners=align_corners)
    #     output10 = F.grid_sample(x, torch.stack((vgrid_x_ceil, vgrid_y_floor), dim=3), 
    #                              mode='nearest', 
    #                              padding_mode=padding_mode, 
    #                              align_corners=align_corners)
    #     output11 = F.grid_sample(x, torch.stack((vgrid_x_ceil, vgrid_y_ceil), dim=3), 
    #                              mode='nearest', 
    #                              padding_mode=padding_mode, 
    #                              align_corners=align_corners)

    #     return torch.cat([output00, output01, output10, output11], 1)

    # scale grid_flow to [-1,1]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    grid_flow = grid_flow.type(x.type())

    output = F.grid_sample(x, grid_flow, mode=interp_mode,
                           padding_mode=padding_mode,
                           align_corners=align_corners)
    return output

# Optional: Function to clear cache if memory becomes a concern
def clear_grid_cache():
    """Clear the grid cache to free memory."""
    global _GRID_CACHE
    _GRID_CACHE.clear()