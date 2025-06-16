# relative_pos_bias_2d.py

import torch
import torch.nn as nn
from timm.layers import trunc_normal_

class RelativePositionBias2D(nn.Module):
    def __init__(self, window_size, num_heads, init_std=0.02):
        """
        window_size: tuple of ints (Wh, Ww)
        num_heads: number of attention heads
        """
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads

        # create a parameter table of shape [num_relative_positions, num_heads]
        num_rel_positions = ((2 * window_size[0] - 1) * (2 * window_size[1] - 1))
        self.relative_position_bias_table = nn.Parameter(torch.zeros(num_rel_positions, num_heads))

        # compute and register the relative position index buffer
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w), dim=0)  # (2, Wh, Ww)
        coords_flat = coords.flatten(1)  # (2, N)
        # compute pairwise relative coords, shift to non‐negative
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, N, N)
        rel = rel.permute(1, 2, 0).contiguous()  # (N, N, 2)
        rel[..., 0] += window_size[0] - 1
        rel[..., 1] += window_size[1] - 1
        rel[..., 0] *= (2 * window_size[1] - 1)
        rel_index = rel.sum(-1)  # (N, N)
        self.register_buffer("relative_position_index", rel_index)

        # init
        trunc_normal_(self.relative_position_bias_table, std=init_std)

    def forward(self):
        """
        Returns:
            bias: tensor of shape (1, num_heads, N, N)
        """
        N = (
            self.window_size[0]
            * self.window_size[1]
        )
        # index into the table and reshape
        bias = self.relative_position_bias_table[
            self.relative_position_index[:N, :N].reshape(-1)
        ]
        bias = bias.reshape(N, N, self.num_heads).permute(2, 0, 1)  # (nH, N, N)
        return bias.unsqueeze(0)  # (1, nH, N, N)