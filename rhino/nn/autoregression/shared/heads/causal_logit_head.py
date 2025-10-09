import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from rhino.nn.autoregression.shared.ops.masked_conv2d import MaskedConv2d

class CausalLogitHead(nn.Module):
    """
    Channel-causal discrete head WITHOUT internal embeddings.
    - Inputs:
        h: (B, D, H, W)  spatial-causal trunk features (safe at center)
        target_embs: (B, E*C, H, W), interleaved as rgb,rgb,rgb,...
    - Outputs:
        logits: (B, V, C, H, W)
    """
    def __init__(self,
                 trunk_channels: int,
                 C: int,
                 K: int,
                 d_embed: int,          # must match target_embs grouped size per channel
                 h_channels: int = 256):
        super().__init__()
        self.C = C
        self.K = K
        self.dE = d_embed
        self.hc = h_channels

        # -------- type maps (rgb,rgb,rgb,...) --------
        # target_embs channels are grouped as C per embed dim: [R0,G0,B0, R1,G1,B1, ...]
        t_in_targets = torch.arange(C).repeat(d_embed)              # length dE*C: 0,1,2,0,1,2,...

        # hidden channels use interleaved types 0,1,2,0,1,2,...
        t_hidden = torch.arange(h_channels) % C

        # fuse input is [hidden_from_targets, trunk_proj]; make trunk type=0 so it's
        # visible to ALL outputs at center under Mask 'B' (since 0 <= out_type).
        t_in_fuse = torch.cat([t_hidden, torch.zeros(h_channels, dtype=torch.long)])

        # final logits channels are (C*K) laid out as [R,G,B, R,G,B, ...]
        t_out_logits = torch.arange(C).repeat(K)                    # length C*K

        # -------- layers --------
        # 1) From target_embs -> hidden (Mask 'A': center uses strict < channel order)
        self.conv_in = MaskedConv2d(
            'A', C * d_embed, h_channels, kernel_size=1, n_types=C,
            type_map_in=t_in_targets, type_map_out=t_hidden
        )

        # 2) Trunk projection (safe for all channels; no gating here)
        self.trunk_proj = nn.Conv2d(trunk_channels, h_channels, kernel_size=1, bias=True)

        # 3) Fuse hidden-from-targets with trunk (Mask 'B': center uses <= channel order)
        self.conv_fuse = MaskedConv2d(
            'B', 2 * h_channels, h_channels, kernel_size=1, n_types=C,
            type_map_in=t_in_fuse, type_map_out=t_hidden
        )

        # 4) Produce logits (still Mask 'B', typed per (channel, class))
        self.conv_out = MaskedConv2d(
            'B', h_channels, C * K, kernel_size=1, n_types=C,
            type_map_out=t_out_logits
        )

        # 1x1 to match arbitrary trunk width if needed
        self._match_trunk = (trunk_channels != h_channels)
        if self._match_trunk:
            self.trunk_proj = nn.Conv2d(trunk_channels, h_channels, kernel_size=1, bias=True)

    def forward(self, h: torch.Tensor, target_embs: torch.Tensor) -> torch.Tensor:
        """
        target_embs must be interleaved as [R0,G0,B0, R1,G1,B1, ...] with size B, E*C, H, W
        Returns logits shaped (B, V, C, H, W).
        """
        B, _, H, W = h.shape
        expected = self.C * self.dE
        assert target_embs.shape[1] == expected, \
            f"target_embs has {target_embs.shape[1]} channels, expected C*d_embed={expected}"

        # Avoid backprop into target_embs if you treat them as fixed teacher-forcing signals:
        # target_embs = target_embs.detach()

        hidden_from_tgt = F.relu(self.conv_in(target_embs))   # Mask 'A' enforces channel causality from targets
        trunk_h = self.trunk_proj(h)                          # plain 1x1; safe for all channels
        fuse = torch.cat([hidden_from_tgt, trunk_h], dim=1)

        fuse = F.relu(self.conv_fuse(fuse))                   # Mask 'B'
        logits = self.conv_out(fuse)                          # (B, V*C, H, W), interleaved [R,G,B, R,G,B, ...]

        # reshape to (B, V, C, H, W) to use CE per channel easily
        logits = rearrange(logits, 'b (v c) h w -> b v c h w', c=self.C, v=self.K)
        return logits
