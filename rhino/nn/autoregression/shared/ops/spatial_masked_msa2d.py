import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from rhino.nn.shared.channel_layer_norm import ChannelLayerNorm

class CausalSelfAttention2D(nn.Module):
    """
    Multi-head self-attention over spatial positions with a raster-order mask.
    We **allow attending to the current position** (<=) so that the first token
    still has a valid key. Channel ordering within the current pixel remains
    enforced by MaskedConv2d layers in the conv path.

    Input/Output: (B, C, H, W)
    """
    def __init__(
        self,
        channels: int,
        heads: int = 8,
        attn_dim: int | None = None,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
    ):
        super().__init__()
        self.channels = channels
        self.heads = heads
        d_model = channels if attn_dim is None else attn_dim
        assert d_model % heads == 0, "attn_dim must be divisible by heads"

        self.qkv = nn.Conv2d(channels, 3 * d_model, kernel_size=1, bias=True)
        self.proj = nn.Conv2d(d_model, channels, kernel_size=1, bias=True)

        self.attn_drop = nn.Dropout(attn_dropout)
        self.resid_drop = nn.Dropout(resid_dropout)

        self.d_model = d_model
        self.d_head = d_model // heads

        # simple absolute positional biases for H and W (broadcast add)
        self.pos_h = nn.Parameter(torch.zeros(1, heads, 1, 1))
        self.pos_w = nn.Parameter(torch.zeros(1, heads, 1, 1))

        self.norm_q = ChannelLayerNorm(channels)
        # self.norm_kv = ChannelLayerNorm(channels)

    @staticmethod
    def _make_causal_spatial_mask(H: int, W: int, device=None) -> torch.Tensor:
        """
        Returns (L, L) boolean mask where True entries are KEEP (allowed to attend),
        and False entries are BLOCK (future positions). L = H * W.
        We use <= (allow self) to avoid empty rows for the first position.
        """
        L = H * W
        idx = torch.arange(L, device=device)
        # Raster order index i = y * W + x
        # Allow j <= i (previous or current), disallow j > i (future)
        mask = (idx[None, :] <= idx[:, None])
        return mask  # (L, L), bool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        L = H * W
        device = x.device

        # Pre-norm for stability
        q_in = self.norm_q(x)
        # kv_in = self.norm_kv(x)

        q, k, v = torch.split(self.qkv(q_in), [self.d_model, self.d_model, self.d_model], dim=1)
        # Flatten spatial
        def reshape_heads(t):
            t = t.reshape(B, -1, H * W)  # (B, d_model, L)
            t = t.view(B, self.heads, self.d_head, L)  # (B, h, d, L)
            return t

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        # scaled dot-product attention
        attn_scores = torch.einsum('bhdi,bhdj->bhij', q, k) / math.sqrt(self.d_head)  # (B,h,L,L)

        # positional biases (broadcast)
        attn_scores = attn_scores + self.pos_h + self.pos_w

        causal = self._make_causal_spatial_mask(H, W, device=device)  # (L, L)
        
        # dtype-aware big negative for masked positions (fp16-safe)
        neg_large = torch.finfo(attn_scores.dtype).min
        attn_scores = attn_scores.masked_fill(~causal[None, None, :, :], neg_large)

        attn = torch.softmax(attn_scores, dim=-1)

        out = torch.einsum('bhij,bhdj->bhdi', attn, v)  # (B,h,d,L)
        out = out.reshape(B, self.d_model, L).view(B, self.d_model, H, W)
        out = self.proj(out)
        out = self.resid_drop(out)
        return out
