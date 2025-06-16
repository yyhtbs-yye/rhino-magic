import torch
import torch.nn as nn

class WindowMultiheadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # QKV and output projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None, pe=None, rpe=None):
        """
        x: (B_*windows, N, C)
        mask: optional attention mask, shape (num_windows, N, N)
        """
        B_, N, C = x.shape
        
        # add positional bias
        if pe is not None:
            x = x + pe
        
        qkv = (self.qkv(x)                                              # (B_*windows, N, 3*C)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)     # (B_*windows, N, 3, nH, C//nH)
            .permute(2, 0, 3, 1, 4)                                     # (3, B_*windows, nH, N, C//nH)
            .contiguous()   
        )
        q, k, v = qkv[0], qkv[1], qkv[2]                                # (B_*windows, nH, N, C//nH)

        # scaled dot-product attention
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)                                  # (B_*windows, nH, N, N)

        # add relative positional bias
        if rpe is not None:
            attn = attn + rpe                                           # (B_*windows, nH, N, N)

        # apply mask if given
        if mask is not None:
            nW = mask.shape[0]
            attn = (
                attn.view(B_ // nW, nW, self.num_heads, N, N)
                + mask.unsqueeze(1).unsqueeze(0)
            ).view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)                                       # (B_*windows, nH, N, N)

        # attention output
        x = (attn @ v                                                   # (B_*windows, nH, N, C//nH)
                ).transpose(1, 2                                        # (B_*windows, N, nH, C//nH)
                    ).reshape(B_, N, C)                                 # (B_*windows, N, C)
        x = self.proj(x)                                                # (B_*windows, N, C)
        return x

