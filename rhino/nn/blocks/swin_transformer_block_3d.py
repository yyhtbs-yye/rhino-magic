import torch
import torch.nn as nn

from rhino.nn.minimals.mlp import TwoLayerMLP
from rhino.nn.minimals.wmsa import WindowMultiheadSelfAttention
from rhino.nn.minimals.pe.relative_position_bias_3d import RelativePositionBias3D

from rhino.nn.utils import windowing, masking
    
class SwinTransformerBlock3D(nn.Module):
    
    def __init__(self, embed_dim, num_heads, window_size, patch_size, 
                 mlp_ratio, qkv_bias, trigger=False, 
                 enable_condition=False, cond_dim=None): 
        
        super().__init__() 
        self.embed_dim = embed_dim 
        self.num_heads = num_heads 
        self.window_size = window_size
        self.patch_size = patch_size
        self.shift_size = (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2) if trigger else (0, 0, 0)
        self.mlp_ratio = mlp_ratio
        self.trigger = trigger
 
        self.norm1 = nn.LayerNorm(embed_dim) 
        self.norm2 = nn.LayerNorm(embed_dim)
 
        self.attn = WindowMultiheadSelfAttention( 
            embed_dim=embed_dim, num_heads=num_heads, 
            qkv_bias=qkv_bias, 
        )
        self.rpe = RelativePositionBias3D(window_size=self.window_size, 
                                          num_heads=num_heads, init_std=0.02)
 
        mlp_hidden_dim = int(embed_dim * mlp_ratio) 
        self.mlp = TwoLayerMLP(in_features=embed_dim, mid_features=mlp_hidden_dim, act_layer=nn.GELU)
        self.attn_mask = dict() 
 
        self.enable_condition = enable_condition
        if enable_condition:
            self.cond_dim = cond_dim
            self.cond_proj =nn.Linear(cond_dim, embed_dim * 2) if cond_dim is not None else None
        else:
            self.cond_dim = None
            self.cond_proj = None

    def forward(self, x, cond=None):
        B, T, H, W, C = x.shape 
        
        shortcut = x
        x = self.norm1(x)
        
        # Apply cyclic shift if this is a shifted block
        if self.trigger:
            skey = (T, H, W, *self.window_size, *self.shift_size)
            if skey not in self.attn_mask:
                self.attn_mask[skey] = masking.compute_mask_3d((T, H, W), tuple(self.window_size), self.shift_size).to(x.device)
            
            x = torch.roll(
                x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
            mask = self.attn_mask[skey]
        else:
            mask = None
 
        # Window partition and attention
        x = windowing.window_partition_3d(x, self.window_size) 
        
        rpe = self.rpe()
        x = self.attn(x, mask=mask, rpe=rpe)

        x = windowing.window_reverse_3d(x.view(-1, *self.window_size, C), self.window_size, B, T, H, W)
        
        # Reverse cyclic shift if this was a shifted block
        if self.trigger:
            x = torch.roll(x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))
        
        # First residual connection
        x = shortcut + x

        mlp_shortcut = x

        x = self.mlp(self.norm2(x))

        if self.enable_condition is True:
            assert self.cond_proj is not None and cond is not None
            # Apply conditional scaling and shifting
            scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
            scale = scale[:, None, None, None, :]  # broadcast to T × H × W
            shift = shift[:, None, None, None, :]

            # MLP block
            x = mlp_shortcut + x * (1 + scale) + shift
        else:
            # MLP block without condition
            x = mlp_shortcut + x
        return x