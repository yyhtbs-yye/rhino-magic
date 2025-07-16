import torch
import torch.nn as nn
import torch.nn.functional as F
from rhino.nn.minimals.mlp import TwoLayerMLP
from rhino.nn.minimals.wmsa import WindowMultiheadSelfAttention
from rhino.nn.pe.relative_position_bias_2d import RelativePositionBias2D

from einops import rearrange
    
class ViTBlock2D(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size, patch_size, 
                 mlp_ratio, qkv_bias, trigger=None, 
                 enable_condition=False, cond_dim=None): 
        
        super().__init__() 
        self.embed_dim = embed_dim 
        self.num_heads = num_heads 
        self.window_size = window_size 
        self.patch_size = patch_size 
        self.mlp_ratio = mlp_ratio

        patch_res = (window_size[0] // self.patch_size[0], window_size[1] // self.patch_size[1])
 
        self.norm1 = nn.LayerNorm(embed_dim) 
        self.norm2 = nn.LayerNorm(embed_dim)

        self.attn = WindowMultiheadSelfAttention( 
            embed_dim=embed_dim, num_heads=num_heads,
            qkv_bias=qkv_bias, 
        )
        self.rpe = RelativePositionBias2D(window_size=patch_res, 
                                        num_heads=num_heads, init_std=0.02)

        high_dim = embed_dim * self.patch_size[0] * self.patch_size[1]
        self.fc_down = nn.Linear(high_dim, embed_dim)
        self.fc_up = nn.Linear(embed_dim, high_dim)

        mlp_hidden_dim = int(embed_dim * mlp_ratio) 
        self.mlp = TwoLayerMLP(in_features=embed_dim, mid_features=mlp_hidden_dim, act_layer=nn.SiLU)

        self.enable_condition = enable_condition
        if enable_condition:
            self.cond_dim = cond_dim
            self.cond_proj =nn.Linear(cond_dim, embed_dim * 2) if cond_dim is not None else None
        else:
            self.cond_dim = None
            self.cond_proj = None

    def forward(self, x, cond=None): 
        B, H, W, C = x.shape 
        
        patch_res = (H // self.patch_size[0], W // self.patch_size[1])

        shortcut = x
        x = self.norm1(x)

        x = rearrange(x, 'B H W C -> B C H W')
        
        # B, C*self.patch_size*self.patch_size, patch_res*patch_res
        x = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)              

        rpe = self.rpe()

        # Inter-patch attention
        x = rearrange(x, 'B (C P0 P1) (R0 R1) -> B (R0 R1) (P0 P1 C)', C=C,
                      P0=self.patch_size[0], P1=self.patch_size[1], 
                      R0=patch_res[0], R1=patch_res[1])
        rpe = rpe[..., :patch_res[0]*patch_res[1], :patch_res[0]*patch_res[1]]
        
        x = self.fc_down(x)
        x = self.attn(x, mask=None, rpe=rpe)
        x = self.fc_up(x)

        x = rearrange(x, 'B (R0 R1) (P0 P1 C) -> B (C P0 P1) (R0 R1)', C=C,
                      P0=self.patch_size[0], P1=self.patch_size[1], 
                      R0=patch_res[0], R1=patch_res[1])
        
        x = F.fold(x, output_size=(H, W), kernel_size=self.patch_size, stride=self.patch_size)

        x = rearrange(x, 'B C H W -> B H W C')

        # First residual connection
        x = shortcut + x

        mlp_shortcut = x

        x = self.mlp(self.norm2(x))

        if self.enable_condition is True:
            assert self.cond_proj is not None and cond is not None
            # Apply conditional scaling and shifting
            scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
            scale = scale[:, None, None, :]  # broadcast to H × W
            shift = shift[:, None, None, :]

            # MLP block
            x = mlp_shortcut + x * (1 + scale) + shift
        else:
            # MLP block without condition
            x = mlp_shortcut + x
        return x