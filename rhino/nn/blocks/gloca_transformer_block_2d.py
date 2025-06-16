import torch
import torch.nn as nn
import torch.nn.functional as F
from rhino.nn.blocks.mlp import TwoLayerMLP
from rhino.nn.blocks.wmsa import WindowMultiheadSelfAttention
from rhino.nn.pe.relative_position_bias_2d import RelativePositionBias2D

from einops import rearrange
    
class GlocaTransformerBlock2D(nn.Module):
    
    def __init__(self, dim, num_heads, image_size, patch_size, mlp_ratio, qkv_bias, is_local=False): 
        
        super().__init__() 
        self.dim = dim 
        self.num_heads = num_heads 
        self.image_size = image_size 
        self.patch_size = patch_size 
        self.mlp_ratio = mlp_ratio
        self.is_local = is_local

        patch_res = (image_size[0] // self.patch_size[0], image_size[1] // self.patch_size[1])
 
        self.norm1 = nn.LayerNorm(dim) 
        self.norm2 = nn.LayerNorm(dim)

        if self.is_local: # Local
            self.rpe = RelativePositionBias2D(window_size=self.patch_size, 
                                            num_heads=num_heads, init_std=0.02)
            self.attn = WindowMultiheadSelfAttention( 
                dim=dim, num_heads=num_heads,
                qkv_bias=qkv_bias, 
            )
        else:       # global
            self.rpe = RelativePositionBias2D(window_size=patch_res, 
                                            num_heads=num_heads, init_std=0.02)
            self.attn = WindowMultiheadSelfAttention( 
                dim=dim, num_heads=num_heads,
                qkv_bias=qkv_bias, 
            )

            high_dim = dim * self.patch_size[0] * self.patch_size[1]
            self.fc_down = nn.Linear(high_dim, dim)
            self.fc_up = nn.Linear(dim, high_dim)

        mlp_hidden_dim = int(dim * mlp_ratio) 
        self.mlp = TwoLayerMLP(in_features=dim, mid_features=mlp_hidden_dim, act_layer=nn.GELU)

    def forward(self, x): 
        B, H, W, C = x.shape 
        
        patch_res = (H // self.patch_size[0], W // self.patch_size[1])

        shortcut = x
        x = self.norm1(x)

        x = rearrange(x, 'B H W C -> B C H W')
        
        # B, C*self.patch_size*self.patch_size, patch_res*patch_res
        x = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)              

        rpe = self.rpe()

        if self.is_local:
            x = rearrange(x, 'B (C P0 P1) (R0 R1) -> (B R0 R1) (P0 P1) C', C=C,
                          P0=self.patch_size[0], P1=self.patch_size[1], 
                          R0=patch_res[0], R1=patch_res[1])
            rpe = rpe[..., :self.patch_size[0]*self.patch_size[1], :self.patch_size[0]*self.patch_size[1]]
            x = self.attn(x, mask=None, rpe=rpe)
        else:
            x = rearrange(x, 'B (C P0 P1) (R0 R1) -> B (R0 R1) (P0 P1 C)', C=C,
                          P0=self.patch_size[0], P1=self.patch_size[1], 
                          R0=patch_res[0], R1=patch_res[1])
            rpe = rpe[..., :patch_res[0]*patch_res[1], :patch_res[0]*patch_res[1]]
            
            x = self.fc_down(x)
            x = self.attn(x, mask=None, rpe=rpe)
            x = self.fc_up(x)

       

        if self.is_local:
            x = rearrange(x, '(B R0 R1) (P0 P1) C -> B (C P0 P1) (R0 R1)', C=C,
                          P0=self.patch_size[0], P1=self.patch_size[1], 
                          R0=patch_res[0], R1=patch_res[1])
        else:
            x = rearrange(x, 'B (R0 R1) (P0 P1 C) -> B (C P0 P1) (R0 R1)', C=C,
                          P0=self.patch_size[0], P1=self.patch_size[1], 
                          R0=patch_res[0], R1=patch_res[1])
        
        x = F.fold(x, output_size=(H, W), kernel_size=self.patch_size, stride=self.patch_size)

        x = rearrange(x, 'B C H W -> B H W C')

        # First residual connection
        x = shortcut + x
        
        # MLP block
        x = x + self.mlp(self.norm2(x))
 
        return x