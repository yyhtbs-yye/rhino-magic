import torch
import torch.nn as nn

from rhino.nn.blocks.mlp import TwoLayerMLP
from rhino.nn.blocks.wmsa import WindowMultiheadSelfAttention
from rhino.nn.pe.relative_position_bias_3d import RelativePositionBias3D

from rhino.nn.utils import windowing, masking
    
class SwinTransformerBlock3D(nn.Module):
    
    def __init__(self, dim, num_heads, window_size, shift_size, mlp_ratio, qkv_bias, shifted=False): 
        
        super().__init__() 
        self.dim = dim 
        self.num_heads = num_heads 
        self.window_size = window_size 
        self.shift_size = shift_size 
        self.mlp_ratio = mlp_ratio
        self.shifted = shifted
 
        self.norm1 = nn.LayerNorm(dim) 
        self.norm2 = nn.LayerNorm(dim)
 
        self.attn = WindowMultiheadSelfAttention( 
            dim=dim, num_heads=num_heads, 
            qkv_bias=qkv_bias, 
        )
        self.rpe = RelativePositionBias3D(window_size=self.window_size, 
                                          num_heads=num_heads, init_std=0.02)
 
        mlp_hidden_dim = int(dim * mlp_ratio) 
        self.mlp = TwoLayerMLP(in_features=dim, mid_features=mlp_hidden_dim, act_layer=nn.GELU)
        self.attn_mask = dict() 
 
    def forward(self, x): 
        B, T, H, W, C = x.shape 
        
        shortcut = x
        x = self.norm1(x)
        
        # Apply cyclic shift if this is a shifted block
        if self.shifted:
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
        if self.shifted:
            x = torch.roll(x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))
        
        # First residual connection
        x = shortcut + x
        
        # MLP block
        x = x + self.mlp(self.norm2(x))
 
        return x
    