import torch
import torch.nn as nn
from timm.layers import trunc_normal_
from rhino.nn.blocks.rtgb import RTGB
from rhino.nn.blocks.gloca_transformer_block_2d import GlocaTransformerBlock2D

from einops import rearrange

class GlocaViT2D(nn.Module):
    """
    Gloca Vision Transformer backbone using RTGB stages on 2D image data.
    Input:  (B, embed_dim, H, W)
    Output: (B, embed_dim, H, W)
    """
    def __init__(self,
                 embed_dim  = 64,
                 depths     = (4, 4, 4, 4),
                 num_heads  = (4, 4, 4, 4),
                 image_size = (64, 64),
                 patch_size = (8, 8),
                 mlp_ratio  = 4.,
                 qkv_bias   = True):
        super().__init__()

        self.pos_drop = nn.Dropout(p=0.)

        # build residual gloca‐style stages
        self.stages = nn.ModuleList([
            RTGB(
                block_cls   = GlocaTransformerBlock2D,
                embed_dim   = embed_dim,
                depth       = d,
                num_heads   = h,
                image_size  = image_size,
                patch_size  = patch_size,
                mlp_ratio   = mlp_ratio,
                qkv_bias    = qkv_bias
            ) for d, h in zip(depths, num_heads)
        ])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (B, in_channels, H, W)
        B, C, H, W = x.shape

        x = rearrange(x, 'b c h w -> b h w c')

        x = self.pos_drop(x)

        # pass through all RTGB stages
        for stage in self.stages:
            x = stage(x)

        # back to (B, C, H', W')
        x = rearrange(x, 'b h w c -> b c h w', h=H, w=W)
        return x
    

if __name__=="__main__":
    # Instantiate model
    model = GlocaViT2D(
        embed_dim=64,
        num_heads=[4, 4, 4],
        depths=(2, 2, 2),           # Reduced depth for quick test
        image_size=(256, 256),
        patch_size=(16, 16),
    )

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Create dummy input: (batch_size, channels, height, width)
    B, C, H, W = 2, 64, 256, 256  # You can modify sizes for your own tests
    dummy_input = torch.randn(B, C, H, W).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    # Verify output
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    assert output.shape == dummy_input.shape, "Output shape does not match input shape!"

    print("✅ Forward pass successful!")
