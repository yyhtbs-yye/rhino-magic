import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_
from rhino.nn.blocks.rtsb import RTSB
from rhino.nn.blocks.swin_transformer_block_2d import SwinTransformerBlock2D
from einops import rearrange

class SwinUNet2D(nn.Module):
    """
    U-Net-style Swin Vision Transformer backbone for 2-D images.
    Input:  (B, C, H, W)
    Output: (B, C, H, W)
    """

    def __init__(
        self,
        embed_dims:   list[int] = [64, 128, 256],
        num_heads:    list[int] = [2, 4, 8],
        depths:       tuple[int, ...] = (4, 4, 4),
        window_size:  tuple[int, int] = (8, 8),
        mlp_ratio:    float = 4.0,
        qkv_bias:     bool = True,
    ):
        super().__init__()

        self.pos_drop = nn.Dropout(p=0.0)
        self.embed_dims = embed_dims
        self.num_heads = num_heads

        self.encoder_stages = nn.ModuleList(
            [
                RTSB(
                    block_cls=SwinTransformerBlock2D,
                    embed_dim=embed_dims[i],
                    depth=depths[i],
                    num_heads=num_heads[i],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                )
                for i in range(len(depths))
            ]
        )

        self.decoder_stages = nn.ModuleList(
            [
                RTSB(
                    block_cls=SwinTransformerBlock2D,
                    embed_dim=embed_dims[len(depths) - 1 - i],
                    depth=depths[len(depths) - 1 - i],
                    num_heads=num_heads[len(depths) - 1 - i],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                )
                for i in range(len(depths))
            ]
        )

        self.down_projs = nn.ModuleList(
            [
                nn.Linear(embed_dims[i], embed_dims[i + 1])
                for i in range(len(embed_dims) - 1)
            ]
        )
        self.up_projs = nn.ModuleList(
            [
                nn.Linear(embed_dims[i + 1], embed_dims[i])
                for i in range(len(embed_dims) - 1)
            ]
        )
        self.skip_projs = nn.ModuleList(
            [
                nn.Linear(embed_dims[i], embed_dims[i])
                for i in range(len(embed_dims) - 1)
            ]
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape

        # Flatten spatial dims → sequence and drop positional tokens if any
        x = rearrange(x, "b c h w -> b h w c")
        x = self.pos_drop(x)

        skip_connections: list[torch.Tensor] = []
        spatial_dims: list[tuple[int, int]] = [(H, W)]

        for i, stage in enumerate(self.encoder_stages):
            x = stage(x)

            # Save skips except at the deepest layer
            if i < len(self.encoder_stages) - 1:
                skip_connections.append(x)

                # Down-project channel dim
                x = self.down_projs[i](x)

                # Compute new resolution
                curr_H, curr_W = spatial_dims[-1]
                new_H, new_W = curr_H // 2, curr_W // 2
                spatial_dims.append((new_H, new_W))

                # Reshape back to image for interpolation
                x = rearrange(x, "b h w c -> b c h w", h=curr_H, w=curr_W)
                x = F.interpolate(
                    x, size=(new_H, new_W), mode="bilinear", align_corners=False
                )
                x = rearrange(x, "b c h w -> b h w c")

        for i, stage in enumerate(self.decoder_stages):
            x = stage(x)

            if i < len(self.decoder_stages) - 1:
                # Up-project channel dim
                x = self.up_projs[len(self.up_projs) - 1 - i](x)

                # Interpolate to previous resolution
                curr_H, curr_W = spatial_dims[-(i + 1)]
                new_H, new_W = spatial_dims[-(i + 2)]

                x = rearrange(x, "b h w c -> b c h w", h=curr_H, w=curr_W)
                x = F.interpolate(
                    x, size=(new_H, new_W), mode="bilinear", align_corners=False
                )
                x = rearrange(x, "b c h w -> b h w c")

                # Add skip connection (with potential linear projection)
                skip_idx = len(skip_connections) - 1 - i
                skip = self.skip_projs[skip_idx](skip_connections[skip_idx])
                x = x + skip

        # Reshape back to image format
        x = rearrange(x, "b h w c -> b c h w", h=H, w=W)
        return x

if __name__=="__main__":
    # Instantiate model
    model = SwinUNet2D(
        embed_dims=[64, 128, 256],
        num_heads=[2, 4, 8],
        depths=(2, 2, 2),           # Reduced depth for quick test
        window_size=(8, 8),
    )

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Create dummy input: (batch_size, channels, height, width)
    B, C, H, W = 2, 64, 128, 128  # You can modify sizes for your own tests
    dummy_input = torch.randn(B, C, H, W).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    # Verify output
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    assert output.shape == dummy_input.shape, "Output shape does not match input shape!"

    print("✅ Forward pass successful!")
