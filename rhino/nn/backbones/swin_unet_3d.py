import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_
from rhino.nn.blocks.rtsb import RTSB
from rhino.nn.blocks.swin_transformer_block_3d import SwinTransformerBlock3D
from einops import rearrange

class SwinUNet3D(nn.Module):
    """
    U-Net Style Swin Vision Transformer backbone using RTSB stages on 3D volume data.
    Input:  (B, embed_dim, T, H, W)
    Output: (B, embed_dim, T, H, W)
    """
    def __init__(self,
                 embed_dims=[64, 128, 256],
                 num_heads=[2, 4, 8],
                 depths=(4, 4, 4),
                 window_size=(4, 8, 8),
                 mlp_ratio=4.,
                 qkv_bias=True):
        super().__init__()
        self.pos_drop = nn.Dropout(p=0.)
        
        # Define embedding dimensions for encoder/decoder
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        
        # Encoder stages (downsampling)
        self.encoder_stages = nn.ModuleList([
            RTSB(
                block_cls   = SwinTransformerBlock3D,
                embed_dim   = self.embed_dims[i],
                depth       = depths[i],
                num_heads   = self.num_heads[i],
                window_size = window_size,
                mlp_ratio   = mlp_ratio,
                qkv_bias    = qkv_bias
            ) for i in range(len(depths))
        ])
        
        # Decoder stages (upsampling)
        self.decoder_stages = nn.ModuleList([
            RTSB(
                block_cls   = SwinTransformerBlock3D,
                embed_dim   = self.embed_dims[len(depths)-1-i],
                depth       = depths[len(depths)-1-i],
                num_heads   = self.num_heads[len(depths)-1-i],
                window_size = window_size,
                mlp_ratio   = mlp_ratio,
                qkv_bias    = qkv_bias
            ) for i in range(len(depths))
        ])
        
        # Projection layers for dimension changes
        self.down_projs = nn.ModuleList([
            nn.Linear(self.embed_dims[i], self.embed_dims[i+1]) 
            for i in range(len(self.embed_dims)-1)
        ])
        
        self.up_projs = nn.ModuleList([
            nn.Linear(self.embed_dims[i+1], self.embed_dims[i]) 
            for i in range(len(self.embed_dims)-1)
        ])
        
        # Skip connection projection layers
        self.skip_projs = nn.ModuleList([
            nn.Linear(self.embed_dims[i], self.embed_dims[i]) 
            for i in range(len(self.embed_dims)-1)
        ])
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x: (B, embed_dim, T, H, W)
        B, C, T, H, W = x.shape
        
        # Convert to sequence format
        x = rearrange(x, 'b c t h w -> b t h w c')
        x = self.pos_drop(x)
        
        # Store skip connections and spatial dimensions
        skip_connections = []
        spatial_dims = [(T, H, W)]
        
        # Encoder path
        for i, stage in enumerate(self.encoder_stages):
            # Process through stage
            x = stage(x)
            
            # Store skip connection (except for the last stage)
            if i < len(self.encoder_stages) - 1:
                skip_connections.append(x)
                
                # Project to next dimension
                x = self.down_projs[i](x)
                
                # Downsample spatially using interpolation
                curr_T, curr_H, curr_W = spatial_dims[-1]
                new_T, new_H, new_W = curr_T, curr_H // 2, curr_W // 2
                spatial_dims.append((new_T, new_H, new_W))
                
                # Reshape for interpolation
                x = rearrange(x, 'b t h w c -> b c t h w', t=curr_T, h=curr_H, w=curr_W)
                x = F.interpolate(x, size=(new_T, new_H, new_W), mode='trilinear', align_corners=False)
                x = rearrange(x, 'b c t h w -> b t h w c')
        
        # Decoder path
        for i, stage in enumerate(self.decoder_stages):
            # Process through stage
            x = stage(x)
            
            if i < len(self.decoder_stages) - 1:
                # Project to previous dimension
                x = self.up_projs[len(self.up_projs)-1-i](x)
                
                # Upsample spatially using interpolation
                curr_T, curr_H, curr_W = spatial_dims[-(i+1)]
                new_T, new_H, new_W = spatial_dims[-(i+2)]
                
                # Reshape for interpolation
                x = rearrange(x, 'b t h w c -> b c t h w', t=curr_T, h=curr_H, w=curr_W)
                x = F.interpolate(x, size=(new_T, new_H, new_W), mode='trilinear', align_corners=False)
                x = rearrange(x, 'b c t h w -> b t h w c')
                
                # Add skip connection
                skip_idx = len(skip_connections) - 1 - i
                skip = self.skip_projs[skip_idx](skip_connections[skip_idx])
                x = x + skip
        
        # Back to original spatial format
        x = rearrange(x, 'b t h w c -> b c t h w', t=T, h=H, w=W)
        return x
    

if __name__=="__main__":
    # Instantiate model
    model = SwinUNet3D(
        embed_dims=[64, 128, 256],
        num_heads=[2, 4, 8],
        depths=(2, 2, 2),           # Reduced depth for quick test
        window_size=(4, 8, 8),
    )

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Create dummy input: (batch_size, channels, height, width)
    B, C, T, H, W = 2, 64, 16, 128, 128  # You can modify sizes for your own tests
    dummy_input = torch.randn(B, C, T, H, W).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    # Verify output
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    assert output.shape == dummy_input.shape, "Output shape does not match input shape!"

    print("✅ Forward pass successful!")
