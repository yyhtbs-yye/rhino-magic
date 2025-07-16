import torch
import torch.nn as nn
from rhino.nn.blocks.rtsb import RTSB
from rhino.nn.backbones.template_backbone import TemplateBackbone

class LinearConcatRTSB(TemplateBackbone):
    def __init__(self,
                 nn_class,
                 n_stages=4,
                 **args):
        super().__init__(**args)
        
        self.stages = nn.ModuleList([
            RTSB(
                block_cls   = nn_class,
                **args
            ) for n_stage in range(n_stages)
        ])

        self.apply(self._init_weights)

if __name__=="__main__":
    
    from rhino.nn.blocks.vision_transformer_block_2d import ViTBlock2D

    # Move model to device
    device = torch.device("cuda:11" if torch.cuda.is_available() else "cpu")

    # Create dummy input: (batch_size, channels, height, width)
    B, C, H, W = 4, 64, 256, 256  # You can modify sizes for your own tests
    dummy_input = torch.randn(B, C, H, W).to(device)
    ts   = torch.randint(0, 1000, (B,), device=device)

    model = LinearConcatRTSB(
        nn_class        = ViTBlock2D,
        n_stages        = 4,
        depth           = 2,           # Reduced depth for quick test
        embed_dim       = 64,
        num_heads       = 4,
        window_size     = (H, W),
        patch_size      = (16, 16),
        mlp_ratio       = 4.0,
        qkv_bias        = True,
        triggers        = None,  # Example triggers
        enable_condition= True,
        cond_dim        = 256,  # Example condition dimension
    )

    model = model.to(device)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input, ts)

    # Verify output
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    assert output.shape == dummy_input.shape, "Output shape does not match input shape!"

    print("✅ Forward pass successful!")
