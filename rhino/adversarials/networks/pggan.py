import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import ModelMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config
from rhino.adversarials.components.pggan import PixelNorm, ToRGB2D, FromRGB2D, Upsample2D, Downsample2D, DiscriminatorFinalBlock2D

class ProgressiveGrowGenerator(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        in_channels: int = 512,
        out_channels: int = 1,
        sample_size: int = 28,
        block_types: tuple = ("Upsample2D", "Upsample2D"),
        block_out_channels: tuple = (128, 64, 32),
        layers_per_block: int = 1,
        current_stage: int = 0,
        alpha: float = 1.0
    ):
        super().__init__()

        # Initial block: Expand latent to 7x7
        self.initial_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, block_out_channels[0], 7, 1, 0),
            PixelNorm(),
            nn.LeakyReLU(0.2)
        )

        # Upsampling blocks
        self.up_blocks = nn.ModuleList([])
        for i, (block_type, in_ch) in enumerate(zip(block_types, block_out_channels[:-1])):
            out_ch = block_out_channels[i + 1]
            self.up_blocks.append(Upsample2D(in_ch, out_ch))

        # ToRGB blocks for each stage
        self.to_rgb = nn.ModuleList([])
        for ch in block_out_channels:
            self.to_rgb.append(ToRGB2D(ch, out_channels))

    def forward(self, z, return_dict: bool = True):
        x = z.view(z.size(0), self.config.in_channels, 1, 1)
        x = self.initial_block(x)

        if self.config.current_stage == 0:
            out = self.to_rgb[0](x)
            return {"sample": out} if return_dict else out

        for i in range(self.config.current_stage):
            prev = x
            x = self.up_blocks[i](x)
            if i == self.config.current_stage - 1 and self.config.alpha < 1.0:
                y = F.interpolate(prev, scale_factor=2, mode='nearest')
                y = self.to_rgb[i](y)
                out = self.to_rgb[i + 1](x)
                out = (1 - self.config.alpha) * y + self.config.alpha * out
                return {"sample": out} if return_dict else out

        out = self.to_rgb[self.config.current_stage](x)
        return {"sample": out} if return_dict else out

    def progress(self):
        if self.config.current_stage < len(self.up_blocks):
            self.config.current_stage += 1
            self.config.alpha = 0.0

    def update_alpha(self, increment: float = 0.1):
        self.config.alpha = min(1.0, self.config.alpha + increment)

class ProgressiveGrowDiscriminator(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        sample_size: int = 28,
        block_types: tuple = ("DownBlock2D", "DownBlock2D"),
        block_out_channels: tuple = (32, 64, 128),
        layers_per_block: int = 1,
        current_stage: int = 0,
        alpha: float = 1.0
    ):
        super().__init__()

        # FromRGB blocks
        self.from_rgb = nn.ModuleList([])
        for ch in block_out_channels:
            self.from_rgb.append(FromRGB2D(in_channels, ch))

        # Downsampling blocks
        self.down_blocks = nn.ModuleList([])
        for i, (block_type, in_ch) in enumerate(zip(block_types, block_out_channels[1:])):
            out_ch = block_out_channels[i]
            self.down_blocks.append(Downsample2D(out_ch, in_ch))

        # Final block
        self.final_block = DiscriminatorFinalBlock2D(block_out_channels[-1])

    def forward(self, x, return_dict: bool = True):
        if self.config.current_stage == 0:
            feat = self.from_rgb[0](x)
            out = self.final_block(feat)
            return {"sample": out} if return_dict else out

        idx = self.config.current_stage
        feat = self.from_rgb[idx](x)
        for i in range(idx - 1, -1, -1):
            if i == idx - 1 and self.config.alpha < 1.0:
                down = F.avg_pool2d(x, 2)
                y = self.from_rgb[i](down)
                feat = self.down_blocks[i](feat)
                feat = (1 - self.config.alpha) * y + self.config.alpha * feat
            else:
                feat = self.down_blocks[i](feat)

        out = self.final_block(feat)
        return {"sample": out} if return_dict else out

    def progress(self):
        if self.config.current_stage < len(self.down_blocks):
            self.config.current_stage += 1
            self.config.alpha = 0.0

    def update_alpha(self, increment: float = 0.1):
        self.config.alpha = min(1.0, self.config.alpha + increment)

if __name__ == "__main__":
    # Example usage
    generator_config = {
        "sample_size": 28,
        "in_channels": 512,
        "out_channels": 1,
        "block_types": ["UpBlock2D", "UpBlock2D"],
        "block_out_channels": [128, 64, 32],
        "layers_per_block": 1,
        "current_stage": 0,
        "alpha": 1.0
    }

    discriminator_config = {
        "sample_size": 28,
        "in_channels": 1,
        "out_channels": 1,
        "block_types": ["DownBlock2D", "DownBlock2D"],
        "block_types": [],
        "block_out_channels": [32, 64, 128],
        "layers_per_block": 1,
        "current_stage": 0,
        "alpha": 1.0
    }

    # Initialize models
    generator = ProgressiveGrowGenerator.from_config(**generator_config)
    discriminator = ProgressiveGrowDiscriminator.from_config(**discriminator_config)

    # Test forward pass
    z = torch.randn(1, 512)
    gen_output = generator(z)
    print("Generator output shape:", gen_output["sample"].shape)

    x = torch.randn(1, 1, 28, 28)
    disc_output = discriminator(x)
    print("Discriminator output shape:", disc_output["sample"].shape)