import torch
import torch.nn as nn
import torch.nn.functional as F

from rhino.adversarials.components.pggan import PixelNorm, MinibatchStdDev

class PGGenerator(nn.Module):
    def __init__(self, latent_dim=512, output_channels=1):
        super(PGGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.output_channels = output_channels
        self.current_stage = 0
        self.alpha = 1.0  # Start with fully stabilized network
        
        # Initial block (7x7)
        self.initial = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(latent_dim, 128, 7, 1, 0),  # 7x7 output
            nn.LeakyReLU(0.2),
            PixelNorm(),
            nn.Conv2d(128, 128, 3, 1, 1),  # 7x7 maintained
            nn.LeakyReLU(0.2),
            PixelNorm()
        )
        
        # ToRGB blocks for each resolution
        self.to_rgb_blocks = nn.ModuleList([
            nn.Conv2d(128, output_channels, 1, 1, 0),  # 7x7 -> RGB
            nn.Conv2d(64, output_channels, 1, 1, 0),   # 14x14 -> RGB
            nn.Conv2d(32, output_channels, 1, 1, 0)    # 28x28 -> RGB
        ])
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList([
            # 7x7 -> 14x14
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(128, 64, 3, 1, 1),  # 14x14
                nn.LeakyReLU(0.2),
                PixelNorm(),
                nn.Conv2d(64, 64, 3, 1, 1),   # 14x14
                nn.LeakyReLU(0.2),
                PixelNorm()
            ),
            # 14x14 -> 28x28
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(64, 32, 3, 1, 1),   # 28x28
                nn.LeakyReLU(0.2),
                PixelNorm(),
                nn.Conv2d(32, 32, 3, 1, 1),   # 28x28
                nn.LeakyReLU(0.2),
                PixelNorm()
            )
        ])
    
    def forward(self, z):
        # Reshape latent vector to 2D
        x = z.view(z.size(0), self.latent_dim, 1, 1)
        
        # Initial block
        x = self.initial(x)
        
        # Return image based on current stage and alpha
        if self.current_stage == 0:
            # Stage 0: 7x7 resolution
            return self.to_rgb_blocks[0](x)
        else:
            # Process through upsampling blocks up to current stage
            for i in range(self.current_stage):
                features_prev = x
                x = self.up_blocks[i](x)
                
                # During transition phase for the current stage
                if i == self.current_stage - 1 and self.alpha < 1:
                    # Blend with upsampled lower resolution
                    y = F.interpolate(features_prev, scale_factor=2, mode='nearest')
                    y = self.to_rgb_blocks[i](y)
                    out = self.to_rgb_blocks[i+1](x)
                    return (1 - self.alpha) * y + self.alpha * out
            
            # After transition: use higher resolution directly
            return self.to_rgb_blocks[self.current_stage](x)
        
    def progress(self):
        """Progress to the next stage if not at final stage"""
        if self.current_stage < len(self.up_blocks):
            self.current_stage += 1
            self.alpha = 0.0  # Start transitioning from the previous stage
            
    def update_alpha(self, increment=0.1):
        """Update the alpha value for smooth transition"""
        self.alpha = min(1.0, self.alpha + increment)

# Progressive Discriminator
class PGDiscriminator(nn.Module):
    def __init__(self, input_channels=1):
        super(PGDiscriminator, self).__init__()
        self.input_channels = input_channels
        self.current_stage = 0
        self.alpha = 1.0  # Start with fully stabilized network
        
        # FromRGB blocks for each resolution
        self.from_rgb_blocks = nn.ModuleList([
            nn.Conv2d(input_channels, 128, 1, 1, 0),  # RGB -> 7x7 features
            nn.Conv2d(input_channels, 64, 1, 1, 0),   # RGB -> 14x14 features
            nn.Conv2d(input_channels, 32, 1, 1, 0)    # RGB -> 28x28 features
        ])
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList([
            # 14x14 -> 7x7
            nn.Sequential(
                nn.Conv2d(64, 64, 3, 1, 1),   # 14x14
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 128, 3, 1, 1),  # 14x14
                nn.LeakyReLU(0.2),
                nn.AvgPool2d(2)  # 7x7
            ),
            # 28x28 -> 14x14
            nn.Sequential(
                nn.Conv2d(32, 32, 3, 1, 1),   # 28x28
                nn.LeakyReLU(0.2),
                nn.Conv2d(32, 64, 3, 1, 1),   # 28x28
                nn.LeakyReLU(0.2),
                nn.AvgPool2d(2)  # 14x14
            ),
        ])
        
        # Final block (7x7 -> decision)
        self.final = nn.Sequential(
            MinibatchStdDev(),
            nn.Conv2d(128 + 1, 128, 3, 1, 1),  # +1 for minibatch std
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Process based on current stage
        if self.current_stage == 0:
            # Stage 0: 7x7 resolution
            features = self.from_rgb_blocks[0](x)
            return self.final(features)
        else:
            # Start from the highest resolution
            current_res_idx = self.current_stage
            features = self.from_rgb_blocks[current_res_idx](x)
            
            # Process through downsampling blocks from highest to lowest
            for i in range(current_res_idx-1, -1, -1):
                if i == current_res_idx-1 and self.alpha < 1:
                    # During transition: blend with downsampled input
                    x_down = F.avg_pool2d(x, 2)
                    y = self.from_rgb_blocks[current_res_idx-1](x_down)
                    features = self.down_blocks[current_res_idx-1](features)
                    features = (1 - self.alpha) * y + self.alpha * features
                else:
                    # After transition: downsample features directly
                    features = self.down_blocks[i](features)
            
            # Final decision
            return self.final(features)
    
    def progress(self):
        """Progress to the next stage if not at final stage"""
        if self.current_stage < len(self.down_blocks):
            self.current_stage += 1
            self.alpha = 0.0  # Start transitioning
            
    def update_alpha(self, increment=0.1):
        """Update the alpha value for smooth transition"""
        self.alpha = min(1.0, self.alpha + increment)
