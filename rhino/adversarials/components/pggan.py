import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(PixelNorm, self).__init__()
        self.epsilon = epsilon
        
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)

class MinibatchStdDev(nn.Module):
    def __init__(self):
        super(MinibatchStdDev, self).__init__()
    
    def forward(self, x):
        batch_size, _, height, width = x.shape
        # Calculate standard deviation across batch
        y = x - x.mean(dim=0, keepdim=True)
        y = torch.sqrt(y.pow(2).mean(dim=0, keepdim=False) + 1e-8)
        y = y.mean().view(1, 1, 1, 1)
        y = y.repeat(batch_size, 1, height, width)
        # Append as new channel
        return torch.cat([x, y], dim=1)

class ResnetBlock2D(nn.Module):
    """
    Simple ConvBlock with activation and PixelNorm, named like a diffusers ResnetBlock2D.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 activation=nn.LeakyReLU(0.2), use_pixelnorm=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            activation
        ]
        if use_pixelnorm:
            layers.append(PixelNorm())
        self.block = nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        return self.block(x)

class Downsample2D(nn.Module):
    """
    Downsample block named like a diffusers Downsample2D.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            ResnetBlock2D(in_channels, in_channels, 3, 1, 1, use_pixelnorm=False),
            ResnetBlock2D(in_channels, out_channels, 3, 1, 1, use_pixelnorm=False),
            nn.AvgPool2d(2)
        )

    def forward(self, x, **kwargs):
        return self.block(x)

class Upsample2D(nn.Module):
    """
    Upsample block named like a diffusers Upsample2D.
    """
    def __init__(self, in_channels, out_channels, use_pixelnorm=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            ResnetBlock2D(in_channels, out_channels, 3, 1, 1, use_pixelnorm=use_pixelnorm),
            ResnetBlock2D(out_channels, out_channels, 3, 1, 1, use_pixelnorm=use_pixelnorm)
        )

    def forward(self, x, **kwargs):
        return self.block(x)

class ToRGB2D(nn.Module):
    """
    1x1 conv to RGB, named ToRGB2D like diffusers conv_out.
    """
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x, **kwargs):
        return self.conv(x)

class FromRGB2D(nn.Module):
    """
    1x1 conv from RGB, named FromRGB2D like diffusers conv_in.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x, **kwargs):
        return self.conv(x)

class DiscriminatorFinalBlock2D(nn.Module):
    """
    Final block for discriminator, named similar to diffusers output block.
    """
    def __init__(self, in_channels, feature_maps=128, image_res=7):
        super().__init__()
        self.block = nn.Sequential(
            MinibatchStdDev(),
            nn.Conv2d(in_channels+1, feature_maps, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(feature_maps * image_res * image_res, feature_maps),
            nn.LeakyReLU(0.2),
            nn.Linear(feature_maps, 1),
            nn.Sigmoid()
        )

    def forward(self, x, **kwargs):
        return self.block(x)

