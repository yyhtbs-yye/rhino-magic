import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from rhino.regressors.components.pixelcnn import DiagonalBiLSTM, PixelCNNBlock, RowLSTM, MaskedConv2d

class ConfigurablePixelCNN(ModelMixin, ConfigMixin):
    
    @register_to_config
    def __init__(
        self,
        sample_size=None,
        in_channels=3,
        out_channels=3,
        block_types=("PixelCNNBlock", "PixelCNNBlock", "PixelCNNBlock"),
        block_out_channels=[64, 128, 256],
        layers_per_block=1,
        norm_num_groups=32,
        norm_eps=1e-5,
        num_classes=256,
        kernel_size=7
    ):
        super().__init__()

        self.sample_size = sample_size
        
        # Input convolution
        self.input_conv = MaskedConv2d(
            'A',
            in_channels,
            block_out_channels[0],
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        
        # Blocks
        self.blocks = nn.ModuleList()
        current_in_channels = block_out_channels[0]
        
        for i, (block_type, out_channels) in enumerate(zip(block_types, block_out_channels)):
            for _ in range(layers_per_block):
                if block_type == "PixelCNNBlock":
                    block = PixelCNNBlock(
                        current_in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2
                    )
                elif block_type == "DiagonalBiLSTM":
                    block = DiagonalBiLSTM(
                        current_in_channels,
                        out_channels,
                        num_layers=1
                    )
                elif block_type == "RowLSTM":
                    block = RowLSTM(
                        current_in_channels,
                        out_channels,
                        num_layers=1
                    )
                else:
                    raise ValueError(f"Unknown block type: {block_type}")
                
                self.blocks.append(block)
                current_in_channels = out_channels
        
        # Output layers
        self.output_conv1 = nn.Conv2d(
            block_out_channels[-1],
            block_out_channels[-1],
            kernel_size=1
        )
        self.output_conv2 = nn.Conv2d(
            block_out_channels[-1],
            out_channels * num_classes,
            kernel_size=1
        )
        
        # Normalization
        self.norm = nn.GroupNorm(
            norm_num_groups,
            block_out_channels[0],
            eps=norm_eps
        )
        
    def forward(self, x, states=None):
        batch_size, channels, height, width = x.size()
        
        # Initial convolution and normalization
        x = self.input_conv(x)
        x = self.norm(x)
        x = F.relu(x)
        
        # Process blocks
        if states is None:
            states = [None] * len(self.blocks)
        
        new_states = []
        for block, state in zip(self.blocks, states):
            if isinstance(block, (DiagonalBiLSTM, RowLSTM)):
                x, block_states = block(x, state)
                new_states.append(block_states)
            else:
                x = block(x)
                new_states.append(None)
        
        # Output layers
        x = F.relu(self.output_conv1(x))
        x = self.output_conv2(x)
        
        # Reshape for softmax
        x = x.view(batch_size, channels, self.config.num_classes, height, width)

        x = rearrange(x, 'b c n h w -> b c h w n')
        
        return {'sample': x, 'new_states': new_states}
    
    @classmethod
    def from_config(cls, config):
        """
        Instantiate a ConfigurablePixelCNN model from a configuration dictionary.
        
        Args:
            config (dict): Configuration dictionary with model parameters.
        
        Returns:
            ConfigurablePixelCNN: Instantiated model.
        """
        return cls(**config)