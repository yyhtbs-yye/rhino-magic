from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.autoencoders import AutoencoderDC
import torch.nn as nn

class AutoencoderKLWrapper(nn.Module):

    def __init__(self, autoencoder):
        super().__init__()  # This is essential for nn.Module subclasses

        self.autoencoder = autoencoder
        self.config = self.autoencoder.config

        self.eval()

    def encode(self, x):

        z = self.autoencoder.encode(x).latent_dist.mean
        z = z * self.config.scaling_factor
        return z

    def decode(self, z):
        z = 1 / self.config.scaling_factor * z
        x = self.autoencoder.decode(z).sample
        return x
    
    @staticmethod
    def from_pretrained(pretrained_model_name_or_path):
        autoencoder = AutoencoderKL.from_pretrained(pretrained_model_name_or_path)
        return AutoencoderKLWrapper(autoencoder)

    @staticmethod
    def from_config(config):
        autoencoder = AutoencoderKL.from_config(config)
        return AutoencoderKLWrapper(autoencoder)

class AutoencoderDCWrapper(nn.Module):

    def __init__(self, autoencoder):
        super().__init__()  # This is essential for nn.Module subclasses

        self.autoencoder = autoencoder
        self.config = self.autoencoder.config

    def encode(self, x):
        z = self.autoencoder.encode(x).latent
        z = z * self.config.scaling_factor
        return z

    def decode(self, z):
        z = 1 / self.config.scaling_factor * z
        x = self.autoencoder.decode(z).sample
        return x

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path):
        autoencoder = AutoencoderDC.from_pretrained(pretrained_model_name_or_path)
        
        return AutoencoderDCWrapper(autoencoder)

    @staticmethod
    def from_config(config):
        autoencoder = AutoencoderDC.from_config(config)
        
        return AutoencoderDCWrapper(autoencoder)