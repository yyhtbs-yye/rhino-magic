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
        return self.autoencoder.encode(x).latent_dist.mean

    def decode(self, z):
        return self.autoencoder.decode(z).sample
    
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
        latent_output = self.autoencoder.encode(x).latent
        return latent_output

    def decode(self, z):
        decoded_output = self.autoencoder.decode(z).sample
        return decoded_output
    

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path):
        autoencoder = AutoencoderDC.from_pretrained(pretrained_model_name_or_path)
        
        return AutoencoderDCWrapper(autoencoder)

    @staticmethod
    def from_config(config):
        autoencoder = AutoencoderDC.from_config(config)
        
        return AutoencoderDCWrapper(autoencoder)