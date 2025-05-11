import torch
import torch.nn as nn
import torch.nn.functional as F

# Noise implementations
class CleanNoise:

    @classmethod
    def from_config(cls, config) -> 'CleanNoise':
        return cls()

    def __call__(self, data):
        return data

class GaussianNoise:
    def __init__(self, sigma = 0.1):
        self.sigma = sigma

    @classmethod
    def from_config(cls, config) -> 'GaussianNoise':
        params = config.get('params', {})
        return cls(sigma=params.get('sigma', 0.1)
        )

    def __call__(self, data):
        return data + torch.randn_like(data) * self.sigma

class PoissonNoise:
    def __init__(self, rate = 1.0):
        self.rate = rate

    @classmethod
    def from_config(cls, config) -> 'PoissonNoise':
        params = config.get('params', {})
        return cls(rate=params.get('rate', 1.0)
        )

    def __call__(self, data):
        data = (data + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        data = data.clamp(0, 1)
        data = torch.poisson(data * 255.0 * self.rate) / (255.0 * self.rate)
        return (data * 2.0 - 1.0).clamp(-1, 1)