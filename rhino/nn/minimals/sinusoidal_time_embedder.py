import math
import torch
import torch.nn as nn

class SinusoidalTimeEmbedder(nn.Module):
    """
    Deterministic 1-D sinusoidal embedding for diffusion timesteps.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        t : (B,) int or float tensor of diffusion timesteps 0 … T
        Returns
        -------
        emb : (B, embed_dim) float tensor
        """
        half = self.embed_dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32)
            * -(math.log(10000.0) / (half - 1))
        )
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.embed_dim % 2 == 1:  # zero‑pad if embed_dim is odd
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb
