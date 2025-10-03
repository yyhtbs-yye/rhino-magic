import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from rhino.nn.autoregression.pixelcnn.pixelcnn_modules import MaskedConv2d, PixelCNNResidualBlock

# -------------------------------
# PixelCNN (multi-channel, channel-AR via W -> W*C remap)
# -------------------------------
class PixelCNN(nn.Module):
    """
    Multi-channel PixelCNN for discrete token grids (e.g., VQ indices).

    Inputs:
        x: LongTensor of shape [B, C, H, W], values in [0, vocab_size-1]
    Forward:
        returns logits of shape [B, vocab_size, C, H, W]
    Sampling:
        sample(x0, steps=H*W*C, temperature=1.0, greedy=False)
        x0 should be zeros-like long tensor of shape [B, C, H, W]
    """
    def __init__(
        self,
        vocab_size: int,
        in_channels: int,           # data channels C
        hidden_channels: int = 128,
        n_layers: int = 8,
        kernel_size: int = 3,
        embed_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.data_channels = int(in_channels)
        self.embed_dim = int(embed_dim)

        # Shared token embedding (per "subpixel" position)
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)

        # First masked conv (type 'A') after embedding projection
        self.in_proj = MaskedConv2d("A", self.embed_dim, hidden_channels, kernel_size=kernel_size)

        # Residual stack with 'B' masks
        blocks = []
        for _ in range(n_layers):
            blocks.append(PixelCNNResidualBlock(hidden_channels, kernel_size=kernel_size, dropout=dropout))
        self.blocks = nn.Sequential(*blocks)

        # Output head to logits (1x1 conv over features)
        self.out_head = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=hidden_channels),
            nn.ReLU(inplace=True),
            MaskedConv2d("B", hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, self.vocab_size, kernel_size=1),
        )

        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming init for convs
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, MaskedConv2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    # ------- helpers: reshape to 2D grid with channel-AR along width -------
    def _pack(self, x_long: torch.Tensor):
        """
        [B, C, H, W] --> tokens [B, H, W*C]
        Then embed -> [B, E, H, W*C]
        """
        assert x_long.dim() == 4, "expected [B, C, H, W]"
        b, c, h, w = x_long.shape
        x_long = rearrange(x_long, "b c h w -> b h (w c)")          # [B, H, W*C]
        x_emb = self.embedding(x_long.clamp(min=0).long())          # [B, H, W*C, E]
        x_emb = x_emb.permute(0, 3, 1, 2).contiguous()              # [B, E, H, W*C]
        return x_emb, (b, c, h, w)

    def _unpack_logits(self, logits_2d: torch.Tensor, shape_tuple):
        """
        [B, V, H, W*C] --> [B, V, C, H, W]
        """
        b, c, h, w = shape_tuple
        logits = rearrange(logits_2d, "b v h wc -> b v h wc")
        logits = rearrange(logits, "b v h (w c) -> b v c h w", c=c, w=w)
        return logits

    # ------------------------------- forward -------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Teacher-forced forward pass.
        x: LongTensor [B, C, H, W]
        returns logits: [B, vocab_size, C, H, W]
        """
        x = x.long()
        x_emb, shape_tuple = self._pack(x)                   # [B, E, H, W*C]
        h = self.in_proj(x_emb)                              # 'A' mask
        h = self.blocks(h)                                   # residual stack (all 'B' masks)
        logits_2d = self.out_head(h)                         # [B, V, H, W*C]
        logits = self._unpack_logits(logits_2d, shape_tuple) # [B, V, C, H, W]
        return logits

    # ------------------------------- sampling -------------------------------
    @torch.no_grad()
    def sample(
        self,
        x0: torch.Tensor,
        steps: int = None,
        temperature: float = 1.0,
        greedy: bool = False,
    ) -> torch.Tensor:
        """
        Autoregressive sampling in raster order with per-pixel channel AR.
        x0: [B, C, H, W] LongTensor (usually zeros_like of target shape)
        steps: optional; defaults to H*W*C
        """
        assert x0.dim() == 4, "sample expects [B, C, H, W]"
        b, c, h, w = x0.shape
        total = h * w * c
        if steps is None:
            steps = total
        steps = int(min(steps, total))

        x = x0.long().clone()

        # Iterate raster order: y -> x -> channel
        filled = 0
        for y in range(h):
            for xw in range(w):
                for ch in range(c):
                    if filled >= steps:
                        break
                    logits = self.forward(x)  # [B, V, C, H, W]
                    logits_ = logits[:, :, ch, y, xw]  # [B, V]
                    if temperature != 1.0:
                        logits_ = logits_ / max(1e-8, float(temperature))
                    probs = F.softmax(logits_, dim=-1)  # [B, V]
                    if greedy:
                        next_token = probs.argmax(dim=-1)
                    else:
                        next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
                    x[:, ch, y, xw] = next_token
                    filled += 1
                if filled >= steps:
                    break
            if filled >= steps:
                break
        return x
