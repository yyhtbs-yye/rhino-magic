
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Local imports (assume same folder) — adjust path as needed in your project.
from rhino.nn.autoregression.shared.masked_conv2d import MaskedConv2d
from rhino.nn.autoregression.pixelsnail.pixelsnail_modules import PixelSNAILBlock

class PixelSNAIL(nn.Module):
    """
    PixelSNAIL-style autoregressive image model.

    Inputs:
        vocab_size: number of discrete values per channel (e.g., 256 for 8-bit)
        in_channels: number of data channels (e.g., 3 for RGB)
        hidden_channels: model width
        n_blocks: number of PixelSNAIL residual blocks
        kernel_size: masked conv kernel for the gated branch inside each block
        embed_dim: per-value embedding dimension (shared across channels)
        heads: number of attention heads
        attn_dim: internal attention model dim (defaults to hidden_channels)
        dropout: dropout for residual paths
        attn_dropout: dropout inside attention weights
    """
    def __init__(
        self,
        vocab_size: int,
        in_channels: int,
        hidden_channels: int = 384,
        n_blocks: int = 8,
        kernel_size: int = 3,
        embed_dim: int = 32,
        heads: int = 8,
        attn_dim: int | None = None,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        assert vocab_size >= 2
        assert in_channels >= 1
        assert hidden_channels >= in_channels, "hidden_channels should be >= in_channels"
        # (Optional but recommended) even split of hidden channels across types
        assert hidden_channels % in_channels == 0, \
            "hidden_channels should be divisible by in_channels for even type cycling."

        self.V = vocab_size
        self.C = in_channels
        self.HC = hidden_channels
        self.E = embed_dim

        # ---- Per-value embedding shared across channels ----
        # Applied independently to each (B,C,H,W) integer.
        self.embedding = nn.Embedding(self.V, self.E)

        # Type map for the first layer inputs: [0..C-1] repeated E times
        type_map_in_first = torch.arange(self.C).repeat(self.E)

        # ---- First layer: Mask A (no self-connection at the center) ----
        self.conv_in = MaskedConv2d(
            mask_type='A',
            in_channels=self.C * self.E,
            out_channels=self.HC,
            kernel_size=kernel_size,
            n_types=self.C,
            type_map_in=type_map_in_first
            # type_map_out left as cyclic by C (ok for hidden)
        )
        self.act_in = nn.ReLU(inplace=True)

        # ---- PixelSNAIL residual blocks (attention + gated masked conv) ----
        blocks = []
        for _ in range(n_blocks):
            blocks.append(
                PixelSNAILBlock(
                    channels=self.HC,
                    n_types=self.C,
                    kernel_size=kernel_size,
                    heads=heads,
                    attn_dim=attn_dim if attn_dim is not None else self.HC,
                    attn_dropout=attn_dropout,
                    dropout=dropout,
                )
            )
        self.blocks = nn.Sequential(*blocks)

        # Type map for the output layer: [0..C-1] repeated V times
        type_map_out_head = torch.arange(self.C).repeat(self.V)
        self.head = MaskedConv2d(
            mask_type='B',
            in_channels=self.HC,
            out_channels=self.C * self.V,
            kernel_size=1,
            n_types=self.C,
            type_map_out=type_map_out_head,
        )

        self.reset_parameters()

    # ----- init -----
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d,)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, MaskedConv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    # ----- forward -----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) Long in [0, V-1]
        returns: logits (B, V, C, H, W)
        """
        assert x.dtype == torch.long, "Expect LongTensor with class indices."
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, H, W) for grayscale images

        B, C, H, W = x.shape

        eh = self.embedding(rearrange(x, 'b c h w -> b (h w) c'))  # B C H W -> B, H*W, C, E
        z = rearrange(eh, 'b (h w) c e -> b (e c) h w', h=H, w=W)  # B, H*W, C, E -> B, E*C, H, W

        h = self.act_in(self.conv_in(z))

        h = self.blocks(h)

        logits_VCHW = self.head(h)
        logits = rearrange(logits_VCHW, "b (v c) h w -> b v c h w", c=C, v=self.V)

        return logits

    # ----- sampling -----
    @torch.no_grad()
    def sample(self, x0: torch.Tensor, temperature: float = 1.0, greedy: bool = False) -> torch.Tensor:
        """
        Autoregressive sampling in raster order, channel order 0..C-1.

        Args:
            x0 : (B, C, H, W) Long. Use >=0 to keep a value fixed, <0 to fill.
            temperature : float > 0
            greedy : if True, take argmax; else sample from softmax

        Returns:
            (B, C, H, W) Long in [0, V-1]
        """
        self.eval()
        device = next(self.parameters()).device
        x = x0.to(device)
        if x.dtype != torch.long:
            x = x.long()

        B, C, H, W = x.shape
        V = self.V

        fixed = (x >= 0)
        x = torch.where(fixed, x, torch.zeros_like(x))

        for y in range(H):
            for xcol in range(W):
                for c in range(C):
                    logits_BVCHW = self.forward(x)                # (B, V, C, H, W)
                    logits_c = logits_BVCHW[:, :, c, y, xcol]     # (B, V)

                    if temperature != 1.0:
                        logits_c = logits_c / float(max(1e-8, temperature))

                    if greedy:
                        next_val = logits_c.argmax(dim=-1)        # (B,)
                    else:
                        probs = F.softmax(logits_c, dim=-1)
                        probs = torch.nan_to_num(probs, nan=0.0)
                        next_val = torch.multinomial(probs, num_samples=1).squeeze(1)

                    x[:, c, y, xcol] = next_val

        return x

# -------------------------
# Tiny demo / causality test
# -------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, C, H, W = 2, 3, 16, 16

    model = PixelSNAIL(
        vocab_size=256,
        in_channels=C,
        hidden_channels=192,   # divisible by C
        n_blocks=4,
        kernel_size=5,
        embed_dim=32,
        heads=4,
        attn_dim=192,
        dropout=0.0,
        attn_dropout=0.0,
    ).to(device)
    model.eval()

    # forward shape check
    x = torch.randint(0, 256, (B, C, H, W), device=device)
    logits = model(x)
    print("logits shape:", tuple(logits.shape))  # (B, V, C, H, W)

    # structural causality check (no dependence on future pixels / later channels)
    y0, x0, ch = H // 2, W // 2, 1
    base = logits[0, :, ch, y0, x0].detach()

    x_masked = x.clone()
    # zero rows strictly below y0, and columns strictly to the right on row y0
    x_masked[:, :, y0 + 1:, :] = 0
    x_masked[:, :, y0, x0 + 1:] = 0
    # zero *later channels* at (y0, x0)
    if ch + 1 < C:
        x_masked[:, ch + 1:, y0, x0] = 0

    logits2 = model(x_masked)
    test = logits2[0, :, ch, y0, x0].detach()

    diff = (base - test).abs().max().item()
    print("causal_diff:", diff)  # should be ~0 (exact 0 in float32 with these masks)
