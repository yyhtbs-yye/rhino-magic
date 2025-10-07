import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from rhino.nn.autoregression.pixelcnn.pixelcnn_modules import MaskedConv2d, ResMaskedBlock

class PixelCNN(nn.Module):
    
    def __init__(
        self,
        vocab_size: int,
        in_channels: int,           # data channels C
        hidden_channels: int = 540, # prefer divisible by C (for even typing)
        n_blocks: int = 6,
        kernel_size: int = 3,
        embed_dim: int = 32,
        dropout: float = 0.0,
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
        self.p_drop = float(dropout)

        # ---- Per-value embedding shared across channels ----
        # Applied independently to each (B,C,H,W) integer.
        self.embedding = nn.Embedding(self.V, self.E)

        # Type map for the first layer inputs: [RGB,RGB,RGB,...] (E times), is it E*C in size?
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

        # ---- n_blocks of Mask-B 3x3 ----
        self.blocks = nn.Sequential(*[ResMaskedBlock(self.HC, k=7, n_types=self.C, p=0.0)
                                for _ in range(n_blocks)])


        # Type map for the output layer: [RGB,RGB,RGB,...] (V times), is it V*C in size?
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
        B, C, H, W = x.shape

        eh = self.embedding(rearrange(x, 'b c h w -> b (h w) c'))  # B C H W -> B, H*W, C, E
        z = rearrange(eh, 'b (h w) c e -> b (e c) h w', h=H, w=W)  # B, H*W, C, E -> B, E*C, H, W

        h = self.act_in(self.conv_in(z)) # [RGB]*emb -> R->G->B casual_masking -> [RGB]*hidden

        h = self.blocks(h)

        logits_VCHW = self.head(h)       # [RGB]*emb -> R->G->B casual_masking -> [RGB]*vocab

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

    model = PixelCNN(
        vocab_size=256,
        in_channels=C,
        hidden_channels=120,   # divisible by 3
        n_blocks=4,
        kernel_size=5,
        embed_dim=32,
        dropout=0.0,
    ).to(device)
    model.eval()

    # forward shape check
    x = torch.randint(0, 256, (B, C, H, W), device=device)
    logits = model(x)
    print("logits shape:", tuple(logits.shape))  # (B, V, C, H, W)

    # simple structural causality check (no dependence on future pixels / later channels)
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


# -------------------------
# Tiny demo / causality test
# -------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, C, H, W = 2, 3, 16, 16

    model = PixelCNN(
        vocab_size=256,
        in_channels=C,
        hidden_channels=120,   # divisible by 3
        n_blocks=4,
        kernel_size=5,
        embed_dim=32,
        dropout=0.0,
    ).to(device)
    model.eval()

    # forward shape check
    x = torch.randint(0, 256, (B, C, H, W), device=device)
    logits = model(x)
    print("logits shape:", tuple(logits.shape))  # (B, V, C, H, W)

    # simple structural causality check (no dependence on future pixels / later channels)
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
