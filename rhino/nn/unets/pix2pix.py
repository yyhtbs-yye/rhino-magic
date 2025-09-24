import math
from typing import Iterable, Sequence, Tuple, Optional, List
import torch
from torch import nn
import torch.nn.functional as F


# ---------------------------
# Building blocks
# ---------------------------

def kaiming_init_(m: nn.Module):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class ResBlock(nn.Module):
    """
    Pre-activation residual block with GroupNorm and SiLU (a.k.a. swish).
    Optional in/out channel change and (down/up) sampling controlled externally.
    """
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0, gn_groups: int = 32):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.norm1 = nn.GroupNorm(num_groups=min(gn_groups, in_ch), num_channels=in_ch)
        self.act1 = nn.SiLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=min(gn_groups, out_ch), num_channels=out_ch)
        self.act2 = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    """Lightweight self-attention over spatial dims at a given resolution."""
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.num_heads = num_heads
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)  # (B,C,H,W) each
        # reshape to (B, heads, c_per_head, HW)
        def reshape(t):
            t = t.view(B, self.num_heads, C // self.num_heads, H * W)
            return t
        q, k, v = map(reshape, (q, k, v))
        attn = torch.matmul(q.transpose(-2, -1), k) / math.sqrt(C // self.num_heads)  # (B, heads, HW, HW)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v.transpose(-2, -1)).transpose(-2, -1)  # (B, heads, HW, c_ph)
        out = out.reshape(B, C, H, W)
        return x + self.proj(out)


class Downsample(nn.Module):
    """Anti-aliased downsample via strided conv (3x3, s=2)."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Upsample(nn.Module):
    """Nearest-neighbor upsample + conv (3x3) for stability."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


# ---------------------------
# U-Net Generator
# ---------------------------
class UNet(nn.Module):
    """
    U-Net style generator that maps per-pixel noise X ~ N(0,1) with shape (B,3,H,W)
    to an RGB image (B,3,H,W). Spatial dims are reduced in the encoder (growing channels),
    then restored in the decoder with skip connections.
    It is established for image-to-image GANs (pix2pix), with skip connections helping 
    preserve spatial detail. Your "noise image -> U-Net -> image" adapts that idea by 
    letting the encoder learn what information to keep/discard from spatially-correlated 
    noise before decoding. 

    Args:
        in_ch: input channels (3 for RGB-shaped noise)
        out_ch: output channels (3 for RGB image)
        base_ch: base channel count
        channel_mult: multiplicative factors per resolution level
        num_res_blocks: residual blocks per level (encoder/decoder)
        attn_resolutions: apply self-attention at levels whose min(H,W) equals any listed value
        dropout: dropout inside ResBlocks
        gn_groups: GroupNorm groups
    """
    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 3,
        base_ch: int = 64,
        channel_mult: Sequence[int] = (1, 2, 4, 8),
        output_scale=256,
        num_res_blocks: int = 2,
        attn_resolutions: Iterable[int] = (16,),  # e.g., add attention at 16x16
        dropout: float = 0.0,
        gn_groups: int = 32,
        final_act: str|None = "tanh",  # None, "tanh" or "clamp"
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.base_ch = base_ch
        self.channel_mult = tuple(channel_mult)
        self.output_scale = output_scale
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = set(attn_resolutions)
        self.dropout = dropout
        self.gn_groups = gn_groups
        self.final_act = final_act

        # Input stem
        self.in_proj = nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1)

        # Encoder
        enc_blocks = []
        ch = base_ch
        self.enc_channels: List[int] = [ch]
        self.downs: nn.ModuleList = nn.ModuleList()
        self.enc_towers: nn.ModuleList = nn.ModuleList()
        self.enc_attn: nn.ModuleList = nn.ModuleList()

        for i, mult in enumerate(self.channel_mult):
            out_ch_level = base_ch * mult
            tower = []
            for _ in range(self.num_res_blocks):
                tower.append(ResBlock(ch, out_ch_level, dropout=dropout, gn_groups=gn_groups))
                ch = out_ch_level
                self.enc_channels.append(ch)
            self.enc_towers.append(nn.Sequential(*tower))
            self.enc_attn.append(SelfAttention2d(ch) if i > 0 else nn.Identity())  # optional: start attn after first level
            if i != len(self.channel_mult) - 1:
                self.downs.append(Downsample(ch))
            else:
                self.downs.append(nn.Identity())  # no down at the last level

        # Bottleneck
        self.mid = nn.Sequential(
            ResBlock(ch, ch, dropout=dropout, gn_groups=gn_groups),
            SelfAttention2d(ch),
            ResBlock(ch, ch, dropout=dropout, gn_groups=gn_groups),
        )

        # Decoder
        self.ups: nn.ModuleList = nn.ModuleList()
        self.dec_towers: nn.ModuleList = nn.ModuleList()
        self.dec_attn: nn.ModuleList = nn.ModuleList()

        for i, mult in reversed(list(enumerate(self.channel_mult))):
            out_ch_level = base_ch * mult
            tower = []
            # after concat skip: channels double (ch + skip_ch)
            # implement via first ResBlock taking (ch + skip) → out_ch_level
            for j in range(self.num_res_blocks + 1):  # +1 to handle the concatenation projection
                in_feats = ch + (self._skip_channels_at_level(i) if j == 0 else 0)
                tower.append(ResBlock(in_feats, out_ch_level, dropout=dropout, gn_groups=gn_groups))
                ch = out_ch_level
            self.dec_towers.append(nn.Sequential(*tower))
            self.dec_attn.append(SelfAttention2d(ch) if i > 0 else nn.Identity())
            if i != 0:
                self.ups.append(Upsample(ch))
            else:
                self.ups.append(nn.Identity())

        # Output head
        self.out_norm = nn.GroupNorm(num_groups=min(gn_groups, ch), num_channels=ch)
        self.out_act = nn.SiLU(inplace=True)
        self.out_proj = nn.Conv2d(ch, out_ch, kernel_size=3, padding=1)

        # Init
        self.apply(kaiming_init_)


    def get_noise(self, batch_size, device):

        z = torch.randn((batch_size, self.in_ch, self.output_scale, self.output_scale)).to(device)

        return z

    def _skip_channels_at_level(self, level_idx: int) -> int:
        """Number of channels to be concatenated from encoder at a given level."""
        mult = self.channel_mult[level_idx]
        return self.base_ch * mult

    def _collect_enc_feats(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[Tuple[int, int]]]:
        """
        Run encoder, returning the final feature, a list of skip features, and their shapes.
        """
        skips: List[torch.Tensor] = []
        shapes: List[Tuple[int, int]] = []

        h = self.in_proj(x)
        cur_res = (h.shape[-2], h.shape[-1])
        for i, (tower, attn, down) in enumerate(zip(self.enc_towers, self.enc_attn, self.downs)):
            h = tower(h)
            # optional attention if resolution matches user set; we check by min(H,W)
            if min(cur_res) in self.attn_resolutions:
                h = attn(h)
            skips.append(h)
            shapes.append(cur_res)
            if not isinstance(down, nn.Identity):
                h = down(h)
                cur_res = (h.shape[-2], h.shape[-1])

        return h, skips, shapes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: noise tensor of shape (B, 3, H, W) sampled from N(0,1).
        Output: (B, 3, H, W) in [-1,1] if final_act='tanh', else unclamped logits.
        """
        B, C, H, W = x.shape
        # sanity: H,W must be divisible by 2**(L-1) where L=len(channel_mult)
        min_div = 2 ** (len(self.channel_mult) - 1)
        if (H % min_div) != 0 or (W % min_div) != 0:
            raise ValueError(f"H and W must be divisible by {min_div}; got {(H,W)}")

        # Encoder
        h, skips, shapes = self._collect_enc_feats(x)

        # Mid
        h = self.mid(h)

        # Decoder (mirror levels)
        for i, (tower, attn, up) in enumerate(zip(self.dec_towers, self.dec_attn, self.ups)):
            # concat skip from corresponding encoder level (reverse indexing)
            skip = skips[-(i + 1)]
            # if shapes mismatch due to odd sizes, center-crop skip
            if skip.shape[-2:] != h.shape[-2:]:
                dh = skip.shape[-2] - h.shape[-2]
                dw = skip.shape[-1] - h.shape[-1]
                skip = skip[..., dh // 2: skip.shape[-2] - math.ceil(dh / 2),
                            dw // 2: skip.shape[-1] - math.ceil(dw / 2)]
            h = torch.cat([h, skip], dim=1)
            h = tower(h)
            # attention at this resolution?
            if min(h.shape[-2], h.shape[-1]) in self.attn_resolutions:
                h = attn(h)
            h = up(h)

        # Output head
        h = self.out_proj(self.out_act(self.out_norm(h)))
        if self.final_act == None:
            return h
        elif self.final_act == "tanh":
            return torch.tanh(h)
        elif self.final_act == "clamp":
            return torch.clamp(h, -1, 1)
        else:
            raise ValueError(f"Unknown final_act: {self.final_act}")


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    G = UNet(
        in_ch=3, out_ch=3, base_ch=64, channel_mult=(1, 2, 4, 8),
        num_res_blocks=2, attn_resolutions=(16,), dropout=0.0, gn_groups=32, final_act="tanh"
    )
    # H and W divisible by 2**(len(channel_mult)-1) = 8 here
    x = torch.randn(4, 3, 128, 128)  # noise image
    y = G(x)  # (4, 3, 128, 128), Tanh output in [-1, 1]
    print(y.shape)
