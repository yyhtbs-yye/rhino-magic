from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- shifts ----------
def down_shift(x: torch.Tensor) -> torch.Tensor:
    # pad top with zeros, drop last row
    return F.pad(x, (0, 0, 1, 0))[:, :, :-1, :]


def right_shift(x: torch.Tensor) -> torch.Tensor:
    # pad left with zeros, drop last col
    return F.pad(x, (1, 0, 0, 0))[:, :, :, :-1]

class ShiftedConv2d(nn.Module):
    """
    Shifted conv that (1) shifts the input, then (2) applies a masked kernel so
    no current/future pixels are ever read in original coordinates.
    Supported shifts: 'down', 'downright'.
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, shift: str):
        super().__init__()
        assert shift in {"down", "downright"}
        self.shift = shift
        self.k = int(kernel_size)
        self.conv = nn.Conv2d(in_ch, out_ch, self.k, padding=self.k // 2, bias=True)

        # Build a (1,1,k,k) mask; broadcast to (out_ch,in_ch,k,k) on use
        r = self.k // 2
        mask = torch.ones(1, 1, self.k, self.k)

        if shift == "down":
            # allow dy <= 0 in shifted space -> zero rows strictly below center
            if r + 1 < self.k:
                mask[..., r + 1 :, :] = 0.0
        else:  # "downright"
            # allow: dy <= 0  OR (dy == +1 and dx <= 0)
            # zero rows with dy >= +2
            if r + 2 < self.k:
                mask[..., r + 2 :, :] = 0.0
            # on the dy == +1 row, zero dx > 0 (i.e., columns to the right of center)
            if r + 1 < self.k and r + 1 < self.k:
                mask[..., r + 1, r + 1 :] = 0.0

        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) shift the input
        if self.shift == "down":
            x = F.pad(x, (0, 0, 1, 0))[:, :, :-1, :]
        else:  # "downright"
            x = F.pad(x, (1, 0, 0, 0))[:, :, :, :-1]      # right shift
            x = F.pad(x, (0, 0, 1, 0))[:, :, :-1, :]      # then down shift

        # 2) apply masked conv weights (broadcast mask over in/out channels)
        w = self.conv.weight * self.mask.to(self.conv.weight.dtype)
        return F.conv2d(
            x, w, self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )


class Nin(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.lin = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


class GatedResidualBlock(nn.Module):
    """
    Gated ResNet block with shifted convolutions and optional 'aux' input
    (typically the v-stack feature for the h-stack).
    """
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        shift: str,
        dropout: float = 0.0,
        aux_channels: int = 0,
    ):
        super().__init__()
        self.act = nn.ELU(inplace=False)
        self.conv1 = ShiftedConv2d(channels, channels, kernel_size, shift)
        self.dropout = nn.Dropout2d(dropout) if dropout and dropout > 0 else nn.Identity()
        self.pre_gate = ShiftedConv2d(channels, 2 * channels, 1, shift)
        self.aux_to_gate = Nin(aux_channels, 2 * channels) if aux_channels > 0 else None
        self.out = Nin(channels, channels)

        nn.init.zeros_(self.out.lin.weight)
        nn.init.zeros_(self.out.lin.bias)

    def forward(self, x: torch.Tensor, aux: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.act(x)
        h = self.conv1(h)
        h = self.dropout(h)

        gate = self.pre_gate(self.act(h))
        if aux is not None and self.aux_to_gate is not None:
            gate = gate + self.aux_to_gate(aux)

        a, b = torch.chunk(gate, chunks=2, dim=1)
        h = torch.tanh(a) * torch.sigmoid(b)
        h = self.out(h)
        return x + h
