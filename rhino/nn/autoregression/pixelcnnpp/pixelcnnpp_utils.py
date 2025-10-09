import torch
import torch.nn.functional as F

def get_n_out_channels_for_dmol(in_channels: int, nr_mix: int, *, coupled: bool = False) -> int:
    """
    Number of output channels for a DMOL head with arbitrary input channels.
    - Always predict: K logits, C*K means, C*K log_scales.
    - If `coupled=True`, also predict triangular channel-coupling coeffs:
      n_coeff = C*(C-1)/2 per mixture (PixelCNN++ style).
    """
    if in_channels < 1:
        raise ValueError("in_channels must be >= 1")
    n_coeff = (in_channels * (in_channels - 1)) // 2 if coupled else 0
    return nr_mix * (1 + 2 * in_channels + n_coeff)
