import torch
import torch.nn as nn
from typing import Dict, Any, Iterable, Tuple, List, Optional
from rhino.nn.losses.weighted_losses import WeightedLoss

import ptwt
import pywt

# -------------------------------
# WPT loss (wavelet packet, WaveletPacket2D)
# -------------------------------
class PixelLossWPT(nn.Module):
    """
    Compute pixel reconstruction loss in WPT (WaveletPacket2D) domain at a fixed `level`.

    L = λ_rec * sum_{node in leaves(level)} w(node) * loss(wp_hat[node], wp_gt[node])

    Args:
        wavelet: pywt wavelet name or object
        level:   WPT depth
        mode:    boundary mode ('reflect' | 'constant' | 'zero' | 'boundary' ...)
        rec_weight: scalar multiplier
        base_loss_fn_str: "mse" | "l1"
        packet_weight_fn: optional callable(key:str)->float to weight each packet;
                          default: weights all-'a' path lower (approx) and others higher.
                          Example: lambda key: 0.5 if set(key)=={'a'} else 1.0
        separable: pass True to use separable implementation (matches fs* APIs)
    """

    def __init__(
        self,
        wavelet: str | pywt.Wavelet = "db2",
        level: int = 3,
        mode: str = "reflect",
        rec_weight: float = 1.0,
        base_loss_fn_str: str = "mse",
        packet_weight_fn: Optional[callable] = None,
        separable: bool = False,
    ):
        super().__init__()
        self.wavelet = wavelet
        self.level = level
        self.mode = mode
        self.rec_weight = rec_weight
        self.rec_loss_fn = WeightedLoss(base_loss_fn_str=base_loss_fn_str)
        self.separable = separable

        if packet_weight_fn is None:
            # default: downweight the pure-lowpass path like 'aaa...'
            def _w(key: str) -> float:
                return 0.5 if set(key) == {"a"} else 1.0
            self.packet_weight_fn = _w
        else:
            self.packet_weight_fn = packet_weight_fn

    @torch.no_grad()
    def _merge_batch_channel(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        n, c = x.shape[:2]
        return x.reshape(n * c, *x.shape[2:]), (n, c)

    def forward(self, train_output: Dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        preds = train_output["preds"]   # [N, C, H, W]
        targets = train_output["targets"]
        # (If you really want to use `weights`, you can pass a dict per key later; we ignore here.)
        _ = train_output.get("weights", None)

        p_flat, nc = self._merge_batch_channel(preds)
        t_flat, _ = self._merge_batch_channel(targets)

        # Build packet trees (lazy). We transform entire [B, H, W] batches; last two axes are transformed.
        wp_p = ptwt.WaveletPacket2D(
            data=p_flat, wavelet=self.wavelet, mode=self.mode, maxlevel=self.level, separable=self.separable
        )
        wp_t = ptwt.WaveletPacket2D(
            data=t_flat, wavelet=self.wavelet, mode=self.mode, maxlevel=self.level, separable=self.separable
        )

        # Get all leaves at `level` in natural order (flat list of keys like "avd", "ddh", etc.)
        keys: List[str] = ptwt.WaveletPacket2D.get_natural_order(self.level)

        loss = torch.zeros((), device=p_flat.device, dtype=p_flat.dtype)
        for k in keys:
            w = float(self.packet_weight_fn(k))
            if w == 0.0:
                continue
            loss = loss + w * self.rec_loss_fn(wp_p[k], wp_t[k], None)

        total = self.rec_weight * loss
        return total
