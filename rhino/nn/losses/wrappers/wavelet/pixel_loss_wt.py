import torch
import torch.nn as nn
from typing import Dict, Any, Iterable, Tuple, List, Optional
from rhino.nn.losses.weighted_loss import WeightedLoss

import ptwt
import pywt

from rhino.nn.losses.wrappers.patches import ptwt_wavedec2

# -------------------------------
# WT loss (discrete wavelet, wavedec2)
# -------------------------------
class PixelLossWT(nn.Module):
    """
    Compute pixel reconstruction loss in WT (wavedec2) domain.

    L = λ_rec * [ wA * loss(cA_hat, cA_gt) + sum_{s=1..L} wD[s] * (loss(cH_s_hat,cH_s_gt)+loss(cV_s_hat,cV_s_gt)+loss(cD_s_hat,cD_s_gt)) ]

    Args:
        wavelet: pywt wavelet name or pywt.Wavelet (e.g. 'db2', 'haar', 'bior4.4')
        level:   WT levels
        mode:    boundary mode (ptwt padding mode, e.g. 'reflect', 'zero', 'constant')
        rec_weight: scalar multiplier on the whole loss
        base_loss_fn_str: "mse" | "l1" (passed into your WeightedLoss)
        approx_weight: weight for the deepest approximation band cA_L
        detail_weight: either a single scalar for all detail bands, or a list/tuple
                       of length `level` (index 1..level, highest scale first) to weight each level.
                       Example for level=3: detail_weight=[1.0, 0.7, 0.5]
    """

    def __init__(
        self,
        wavelet: str | pywt.Wavelet = "db2",
        level: int = 3,
        mode: str = "reflect",
        rec_weight: float = 1.0,
        base_loss_fn_str: str = "mse",
        approx_weight: float = 1.0,
        detail_weight: float | Iterable[float] = 1.0,
    ):
        super().__init__()
        self.wavelet = wavelet
        self.level = level
        self.mode = mode
        self.rec_weight = rec_weight
        self.rec_loss_fn = WeightedLoss(base_loss_fn_str=base_loss_fn_str)

        # normalize detail weights to per-level list [w_L, w_{L-1}, ..., w_1]
        if isinstance(detail_weight, (list, tuple)):
            dw = list(detail_weight)
            assert len(dw) == level, f"detail_weight must have length={level}"
            self.detail_weights = dw
        else:
            self.detail_weights = [float(detail_weight)] * level

        self.approx_weight = float(approx_weight)

    def _merge_batch_channel(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        # expects [N, C, H, W]; merge to [N*C, H, W] for ptwt convenience
        n, c = x.shape[:2]
        xc = x.reshape(n * c, *x.shape[2:])
        return xc, (n, c)

    def _split_batch_channel(self, x: torch.Tensor, shape_nc: Tuple[int, int]) -> torch.Tensor:
        n, c = shape_nc
        return x.reshape(n, c, *x.shape[-2:])

    def _wavedec2_per_tensor(self, x: torch.Tensor):
        # ptwt.wavedec2 transforms over the last two axes; here x is [B, H, W] or [H, W]
        return ptwt_wavedec2.wavedec2(x, self.wavelet, level=self.level, mode=self.mode)

    def forward(self, train_output: Dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        preds = train_output["preds"]   # [N, C, H, W]
        targets = train_output["targets"]
        weights = train_output.get("weights", None)  # optional per-pixel weights

        # merge N and C → do a single transform per map
        p_flat, nc = self._merge_batch_channel(preds)
        t_flat, _ = self._merge_batch_channel(targets)

        # WT: coeffs = [cA_L, (cH_L,cV_L,cD_L), ..., (cH_1,cV_1,cD_1)]
        coeffs_p = self._wavedec2_per_tensor(p_flat)
        coeffs_t = self._wavedec2_per_tensor(t_flat)

        # loss on approximation band
        cA_p = coeffs_p[0]
        cA_t = coeffs_t[0]

        # broadcast weights to cA if provided
        if weights is not None:
            # downsampled bands have smaller spatial size; we can't reuse pixel weights directly
            # so we ignore "weights" for WT by default (or you could downsample weights to match)
            wA = None
        else:
            wA = None

        loss = self.approx_weight * self.rec_loss_fn(cA_p, cA_t, wA)

        # loss on detail bands per level (L→1)
        detail_levels_p = coeffs_p[1:]  # list of tuples
        detail_levels_t = coeffs_t[1:]

        for w_level, (p_triplet, t_triplet) in zip(self.detail_weights, zip(detail_levels_p, detail_levels_t)):
            cH_p, cV_p, cD_p = p_triplet
            cH_t, cV_t, cD_t = t_triplet
            # No per-pixel weights here for the same downsampling reason
            loss += w_level * (
                self.rec_loss_fn(cH_p, cH_t, None)
                + self.rec_loss_fn(cV_p, cV_t, None)
                + self.rec_loss_fn(cD_p, cD_t, None)
            )

        total = self.rec_weight * loss
        return total
