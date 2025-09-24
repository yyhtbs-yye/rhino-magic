import torch
import torch.nn as nn
from typing import Dict, Any
from rhino.losses.weighted_losses import WeightedLoss

class PixelLoss(nn.Module):
    """
    Pure pixel-reconstruction loss.

    L_total = λ_rec * L_rec
      where L_rec is either MSE or L1 (optionally weighted per-pixel).

    Expected keys in `train_output`:
        ├─ "preds"          - output tensor from the model
        ├─ "targets"            - ground-truth image / feature map
        └─ "recon_weights"  - (optional) per-pixel weights for L_rec

    Any extra keys (e.g. "repa_targets") are ignored so existing
    dataloaders / trainers don't need to change.
    """

    def __init__(
        self,
        rec_weight: float = 1.0,
        base_loss_fn_str: str = "mse",   # "mse" | "l1"
    ):
        super().__init__()
        self.rec_loss_fn = WeightedLoss(base_loss_fn_str=base_loss_fn_str)
        self.rec_weight = rec_weight

    def forward(
        self, train_output: Dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        preds = train_output["preds"]
        targets = train_output["targets"]
        weights = train_output.get("weights", None)

        loss_rec = self.rec_loss_fn(preds, targets, weights)

        total = self.rec_weight * loss_rec

        return total
