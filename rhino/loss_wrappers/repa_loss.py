import torch
import torch.nn as nn
from typing import Callable, Union
from rhino.nn.losses.projection_alignment_loss import ProjectionAlignmentLoss
from rhino.nn.losses.weighted_losses import WeightedLoss

class REPALoss(nn.Module):
    """
    High-level wrapper combining
        L = λ_proj * L_proj + λ_rec * L_rec
    """

    def __init__(
        self,
        feature_dim: int,
        target_dim: int,
        feature_extractor_name: str = 'dinov2_patch',
        proj_weight: float = 0.5,
        rec_weight: float = 1.0,
        base_loss_fn_str: str = "mse",              # 'mse' | 'l1'
        proj_kwargs: dict | None = None,
    ):
        super().__init__()
        proj_kwargs = proj_kwargs or {}

        if feature_extractor_name == 'dinov2_patch':
            from rhino.feature_extractors.dinov2_patch import DinoV2Patch
            feature_extractor = DinoV2Patch()
        else:
            raise ValueError(f"Unsupported feature extractor: {feature_extractor_name}")

        self.proj_loss = ProjectionAlignmentLoss(
            feature_dim, target_dim, feature_extractor, **proj_kwargs
        )

        self.rec_loss_fn = WeightedLoss(base_loss_fn_str=base_loss_fn_str)

        self.proj_weight = proj_weight
        self.rec_weight = rec_weight

    def forward(self, train_output):
        zs_hat = train_output["predictions"]
        repa_targets = train_output["repa_targets"]
        recon_targets = train_output["recon_targets"]

        # Optional weights for reconstruction loss
        weights = train_output.get("recon_weights", None)

        loss_proj = self.proj_loss(zs_hat, repa_targets)

        if weights is not None and isinstance(self.rec_loss_fn, WeightedLoss):
            loss_rec = self.rec_loss_fn(zs_hat, recon_targets, weights)
        else:
            loss_rec = self.rec_loss_fn(zs_hat, recon_targets)

        loss_total = self.proj_weight * loss_proj + self.rec_weight * loss_rec
        return loss_total, {"proj": loss_proj, "recon": loss_rec}
