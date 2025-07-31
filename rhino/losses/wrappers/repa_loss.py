import torch.nn as nn
from rhino.losses.projection_alignment_loss import ProjectionAlignmentLoss
from rhino.losses.weighted_losses import WeightedLoss

class REPALoss(nn.Module):
    """
    High-level wrapper combining
        L = λ_proj * L_proj + λ_rec * L_rec
    """

    def __init__(
        self,
        feature_dim: int,
        target_dim: int,
        feature_extractor_name = None,
        proj_weight: float = 0.5,
        rec_weight: float = 1.0,
        base_loss_fn_str: str = "mse",              # 'mse' | 'l1'
        proj_kwargs: dict | None = None,
    ):
        super().__init__()
        proj_kwargs = proj_kwargs or {}
        self.opt_id = 'net'

        if feature_extractor_name is None:
            feature_extractor = None
        elif feature_extractor_name == 'dinov2_patch':
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
        preds = train_output["preds"]
        targets = train_output["targets"]
        weights = train_output.get("weights", None)

        hook_fx = train_output["hook_fx"]

        fx = train_output["fx"]

        loss_proj = None
        for layer_name in hook_fx:
            fx_preds = hook_fx[layer_name]
            if loss_proj is None:
                loss_proj = self.proj_loss(fx_preds, fx)
            else:
                loss_proj += self.proj_loss(fx_preds, fx)

        if weights is not None and isinstance(self.rec_loss_fn, WeightedLoss):
            loss_rec = self.rec_loss_fn(preds, targets, weights)
        else:
            loss_rec = self.rec_loss_fn(preds, targets)

        loss_total = self.proj_weight * loss_proj + self.rec_weight * loss_rec
        return loss_total, {"proj": loss_proj, "recon": loss_rec}
