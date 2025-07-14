import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionAlignmentLoss(nn.Module):
    def __init__(self, feature_dim, target_dim, 
                 feature_extractor=None, 
                 normalize: bool = True, mode: str = "feature"):
        super().__init__()
        assert mode in ("image", "features"), f"Unsupported mode: {mode}"
        self.feature_extractor = feature_extractor
        self.normalize = normalize
        self.mode = mode
        self.proj_head = nn.Linear(feature_dim, target_dim, bias=False)

    def _extract_reference(self, targets):
        if self.mode == "image":
            with torch.no_grad():
                targets = self.feature_extractor(targets)
        elif self.mode == "feature":
            pass
        else:
            raise ValueError(f"Could not determine how to process targets with mode='{self.mode}'.")

        return targets

    def forward(self, features, targets):
        targets = self._extract_reference(targets)

        if len(features) != len(targets): # Ensure features and zs_ref have the same length (batch size)
            raise ValueError(f"Mismatched lengths: features={len(features)} vs zs_ref={len(targets)}")

        loss = 0.0
        for z_pred, z_ref in zip(features, targets):
            z_pred = self.proj_head(z_pred)
            if self.normalize:
                z_pred = F.normalize(z_pred, dim=-1)
                z_ref  = F.normalize(z_ref,  dim=-1)
            loss += -(z_pred * z_ref).sum(dim=-1).mean()

        return loss / len(features)
