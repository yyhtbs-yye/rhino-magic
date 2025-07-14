#!/usr/bin/env python
from typing import List, Sequence, Union
import math, pathlib
import torch, timm
from timm.data import resolve_model_data_config
from torchvision import transforms
from torchvision.transforms import InterpolationMode

class DinoV2Patch:
    """
    Light-weight wrapper around the ViT-G/14 DINOv2 backbone that returns
    *spatial* patch tokens cropped to a 32x32 grid:  **(B, 1024, D)**.
    """

    def __init__(
        self,
        model_name: str = "vit_giant_patch14_dinov2.lvd142m",
        device: Union[str, torch.device] = "cuda:0",
        pad_256_to_296: bool = True,
    ):
        self.device = torch.device(device)
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,        # drop classifier head
            features_only=False,  # keep CLS + patch tokens
        ).to(self.device).eval()

        # build transform matching the backbone's training recipe
        cfg = resolve_model_data_config(self.model)
        pre = [
            transforms.Pad((0, 0, 40, 40), fill=0)
            if pad_256_to_296
            else transforms.Lambda(lambda x: x)
        ]
        pre += [
            transforms.Resize(cfg["input_size"][1], interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(cfg["input_size"][1]),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg["mean"], std=cfg["std"]),
        ]
        self.transform = transforms.Compose(pre)

        # expose patch-grid size so callers can verify/crop differently if desired
        with torch.no_grad():
            dummy = torch.zeros(1, 3, cfg["input_size"][1], cfg["input_size"][1]).to(self.device)
            n_tokens = self.model.forward_features(dummy).shape[1] - 1  # minus CLS
        self.grid_size = int(math.sqrt(n_tokens))  # 37 for ViT-G/14

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        x : torch.Tensor            shape (B, 3, H, W)
             • already on CPU or GPU  
             • already float32, range **[0 … 1]**  
             • NOT yet normalized to DINOv2’s mean / std

        Returns
        -------
        torch.Tensor  (B, 1024, D)  on self.device
        """
        # DINOv2 expects bicubic-resized 296 × 296, then center-cropped to 224 × 224
        # plus mean/std normalisation.  Apply those here.
        # (If you handled this upstream, just comment these two lines out.)
        x = torch.stack([self.transform(img.cpu()) for img in x])  # keep transform
        x = x.to(self.device)

        # Encode
        feats  = self.model.forward_features(x)   # (B, 1+N, D)
        patch  = feats[:, 1:, :]                  # drop CLS  → (B, N, D)

        # Crop centre 32 × 32 from original 37 × 37 grid
        gs     = self.grid_size                  # 37 for ViT-G/14
        B, N, D = patch.shape
        patch  = (patch
                  .view(B, gs, gs, D)[:, :32, :32, :]   # (B, 32, 32, D)
                  .reshape(B, -1, D))                  # (B, 1024, D)

        return patch
