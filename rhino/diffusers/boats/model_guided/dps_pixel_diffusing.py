import torch
import torch.nn.functional as F
from rhino.diffusers.boats.unconditioned.pixel_diffusing import BaseDiffusionModel

class DPSPixelDiffusionModel(BaseDiffusionModel):
    def __init__(self, *args, cfg_scale=3.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_ema = None

    def forward(self, x0, lc0):
        with torch.enable_grad():
            network_in_use = self.model_ema if self.use_ema and hasattr(self, 'network_ema') else self.model
            hx1 = self.solver.solve(network_in_use, x0, measurement=lc0)

        return torch.clamp(hx1, -1, 1)

    def training_step(self, batch, batch_idx):
        # No training, as specified
        return None

    def validation_step(self, batch, batch_idx):
        """
        Validation step with classifier-free guidance.
        """
        x1, lc0 = batch['gt'], batch['lq']
        _, _, H, W = x1.shape
        x0 = torch.randn_like(x1)
        
        # Generate high-quality image with CFG
        hx1 = self.forward(x0, lc0)
        
        # Compute image MSE
        img_mse = F.mse_loss(hx1, x1)
        self.log("val/img_mse", img_mse, on_step=False, on_epoch=True)
        
        # Log additional metrics
        for metric_name, metric in self.metrics.items():
            metric_val = metric(hx1, x1)
            self.log(f"val/{metric_name}", metric_val, on_step=False, on_epoch=True)
        
        # Visualize results
        if self.save_images:
            named_imgs = {
                'groundtruth': x1[:self.num_vis_samples],
                'low-quality': F.interpolate(lc0[:self.num_vis_samples], 
                                            size=(H, W), 
                                            mode='bilinear', 
                                            align_corners=False),
                'sr-quality': hx1[:self.num_vis_samples],
            }
            self._visualize_validation(named_imgs, batch_idx)
        
        return {"val_loss": img_mse}