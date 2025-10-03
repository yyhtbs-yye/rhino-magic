import torch

from rhino.boats.base.base_boat import BaseBoat
from trainer.utils.ddp_utils import move_to_device
from einops import rearrange

class BaseAutoregressionBoat(BaseBoat):

    def __init__(self, config={}):
        super().__init__(config=config)

        assert config is not None, "main config must be provided"

        self.use_ema = self.optimization_config.get('use_ema', False)
        self.use_reference = self.validation_config.get('use_reference', False)

        self.sampler_config = self.boat_config.get('sampler_config', {})
        self.need_flatten = self.sampler_config.get('need_flatten', False)
        self.use_channel_ar = self.sampler_config.get('use_channel_ar', True)

        if self.use_ema:
            self._setup_ema()
            self.ema_start = self.optimization_config.get('ema_start', 0)

    # ------------------------------- Inference -------------------------------

    def predict(self, inputs, num_steps=None):

        assert inputs.dim() == 4, "predict() expects [B, C, H, W]"

        network_in_use = self.models['net_ema'] if self.use_ema and 'net_ema' in self.models else self.models['net']

        b, c, h, w = inputs.shape

        x_hat = network_in_use.sample(self._transform_inputs(inputs), 
                                      steps=self._infer_num_steps((b, c, h, w) if self.need_flatten else None, num_steps))

        outputs = rearrange(x_hat, 'b (h w c) -> b c h w', c=c, h=h, w=w) if self.need_flatten else x_hat

        return outputs
    # ------------------------------- Training -------------------------------

    def training_calc_losses(self, batch):

        gt = batch['gt']  # (B, C, H, W)

        inputs = self._transform_inputs(gt)
        
        # Forward AR net
        logits = self.models['net'](inputs)

        train_output = {'preds': logits, 'targets': inputs, **batch}
        
        net_loss = self.losses['net'](train_output)

        losses = {'total_loss': net_loss, 'net': net_loss}
        return losses

    # ------------------------------- Validation -------------------------------

    def validation_step(self, batch, batch_idx):

        batch = move_to_device(batch, self.device)
        gt = batch['gt']

        x_zeros = torch.zeros_like(gt)

        with torch.no_grad():
            # Generate with matching latent grid size
            x_hat = self.predict(x_zeros)
            valid_output = {'preds': x_hat, 'targets': gt}
            metrics = self._calc_metrics(valid_output)
            named_imgs = {'groundtruth': gt, 'generated': x_hat}

        return metrics, named_imgs
    
    def _transform_inputs(self, inputs):

        outputs = rearrange(inputs, 'b c h w -> b (h w c)') if self.need_flatten else inputs
        
        return outputs

    def _infer_num_steps(self, original_shape, num_steps):
        if num_steps is not None:
            return int(num_steps)

        if len(original_shape) == 4:
            _, c, h, w = original_shape
            return int(h * w * c) if self.use_channel_ar else int(h * w)
        elif len(original_shape) == 2:
            # Fallback: if we only see [B, N], we can't reliably "unpack" H, W, C.
            # Use N as-is.
            return int(original_shape[1])
        else:
            raise ValueError(f"Unexpected input shape {original_shape}, cannot infer num_steps.")
        