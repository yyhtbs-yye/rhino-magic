import torch

from rhino.boats.autoregression.base_autoregression_boat import BaseAutoregressionBoat
from trainer.utils.ddp_utils import move_to_device

class LatentAutoregressionBoat(BaseAutoregressionBoat):

    def __init__(self, config={}):
        super().__init__(config=config)

        if self.models['latent_encoder'] is not None:
            for param in self.models['latent_encoder'].parameters():
                param.requires_grad = False

    def predict(self, inputs):

        z_hat = super().predict(inputs)
        x_hat = self.decode_latents(z_hat)
        result = torch.clamp(x_hat, -1, 1)
        
        return result
        
    def training_calc_losses(self, batch):

        gt = batch['gt']

        with torch.no_grad():
            latents = self.encode_images(gt)

        inputs = self._transform_inputs(latents)
        logits = self.models['net'](inputs)
        train_output = {'preds': logits, 'targets': latents, **batch}
        net_loss = self.losses['net'](train_output)
        losses = {'total_loss': net_loss, 'net': net_loss}

        return losses
        
    def validation_step(self, batch, batch_idx):

        batch = move_to_device(batch, self.device)
        gt = batch['gt']

        with torch.no_grad():
            latents = self.encode_images(gt)
            z_zeros = torch.zeros_like(latents)
            x_hat = self.predict(z_zeros)
            valid_output = {'preds': x_hat, 'targets': gt,}
            metrics = self._calc_metrics(valid_output)
            named_imgs = {'groundtruth': gt, 'generated': x_hat,}

        return metrics, named_imgs
    
    def decode_latents(self, z):
        # Scale latents according to VAE configuration
        x = self.models['latent_encoder'].decode(z)
        return x
    
    def encode_images(self, x):
        # Encode images to latent space
        z = self.models['latent_encoder'].encode(x)
        return z
