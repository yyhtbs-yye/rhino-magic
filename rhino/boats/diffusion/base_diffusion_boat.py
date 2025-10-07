import torch

from rhino.boats.base.base_boat import BaseBoat
from trainer.utils.build_components import build_module

from trainer.utils.ddp_utils import move_to_device

class BaseDiffusionBoat(BaseBoat):

    def __init__(self, config={}):
        super().__init__(config=config)

        assert config is not None, "main config must be provided"

        # Store configurations
        self.boat_config = config.get('boat', {})
        self.optimization_config = config.get('optimization', {})
        self.validation_config = config.get('validation', {})

        self.use_ema = self.optimization_config.get('use_ema', False)
        self.use_reference = self.validation_config.get('use_reference', False)

        if self.use_ema:
            self._setup_ema()
            self.ema_start = self.optimization_config.get('ema_start', 0)

    def predict(self, noise):
        
        network_in_use = self.models['net_ema'] if self.use_ema and 'net_ema' in self.models else self.models['net']

        hx1 = self.models['solver'].solve(network_in_use, noise)

        result = torch.clamp(hx1, -1, 1)
        
        return result
        
    def training_calc_losses(self, batch):

        gt = batch['gt']

        batch_size = gt.size(0)
        
        # Initialize random noise in latent space
        noise = torch.randn_like(gt)

        # Sample random timesteps
        timesteps = self.models['scheduler'].sample_timesteps(batch_size, self.device)
        
        # Add noise to latents according to noise schedule
        xt = self.models['scheduler'].perturb(gt, noise, timesteps)

        # Get targets for the denoising process
        targets = self.models['scheduler'].get_targets(gt, noise, timesteps)
        
        hx1 = self.models['net'](xt, timesteps)['sample']
        
        weights = self.models['scheduler'].get_loss_weights(timesteps)

        train_output = {'preds': hx1,
                        'targets': targets,
                        'weights': weights,
                        **batch}
        
        losses = {'total_loss': torch.tensor(0.0, device=self.device)}

        losses['net'] = self.losses['net'](train_output)
        losses['total_loss'] += losses['net']

        return losses
        
    def validation_step(self, batch, batch_idx):

        batch = move_to_device(batch, self.device)

        gt = batch['gt']

        with torch.no_grad():

            noise = torch.randn_like(gt)

            hx1 = self.predict(noise)

            valid_output = {'preds': hx1, 'targets': gt,}

            metrics = self._calc_metrics(valid_output)

            named_imgs = {'groundtruth': gt, 'generated': hx1,}

        return metrics, named_imgs