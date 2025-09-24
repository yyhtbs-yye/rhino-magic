import torch

from rhino.boats.base.base_boat import BaseBoat
from trainer.utils.build_components import build_module

from trainer.utils.ddp_utils import move_to_device

class BaseDiffusionBoat(BaseBoat):

    def __init__(self, boat_config=None, optimization_config=None, validation_config=None):
        super().__init__(boat_config, optimization_config, validation_config)

        assert boat_config is not None, "boat_config must be provided"

        self.models['net'] = build_module(boat_config['net'])
        self.models['scheduler'] = build_module(boat_config['scheduler'])
        self.models['solver'] = build_module(boat_config['solver'])

        self.boat_config = boat_config
        self.optimization_config = optimization_config or {}
        self.validation_config = validation_config or {}

        self.use_ema = self.optimization_config.get('use_ema', False)
        self.use_reference = validation_config.get('use_reference', False)

        if self.use_ema:
            self._setup_ema()
            self.ema_start = self.optimization_config.get('ema_start', 0)

    def predict(self, x0):
        
        network_in_use = self.models['net_ema'] if self.use_ema and 'net_ema' in self.models else self.models['net']

        hx1 = self.models['solver'].solve(network_in_use, x0)

        result = torch.clamp(hx1, -1, 1)
        
        return result
        
    def training_calc_losses(self, batch, batch_idx):

        x1 = batch['gt']

        batch_size = x1.size(0)
        
        # Initialize random noise in latent space
        x0 = torch.randn_like(x1)

        # Sample random timesteps
        timesteps = self.models['scheduler'].sample_timesteps(batch_size, self.device)
        
        # Add noise to latents according to noise schedule
        xt = self.models['scheduler'].perturb(x1, x0, timesteps)

        # Get targets for the denoising process
        targets = self.models['scheduler'].get_targets(x1, x0, timesteps)
        
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

        x1 = batch['gt']

        with torch.no_grad():

            x0 = torch.randn_like(x1)

            hx1 = self.predict(x0)

            valid_output = {'preds': hx1, 'targets': x1,}

            metrics = self._calc_metrics(valid_output)

            named_imgs = {'groundtruth': x1, 'generated': hx1,}

        return metrics, named_imgs