import torch

from rhino.boats.diffusion.base_diffusion_boat import BaseDiffusionBoat
from trainer.utils.build_components import build_module

from trainer.utils.ddp_utils import move_to_device

class SR3DiffusionBoat(BaseDiffusionBoat):

    def __init__(self, boat_config=None, optimization_config=None, validation_config=None):
        super().__init__(boat_config, optimization_config, validation_config)

        self.models['lq_img_encoder'] = build_module(boat_config['lq_img_encoder'])
        
        self.use_reference = validation_config.get('use_reference', True)

    def predict(self, noise, lq=None):
        
        network_in_use = self.models['net_ema'] if self.use_ema and 'net_ema' in self.models else self.models['net']

        h_lq = self.models['lq_img_encoder'](lq)
            
        x0_hat = self.models['solver'].solve(network_in_use, noise, h_lq)
        
        result = torch.clamp(x0_hat, -1, 1)
        
        return result
        
    def training_calc_losses(self, batch, batch_idx):

        gt = batch['gt']
        lq = batch.get('lq', None)

        h_lq = self.models['lq_img_encoder'](lq)

        batch_size = gt.size(0)

        noise = torch.randn_like(gt)

        timesteps = self.models['scheduler'].sample_timesteps(batch_size, self.device)
        
        xt = self.models['scheduler'].perturb(gt, noise, timesteps)

        targets = self.models['scheduler'].get_targets(gt, noise, timesteps)
        
        x0_hat = self.models['net'](xt, timesteps, encoder_hidden_states=h_lq)['sample']
        
        weights = self.models['scheduler'].get_loss_weights(timesteps)

        train_output = {
            'preds': x0_hat,
            'targets': targets,
            'weights': weights,
            **batch
        }
        
        losses = {'total_loss': torch.tensor(0.0, device=self.device)}

        losses['net'] = self.losses['net'](train_output)
        losses['total_loss'] += losses['net']

        return losses
        
    def validation_step(self, batch, batch_idx):

        batch = move_to_device(batch, self.device)

        gt = batch['gt']
        lq = batch.get('lq', None)

        with torch.no_grad():

            noise = torch.randn_like(gt)

            x0_hat = self.predict(noise, lq)

            valid_output = {'preds': x0_hat, 'targets': gt,}

            metrics = self._calc_metrics(valid_output)

            named_imgs = {'groundtruth': gt, 'generated': x0_hat,}

            if lq is not None:
                named_imgs['low-quality'] = lq

        return metrics, named_imgs
    