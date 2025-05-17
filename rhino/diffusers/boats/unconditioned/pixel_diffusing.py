import torch
from rhino.diffusers.boats.base.base_diffusing import BaseDiffusionBoat

class UnconditionedPixelDiffusionBoat(BaseDiffusionBoat):
    
    def forward(self, x0):

        network_in_use = self.models['model_ema'] if self.use_ema and 'model_ema' in self.models else self.models['model']

        hx1 = self.models['solver'].solve(network_in_use, x0)
        
        return torch.clamp(hx1, -1, 1)
        
    def training_step(self, batch, batch_idx):

        x1 = batch['gt']
        
        batch_size = x1.size(0)
        
        # Add noise to images
        x0 = torch.randn_like(x1)

        # Sample random timesteps
        timesteps = self.models['scheduler'].sample_timesteps(batch_size, self.device)
        
        # Add noise to images according to noise schedule
        xt = self.models['scheduler'].perturb(x1, x0, timesteps)

        targets = self.models['scheduler'].get_targets(x1, x0, timesteps)
        predictions = self.models['model'](xt, timesteps)
        
        if hasattr(predictions, 'sample'):
            predictions = predictions.sample

        weights = self.models['scheduler'].get_loss_weights(timesteps)

        loss = self.losses['model'](predictions, targets, weights=weights)

        self._step(loss)
        
        # Log metrics
        self._log_metric(loss, metric_name='noise_mse', prefix='train')
        
        # Update EMA
        if self.use_ema and self.get_global_step() >= self.ema_start:
            self._update_ema()
        
        return loss
    
    def validation_step(self, batch, batch_idx):

        
        with torch.no_grad():

            x1 = batch['gt']

            batch_size = x1.size(0)

            timesteps = self.models['scheduler'].sample_timesteps(batch_size, self.device)

            x0 = torch.randn_like(x1)

            xt = self.models['scheduler'].perturb(x1, x0, timesteps)

            network_in_use = self.models['model_ema'] if self.use_ema and 'model_ema' in self.models else self.models['model']

            targets = self.models['scheduler'].get_targets(x1, x0, timesteps)
            predictions = network_in_use(xt, timesteps)

            if hasattr(predictions, 'sample'):
                predictions = predictions.sample

            weights = self.models['scheduler'].get_loss_weights(timesteps)

            loss = self.losses['model'](predictions, targets, weights=weights)

            self._log_metric(loss, metric_name='noise_mse', prefix='valid')

            results = self._calc_reference_quality_metrics(predictions, targets)

            self._log_metrics(results, prefix='valid')

            hx1 = self.forward(x0)

            named_imgs = {'groundtruth': x1, 'generated': hx1,}

            self._visualize_validation(named_imgs, batch_idx)

        return loss