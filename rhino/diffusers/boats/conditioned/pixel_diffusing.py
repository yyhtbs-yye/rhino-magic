import torch
from rhino.diffusers.boats.base.base_diffusing import BaseDiffusionBoat
from trainer.utils.build_components import build_module

class ConditionedPixelDiffusionBoat(BaseDiffusionBoat):

    def __init__(self, boat_config=None, optimization_config=None, validation_config=None):
        super().__init__(boat_config, optimization_config, validation_config)
        
        # Condition mapper processes conditioning inputs (text, class labels, etc.)
        self.models['context_encoder'] = build_module(boat_config['context_encoder'])

        self.use_reference = validation_config['use_reference']

    def forward(self, x0, c=None):
            
        network_in_use = self.models['model_ema'] if self.use_ema and 'model_ema' in self.models else self.models['model']

        encoder_hidden_states = self.models['context_encoder'](c)

        hx1 = self.models['solver'].solve(network_in_use, x0, encoder_hidden_states)

        result = torch.clamp(hx1, -1, 1)
        
        return result

    def training_step(self, batch, batch_idx):

        x1 = batch['gt']
        c = batch['ctx']

        batch_size = x1.size(0)

        encoder_hidden_states = self.models['context_encoder'](c)

        # Add x0 to images
        x0 = torch.randn_like(x1)
        
        # Sample random timesteps
        timesteps = self.models['scheduler'].sample_timesteps(batch_size, self.device)
        
        # Add noise to images according to noise schedule
        xt = self.models['scheduler'].perturb(x1, x0, timesteps)

        targets = self.models['scheduler'].get_targets(x1, x0, timesteps)
        predictions = self.models['model'](xt, timesteps, encoder_hidden_states)['sample']
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

        x1 = batch['gt']
        c = batch['ctx']

        batch_size = x1.size(0)

        with torch.no_grad():

            timesteps = self.models['scheduler'].sample_timesteps(batch_size, self.device)

            x0 = torch.randn_like(x1)

            encoder_hidden_states = self.models['context_encoder'](c)

            xt = self.models['scheduler'].perturb(x1, x0, timesteps)

            network_in_use = self.models['model_ema'] if self.use_ema and 'model_ema' in self.models else self.models['model']

            targets = self.models['scheduler'].get_targets(x1, x0, timesteps)
            predictions = network_in_use(xt, timesteps, encoder_hidden_states=encoder_hidden_states)['sample']

            weights = self.models['scheduler'].get_loss_weights(timesteps)
            
            loss = self.losses['model'](predictions, targets, weights=weights)

            self._log_metric(loss, metric_name='noise_mse', prefix='valid')

            hx1 = self.forward(x0, c)
            if self.use_reference:
                results = self._calc_reference_quality_metrics(hx1, x1)
            else:
                results = self._calc_reference_quality_metrics(predictions, targets)

            self._log_metrics(results, prefix='valid')

            named_imgs = {'groundtruth': x1, 'generated': hx1,}
            if self.use_reference:
                named_imgs['reference'] = c

            self._visualize_validation(named_imgs, batch_idx)

        return loss