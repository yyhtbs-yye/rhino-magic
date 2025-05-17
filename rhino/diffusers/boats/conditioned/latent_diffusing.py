import torch
from rhino.diffusers.boats.unconditioned.latent_diffusing import UnconditionedLatentDiffusionBoat
from trainer.utils.build_components import build_module

from trainer.debuggers.memory_leaks import *

class ConditionedLatentDiffusionBoat(UnconditionedLatentDiffusionBoat):
    
    def __init__(self, boat_config=None, optimization_config=None, validation_config=None, memory_debug=False):
        super().__init__(boat_config, optimization_config, validation_config)
        
        # Condition mapper processes conditioning inputs (text, class labels, etc.)
        self.models['context_encoder'] = build_module(boat_config['context_encoder'])

        self.use_reference = validation_config['use_reference']

        if memory_debug:
            original_training_step = self.training_step
            original_step = self._step
            original_log_metric = self._log_metric

            self.training_step = track_object_increases(original_training_step)
            self._step = track_object_increases(original_step)
            self._log_metric = track_object_increases(original_log_metric)

    def forward(self, z0, c):
        
        network_in_use = self.models['model_ema'] if self.use_ema and 'model_ema' in self.models else self.models['model']

        encoder_hidden_states = self.models['context_encoder'](c)
        
        hz1 = self.models['solver'].solve(network_in_use, z0, encoder_hidden_states)

        hx1 = self.decode_latents(hz1)
        
        result = torch.clamp(hx1, -1, 1)
        
        return result
        
    def training_step(self, batch, batch_idx):

        x1 = batch['gt']
        c = batch['ctx'] 

        batch_size = x1.size(0)
        
        with torch.no_grad():
            z1 = self.encode_images(x1)

        encoder_hidden_states = self.models['context_encoder'](c)

        # Initialize random noise in latent space
        z0 = torch.randn_like(z1)

        # Sample random timesteps
        timesteps = self.models['scheduler'].sample_timesteps(batch_size, self.device)
        
        # Add noise to latents according to noise schedule
        zt = self.models['scheduler'].perturb(z1, z0, timesteps)

        targets = self.models['scheduler'].get_targets(z1, z0, timesteps)
        predictions = self.models['model'](zt, timesteps, encoder_hidden_states=encoder_hidden_states)['sample']
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

            z1 = self.encode_images(x1)
            z0 = torch.randn_like(z1)

            encoder_hidden_states = self.models['context_encoder'](c)

            zt = self.models['scheduler'].perturb(z1, z0, timesteps)

            network_in_use = self.models['model_ema'] if self.use_ema and 'model_ema' in self.models else self.models['model']

            targets = self.models['scheduler'].get_targets(z1, z0, timesteps)
            predictions = network_in_use(zt, timesteps, encoder_hidden_states=encoder_hidden_states)['sample']

            weights = self.models['scheduler'].get_loss_weights(timesteps)
            
            loss = self.losses['model'](predictions, targets, weights=weights)

            self._log_metric(loss, metric_name='noise_mse', prefix='valid')

            # Generate images from noise
            hx1 = self.forward(z0, c)

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