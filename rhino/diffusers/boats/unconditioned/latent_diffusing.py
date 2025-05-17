import torch
from rhino.diffusers.boats.base.base_diffusing import BaseDiffusionBoat
from trainer.utils.build_components import build_module

class UnconditionedLatentDiffusionBoat(BaseDiffusionBoat):
    
    def __init__(self, boat_config=None, optimization_config=None, validation_config=None):
        super().__init__(boat_config, optimization_config, validation_config)
        
        # Optional encoder/decoder for latent diffusion models
        self.models['encoder'] = build_module(boat_config['encoder'])

        # disable gradient tracking for encoder
        for param in self.models['encoder'].parameters():
            param.requires_grad = False

    def forward(self, z0):
            
        network_in_use = self.models['model_ema'] if self.use_ema and 'model_ema' in self.models else self.models['model']
        
        hz1 = self.models['solver'].solve(network_in_use, z0)
        
        hx1 = self.decode_latents(hz1)
        
        result = torch.clamp(hx1, -1, 1)
        
        return result
        
    def training_step(self, batch, batch_idx):

        x1 = batch['gt']
        
        batch_size = x1.size(0)
        
        # Encode ground truth images to latent space
        with torch.no_grad():
            z1 = self.encode_images(x1)
        
        # Initialize random noise in latent space
        z0 = torch.randn_like(z1)

        # Sample random timesteps
        timesteps = self.models['scheduler'].sample_timesteps(batch_size, self.device)
        
        # Add noise to latents according to noise schedule
        zt = self.models['scheduler'].perturb(z1, z0, timesteps)

        targets = self.models['scheduler'].get_targets(z1, z0, timesteps)
        predictions = self.models['model'](zt, timesteps)['sample']
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

            z1 = self.encode_images(x1)

            timesteps = self.models['scheduler'].sample_timesteps(batch_size, self.device)

            z0 = torch.randn_like(z1)
            
            zt = self.models['scheduler'].perturb(z1, z0, timesteps)

            network_in_use = self.models['model_ema'] if self.use_ema and 'model_ema' in self.models else self.models['model']

            targets = self.models['scheduler'].get_targets(z1, z0, timesteps)
            predictions = network_in_use(zt, timesteps)['sample']

            weights = self.models['scheduler'].get_loss_weights(timesteps)
            
            loss = self.losses['model'](predictions, targets, weights=weights)

            self._log_metric(loss, metric_name='noise_mse', prefix='valid')

            # Generate images from noise
            hx1 = self.forward(z0)

            results = self._calc_reference_quality_metrics(predictions, targets)

            self._log_metrics(results, prefix='valid')

            named_imgs = {'groundtruth': x1, 'generated': hx1,}

            self._visualize_validation(named_imgs, batch_idx)

        return loss
    
    def decode_latents(self, z):
        """Decode latents to images using the VAE."""
        # Scale latents according to VAE configuration
        z = 1 / self.models['encoder'].config.scaling_factor * z
        x = self.models['encoder'].decode(z)
            
        return x
    def encode_images(self, x):
        """Encode images to latents using the VAE."""
        # Encode images to latent space
        z = self.models['encoder'].encode(x)
        z = z * self.models['encoder'].config.scaling_factor
        
        return z
