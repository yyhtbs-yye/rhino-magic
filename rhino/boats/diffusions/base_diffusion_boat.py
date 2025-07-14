import torch

from rhino.boats.base.base_boat import BaseBoat
from trainer.utils.build_components import build_module

class BaseDiffusionBoat(BaseBoat):

    def __init__(self, boat_config=None, optimization_config=None, validation_config=None):
        super().__init__()

        assert boat_config is not None, "boat_config must be provided"

        self.models['net'] = build_module(boat_config['net'])
        self.models['scheduler'] = build_module(boat_config['scheduler'])
        self.models['solver'] = build_module(boat_config['solver'])
        self.models['context_encoder'] = build_module(boat_config['context_encoder']) if 'context_encoder' in boat_config else None
        self.models['latent_encoder'] = build_module(boat_config['latent_encoder']) if 'latent_encoder' in boat_config else None

        self.boat_config = boat_config
        self.optimization_config = optimization_config or {}
        self.validation_config = validation_config or {}

        self.use_ema = self.optimization_config.get('use_ema', False)
        self.use_reference = validation_config.get('use_reference', False)
        self.image_as_condition = validation_config.get('image_as_condition', False)

        if self.use_ema:
            self._setup_ema()
            self.ema_start = self.optimization_config.get('ema_start', 0)

    def forward(self, z0, c=None):
        
        network_in_use = self.models['net_ema'] if self.use_ema and 'net_ema' in self.models else self.models['net']

        context_encoder = self.models.get('context_encoder', None)

        if c is not None and context_encoder is not None:
            encoder_hidden_states = context_encoder(c)
        else:
            encoder_hidden_states = None
            
        hz1 = self.models['solver'].solve(network_in_use, z0, encoder_hidden_states)

        if self.models['latent_encoder'] is not None:
            hx1 = self.decode_latents(hz1)
        else:
            hx1 = hz1
        
        result = torch.clamp(hx1, -1, 1)
        
        return result
        
    def training_calc_losses(self, batch, batch_idx):

        x1 = batch['gt']
        c = batch.get('cond', None)

        if self.models['context_encoder'] is not None:
            encoder_hidden_states = self.models['context_encoder'](c)
        else:
            encoder_hidden_states = None

        batch_size = x1.size(0)
        
        if self.models['latent_encoder'] is not None:
            with torch.no_grad():
                z1 = self.encode_images(x1)
        else:
            z1 = x1

        # Initialize random noise in latent space
        z0 = torch.randn_like(z1)

        # Sample random timesteps
        timesteps = self.models['scheduler'].sample_timesteps(batch_size, self.device)
        
        # Add noise to latents according to noise schedule
        zt = self.models['scheduler'].perturb(z1, z0, timesteps)

        # Get targets for the denoising process
        targets = self.models['scheduler'].get_targets(z1, z0, timesteps)
        
        preds = self.models['net'](zt, timesteps, encoder_hidden_states=encoder_hidden_states)['sample']
        
        weights = self.models['scheduler'].get_loss_weights(timesteps)

        train_output = {
            'preds': preds,
            'targets': targets,
            'weights': weights,
            **batch
        }
        
        losses = self._calc_losses(train_output)

        return losses

    def training_backward(self, losses):

        total_loss = losses['total_loss']
        
        self._backward(total_loss)


    def training_step(self):

        self._step()

        # Update EMA
        if self.use_ema and self.get_global_step() >= self.ema_start:
            self._update_ema()
        
        return

    def validation_step(self, batch, batch_idx):

        x1 = batch['gt']
        c = batch.get('cond', None)

        with torch.no_grad():

            if self.models['latent_encoder'] is not None:
                z1 = self.encode_images(x1)
            else:
                z1 = x1

            z0 = torch.randn_like(z1)

            hx1 = self.forward(z0, c)

            valid_output = {'preds': hx1, 'targets': x1,}

            metrics = self._calc_metrics(valid_output)

            named_imgs = {'groundtruth': x1, 'generated': hx1,}

            if c is not None and self.image_as_condition:
                named_imgs['condition'] = c

        return metrics, named_imgs
    
    def decode_latents(self, z):
        # Scale latents according to VAE configuration
        x = self.models['latent_encoder'].decode(z)
        return x
    
    def encode_images(self, x):
        # Encode images to latent space
        z = self.models['latent_encoder'].encode(x)
        return z
