import torch
from rhino.boats.base.base_boat import BaseBoat
from trainer.utils.build_components import build_module
'''
from trainer.utils.build_components import build_optimizer
import accelerator
'''
class ControlNetBoat(BaseBoat):

    def __init__(self, boat_config=None, optimization_config=None, validation_config=None):
        super().__init__()

        assert boat_config is not None, "boat_config must be provided"

        self.models['net'] = build_module(boat_config['net'])

        self.models['scheduler'] = build_module(boat_config['scheduler'])
        self.models['solver'] = build_module(boat_config['solver'])
        self.models['condition_encoder'] = build_module(boat_config['condition_encoder'])
        self.models['latent_encoder'] = build_module(boat_config['latent_encoder'])
        self.models['prompt_tokenizer'] = build_module(boat_config['prompt_tokenizer'])
        self.models['prompt_encoder'] = build_module(boat_config['prompt_encoder'])
        

        self.boat_config = boat_config
        self.optimization_config = optimization_config or {}
        self.validation_config = validation_config or {}

        self.use_ema = self.optimization_config.get('use_ema', False)
        self.use_reference = validation_config.get('use_reference', False)
        self.image_as_condition = validation_config.get('image_as_condition', False)

        if self.use_ema:
            self._setup_ema()
            self.ema_start = self.optimization_config.get('ema_start', 0)

        '''
    def configure_optimizers(self):
        super().configure_optimizers()

        self.models['net'], self.optimizers['net'], self.lr_schedulers['net'] = accelerator.prepare(
            self.models['net'], self.optimizers['net'], self.lr_schedulers['net']
        )
        # Move models to device
        self.models['latent_encoder'].to(accelerator.device)
        self.models['prompt_encoder'].to(accelerator.device)
        '''

    def forward(self, z0, prompt=None, c=None):
        
        network_in_use = self.models['net_ema'] if self.use_ema and 'net_ema' in self.models else self.models['net']

        # Prompt → CLIP embeddings -----------------------------------------
        if prompt is not None and self.models.get("prompt_tokenizer") is not None and self.models.get("prompt_encoder") is not None:
            # We assume the prompt_encoder expects a list/str and returns embeddings.
            tokens = self.models['prompt_tokenizer'](prompt)
            prompt_latents = self.models["prompt_encoder"](tokens)
        else:
            prompt_latents = None

        if c is not None and self.models.get("condition_encoder")  is not None:
            condition_latents = self.models["condition_encoder"](c)
        else:
            condition_latents = None
            
        hz1 = self.models['solver'].solve(network_in_use, z0, prompt_latents, condition_latents)

        hx1 = self.decode_latents(hz1)
        
        result = torch.clamp(hx1, -1, 1)
        
        return result
        
    def training_calc_losses(self, batch, batch_idx):

        x1 = batch["gt"]                       # target RGB
        c = batch.get("cond", None)           # spatial condition
        prompt = batch.get("prompt", None)    # text prompt(s)

        if prompt is not None and self.models.get("prompt_tokenizer") is not None and self.models.get("prompt_encoder") is not None:
            # We assume the prompt_encoder expects a list/str and returns embeddings.
            tokens = self.models['prompt_tokenizer'](prompt)
            prompt_latents = self.models["prompt_encoder"](tokens)
        else:
            prompt_latents = None

        if c is not None and self.models.get("condition_encoder")  is not None:
            condition_latents = self.models["condition_encoder"](c)
        else:
            condition_latents = None

        batch_size = x1.size(0)
        
        with torch.no_grad():
            z1 = self.encode_images(x1)

        # Initialize random noise in latent space
        z0 = torch.randn_like(z1)

        timesteps = self.models['scheduler'].sample_timesteps(batch_size, self.device)
        zt = self.models['scheduler'].perturb(z1, z0, timesteps)
        targets = self.models['scheduler'].get_targets(z1, z0, timesteps)
        
        preds = self.models['net'](zt, timesteps, 
                                   prompt_latents=prompt_latents, condition_latents=condition_latents)['sample']
        
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
        c = batch.get("cond", None)
        prompt = batch.get("prompt", None)

        with torch.no_grad():

            z1 = self.encode_images(x1)

            z0 = torch.randn_like(z1)

            hx1 = self.forward(z0, prompt, c)

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