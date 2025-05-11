import torch
import inspect
from diffusers import DDIMScheduler
import importlib

class PosteriorSamplingConditioner:
    def __init__(self, operator, scale=0.5):
        self.operator = operator
        self.scale = scale

    def condition_step(self, x_prev, x_t, x_0_hat, measurement, mask=None):

        Ax = self.operator(x_0_hat)
        
        if mask is not None:
            Ax, measurement = Ax * mask, measurement * mask
        
        difference = measurement - Ax
        norm_value = torch.linalg.norm(difference)
        norm_grad = torch.autograd.grad(outputs=norm_value, inputs=x_prev, create_graph=False)[0]
        
        x_t -= self.scale * norm_grad
        return x_t, norm_value.item()

class DPSConditionedSampler:
    def __init__(self, sampler, eta, operator, scale=0.5):
        self.sampler = sampler
        self.eta = eta
        self.conditioner = PosteriorSamplingConditioner(operator, scale)

    @staticmethod
    def get_class(path, name):
        module = importlib.import_module(path)
        return getattr(module, name)

    @staticmethod
    def build_module(config):
        class_ = DPSConditionedSampler.get_class(config['path'], config['name'])
        if not hasattr(class_, 'from_config'):
            raise AttributeError(f"Class {config['name']} must implement 'from_config' method")
        return class_.from_config(config['config'])
    
    @staticmethod
    def from_config(config):
        config = config.copy()
        num_inference_steps = config.pop('num_inference_steps', 50)
        eta = config.pop('eta', None)
        operator = DPSConditionedSampler.build_module(config['operator'])
        scale = config.pop('scale', 0.5)

        sampler = DDIMScheduler.from_config(config)
        sampler.set_timesteps(num_inference_steps)

        return DPSConditionedSampler(sampler=sampler, eta=eta, operator=operator, scale=scale)

    # @torch.no_grad()
    def solve(self, network, noise, measurement=None, mask=None, seed=None) -> torch.Tensor:
        generator = None

        if seed is not None:
            generator = torch.Generator(device=network.device).manual_seed(seed)
        
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, self.eta)
        
        noise = noise * self.sampler.init_noise_sigma
        
        timesteps_iter = self.sampler.timesteps.to(noise.device)

        for i, t in enumerate(timesteps_iter):
            model_input = self.sampler.scale_model_input(noise, t)
            noise_pred = network(model_input, t)

            noise_pred, noise_var = torch.split(noise_pred, model_input.shape[1], dim=1)

            if hasattr(noise_pred, 'sample'):
                noise_pred = noise_pred.sample
            
            alpha_t = self.sampler.alphas_cumprod[t].to(noise.device)
            with torch.enable_grad():
                noise.requires_grad_(True)
                x_t = self.sampler.step(noise_pred, t, noise, **extra_step_kwargs)["prev_sample"]
                x_0_hat = (noise - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)

                if self.conditioner is not None and measurement is not None:
                    
                    noise, _ = self.conditioner.condition_step(noise, x_t, x_0_hat, measurement, mask)
                
            noise = noise.detach()
        return noise
    
    def prepare_extra_step_kwargs(self, generator, eta):
        """Prepare extra kwargs for the scheduler step."""
        accepts_eta = 'eta' in set(inspect.signature(self.sampler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta and eta is not None:
            extra_step_kwargs['eta'] = eta
        
        accepts_generator = 'generator' in set(inspect.signature(self.sampler.step).parameters.keys())
        if accepts_generator and generator is not None:
            extra_step_kwargs['generator'] = generator
        return extra_step_kwargs