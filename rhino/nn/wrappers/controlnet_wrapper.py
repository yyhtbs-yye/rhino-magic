from diffusers.models.controlnet import ControlNetModel
from diffusers.models.unets import UNet2DConditionModel
import torch.nn as nn

class ControlNetWrapper(nn.Module):

    def __init__(self, unet_name="CompVis/stable-diffusion-v1-4", controlnet_name=None,
                 frozen='unet',
                 enable_xformers_memory_efficient_attention=True):
        super().__init__()

        self.unet = UNet2DConditionModel.from_pretrained(unet_name, subfolder="unet")

        if controlnet_name is None:
            self.controlnet = ControlNetModel.from_unet(self.unet)
        else:
            self.controlnet = ControlNetModel.from_pretrained(controlnet_name)

        self.frozen = frozen
        
        if frozen is None:
            pass
        elif frozen == 'unet':
            self.unet.requires_grad=False
        elif frozen == 'all':
            self.unet.requires_grad=False
            self.controlnet.requires_grad=False
        else:
            print(f'frozen {frozen} is not recognized')
            pass

        
        self.unet_name = unet_name
        self.controlnet_name = controlnet_name

        self.enable_xformers_memory_efficient_attention = enable_xformers_memory_efficient_attention

        # Enable xformers if requested
        if enable_xformers_memory_efficient_attention:
            if hasattr(self.unet, "enable_xformers_memory_efficient_attention"):
                self.unet.enable_xformers_memory_efficient_attention()
            if hasattr(self.controlnet, "enable_xformers_memory_efficient_attention"):
                self.controlnet.enable_xformers_memory_efficient_attention()

    def forward(self, noisy_latents, timesteps, prompt_latents, condition_latents):
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=prompt_latents,
            controlnet_cond=condition_latents,
            return_dict=False,
        )
        
        model_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=prompt_latents,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
        )

        return model_pred
    
    @classmethod
    def from_config(cls, config):
        """
        Args:
            config (dict | str | pathlib.Path):
                * A dict produced by ``model.to_config()``, **or**
        """

        # Allowed keys + defaults
        kwargs = {
            "unet_name": config.get(
                "unet_name", "CompVis/stable-diffusion-v1-4"
            ),
            "controlnet_name": config.get("controlnet_name"),  # may be None
            "frozen": config.get("frozen", 'unet'),
            "enable_xformers_memory_efficient_attention": config.get(
                "enable_xformers_memory_efficient_attention", True
            ),
        }

        # Warn about unknown keys instead of crashing
        unknown = set(config) - set(kwargs)
        if unknown:
            print(f"[ControlNetWrapper.from_config] Ignoring unknown keys: {unknown}")

        return cls(**kwargs)

    # (Optional) helper to round‑trip configs
    def to_config(self) -> dict:
        return {
            "unet_name": self.unet_name,
            "controlnet_name": self.controlnet_name,
            "frozen": self.frozen,
            "enable_xformers_memory_efficient_attention": self.enable_xformers_memory_efficient_attention,
        }
