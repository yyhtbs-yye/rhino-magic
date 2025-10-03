import torch

from rhino.boats.gan.base_gan_boat import BaseGanBoat

from trainer.utils.ddp_utils import get_raw_module

class GanWeightClippingBoat(BaseGanBoat):

    def __init__(self, config={}):
        super().__init__(config=config)
        self.clip_value = 0.01

    def d_step(self, batch, scaler): # start_new_accum, scaler, loss_scale, should_step_now):

        loss_dict = super().d_step(batch, scaler)

        with torch.no_grad():
            for p in get_raw_module(self.models['critic']).parameters():
                p.clamp_(-self.clip_value, self.clip_value)

        return loss_dict