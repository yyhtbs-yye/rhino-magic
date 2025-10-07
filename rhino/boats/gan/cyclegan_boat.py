# cyclegan_boat.py
import copy
import torch
import torch.nn.functional as F

from rhino.boats.gan.base_gan_boat import BaseGanBoat
from trainer.utils.build_components import build_module
from trainer.utils.ddp_utils import move_to_device

class CycleGanBoat(BaseGanBoat):

    def __init__(self, config={}):

        super().__init__(config=config)

        optimization_config = config.get('optimization', {})
        validation_config = config.get('validation', {})

        # CycleGAN-specific weights
        self.cycle_weight = float(optimization_config.get('hyper_parameters', {}).get('cycle_weight', 0.5))
        self.identity_weight = float(optimization_config.get('hyper_parameters', {}).get('identity_weight', 0.1))  # 0.5 * lambda_cyc in the original paper

    @torch.no_grad()
    def predict(self, tensor, direction: str = 'x2y'):
        """
        Predict with EMA (if available) in one of:
          'x2y' | 'y2x' | 'xyx' | 'yxy'
        """
        net = self.models['net_ema'] if self.use_ema and 'net_ema' in self.models else self.models['net']
        return net(tensor, mode=direction)

    def d_step_calc_losses(self, batch):
        """Discriminators: D_X and D_Y adversarial losses on real vs. generated."""
        x, y = batch['src'], batch['tgt']
        
        x = self.models['net_proc'](x)

        # Generate fakes with no grad to generator
        with torch.no_grad():
            y_hat = self.models['net'](x, mode='x2y')  # X -> Y
            x_hat = self.models['net'](y, mode='y2x')  # Y -> X

        # D_Y: real y vs. fake y_hat
        d_y_real, d_x_real = self.models['critic']((y, x), mode='yx')
        d_y_fake, d_x_fake = self.models['critic']((y_hat, x_hat), mode='yx')

        d_loss_y = self.losses['critic'](d_y_real, d_y_fake)
        d_loss_x = self.losses['critic'](d_x_real, d_x_fake)

        # Total D loss
        d_loss = d_loss_x + d_loss_y
        return d_loss

    def g_step_calc_losses(self, batch):
        """Generators: adversarial (fool both D's) + cycle consistency + identity."""
        x, y = batch['src'], batch['tgt']
        
        x = self.models['net_proc'](x)

        include_id = (self.identity_weight > 0.0)
        outs = self.models['net']((x, y), mode='full', include_identity=include_id)

        if include_id:
            y_hat, x_hat, x_cyc, y_cyc, id_y, id_x = outs
        else:
            y_hat, x_hat, x_cyc, y_cyc = outs

        # Adversarial
        g_adv_y = self.losses['critic'](self.models['critic'](y_hat, mode='y'), None)
        g_adv_x = self.losses['critic'](self.models['critic'](x_hat, mode='x'), None)
        g_adv = (g_adv_x + g_adv_y) * self.adversarial_weight

        # Cycle-consistency (L1)
        cyc_loss = self.cycle_weight * (F.l1_loss(x_cyc, x) + F.l1_loss(y_cyc, y))

        # Identity loss (encourage G_XY(y)≈y and G_YX(x)≈x)
        id_loss = 0.0
        if include_id:
            id_loss = self.identity_weight * (F.l1_loss(id_y, y) + F.l1_loss(id_x, x))

        g_loss = g_adv + cyc_loss + id_loss
        return g_loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        batch = move_to_device(batch, self.device)
        x, y = batch['src'], batch['tgt']

        x = self.models['net_proc'](x)

        x2y = self.predict(x, 'x2y')
        y2x = self.predict(y, 'y2x')
        x_cycle = self.predict(x, 'xyx')  # x -> y_hat -> x_cyc
        y_cycle = self.predict(y, 'yxy')  # y -> x_hat -> y_cyc

        # Metrics (reuse your existing metric stack, if any)
        if batch_idx == 0:
            self._reset_metrics()
        valid_output = {'preds': torch.concat((x2y, y2x), dim=0), 'targets': torch.concat((y, x), dim=0), }
        metrics = self._calc_metrics(valid_output)

        named_imgs = {
            'source_x': x, 'target_y': y,
            'generated_yhat': x2y, 'generated_xhat': y2x,
            'cycle_xhat': x_cycle, 'cycle_yhat': y_cycle,
        }
        return metrics, named_imgs