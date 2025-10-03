# pix2pix_boat.py
import torch
import torch.nn.functional as F

from rhino.boats.gan.base_gan_boat import BaseGanBoat
from trainer.utils.ddp_utils import move_to_device
from trainer.utils.build_components import build_module

class Pix2PixBoat(BaseGanBoat):
    """
    Clean, minimal Pix2Pix boat built on top of BaseGanBoat.
    Assumptions (kept simple and consistent):
      - Batch contains two tensors: batch['src'] (source/condition) and batch['tgt'] (target)
      - Generator: net(x) -> y_hat
      - Critic: critic(x, y_or_yhat) -> adversarial score/logit tensor
      - Losses:
          self.losses['critic'](real, fake)   # D step
          self.losses['critic'](fake, None)   # G adversarial part
        Plus an L1 reconstruction term with weight self.l1_weight.
      - Shapes/dims are correct; no adaptation logic.
    """

    def __init__(self, config={}):
        super().__init__(config=config)

        self.net_proc = build_module(boat_config['net_proc'])

        # Pix2Pix L1 weight (lambda_L1 in the paper)
        p2p_cfg = (boat_config or {}).get('pix2pix', {})
        self.l1_weight = float(p2p_cfg.get('l1_weight', 10.0))

    @torch.no_grad()
    def predict(self, x: torch.Tensor, c) -> torch.Tensor:
        net = self.models['net_ema'] if self.use_ema and 'net_ema' in self.models else self.models['net']
        return net(x, c)

    def d_step_calc_losses(self, batch):
        x, y = batch['src'], batch['tgt']

        batch_size = x.size(0)

        noise = self.noise_generator.next(batch_size, device=self.device)

        x = self.net_proc(x)

        # Generate fake target (stop grad through G)
        with torch.no_grad():
            y_hat = self.models['net'](noise, x)

        # Critic on real pair and fake pair
        d_real = self.models['critic'](torch.concat((x, y), dim=1))
        d_fake = self.models['critic'](torch.concat((x, y_hat), dim=1))

        # Standard adversarial D loss
        d_loss = self.losses['critic'](d_real, d_fake)
        return d_loss

    def g_step_calc_losses(self, batch):
        x, y = batch['src'], batch['tgt']

        batch_size = x.size(0)

        noise = self.noise_generator.next(batch_size, device=self.device)

        x = self.net_proc(x)

        # Generate fake target (G requires grad)
        y_hat = self.models['net'](noise, x)

        # Adversarial term: fool the critic on (x, y_hat)
        d_fake_for_g = self.models['critic'](torch.concat((x, y_hat), dim=1))
        g_adv = self.losses['critic'](d_fake_for_g, None) * self.adversarial_weight

        # L1 reconstruction term
        g_l1 = F.l1_loss(y_hat, y) * self.l1_weight

        # Total G loss
        g_loss = g_adv + g_l1
        return g_loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        batch = move_to_device(batch, self.device)
        x, y = batch['src'], batch['tgt']

        batch_size = x.size(0)

        noise = self.noise_generator.next(batch_size, device=self.device)

        x = self.net_proc(x)

        y_hat = self.predict(noise, x)

        # Reset metrics at the start of an epoch
        if batch_idx == 0:
            self._reset_metrics()

        # Reuse your metric stack (adapt keys to your metric implementations)
        valid_output = {'preds': y_hat, 'targets': y}
        metrics = self._calc_metrics(valid_output)

        named_imgs = {'source_x': x, 'target_y': y, 'generated_yhat': y_hat}
        return metrics, named_imgs
