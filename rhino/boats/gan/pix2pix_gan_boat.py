import torch
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel as DDP
from rhino.boats.gan.base_gan_boat import BaseGanBoat
def _raw(m):
    return m.module if isinstance(m, DDP) else m

class Pix2PixGanBoat(BaseGanBoat):

    def __init__(self, boat_config=None, optimization_config=None, validation_config=None):
        super().__init__(boat_config, optimization_config, validation_config)

    # -------------------------
    # Public API
    # -------------------------
    def forward(self, z_noise, cond=None):
        network_in_use = (
            self.models['net_ema']
            if self.use_ema and 'net_ema' in self.models
            else self.models['net']
        )

        return _raw(network_in_use)(z_noise, cond)

    def d_step(self, batch, **args):  # start_new_accum, scaler, loss_scale, should_step_now

        net_G = self.models['net']        # generator
        net_D = self.models['critic']     # discriminator
        d_opt = self.optimizers['critic']

        x_real = batch['gt']
        cond = batch['cond']

        z_noise = torch.zeros_like(x_real)

        # Zero grads only at the start of a new accumulation window
        if args['start_new_accum']:
            d_opt.zero_grad(set_to_none=True)

        # Freeze G, enable D
        net_G.requires_grad_(False)
        net_D.requires_grad_(True)

        cond = self._maybe_resize_like(cond, z_noise)

        # Generate fake samples with current G (no grad to G)
        with torch.no_grad():
            x_fake = _raw(net_G)(z_noise, cond)

        # Forward D on real & fake
        len_real = x_real.shape[0]
        len_fake = x_fake.shape[0]

        bundle_real = torch.cat([x_real, cond], dim=1)
        bundle_fake = torch.cat([x_fake, cond], dim=1)
        bundle_cat = torch.cat([bundle_real, bundle_fake], dim=0)

        d_cat = net_D(bundle_cat)

        d_real, d_fake = torch.split(d_cat, [len_real, len_fake], dim=0)

        d_loss = self.losses['critic'](d_real, d_fake)

        # Backward (scaled for accumulation)
        if args['scaler'] is not None:
            args['scaler'].scale(d_loss * args['loss_scale']).backward()
        else:
            (d_loss * args['loss_scale']).backward()

        # Step D only at the end of the accumulation window
        if args['should_step_now']:
            if args['scaler'] is not None:
                args['scaler'].step(d_opt)
                args['scaler'].update()
            else:
                d_opt.step()

        return d_loss

    def g_step(self, batch, **args):
        net_G = self.models['net']        # generator
        net_D = self.models['critic']     # discriminator
        g_opt = self.optimizers['net']

        x_real = batch['gt']
        cond = batch['cond']

        z_noise = torch.zeros_like(x_real)

        if args['start_new_accum']:
            g_opt.zero_grad(set_to_none=True)

        # Enable G, freeze D so G doesn't update D
        net_G.requires_grad_(True)
        net_D.requires_grad_(False)

        cond = self._maybe_resize_like(cond, z_noise)

        # Recompute fakes WITH grad through G
        x_fake = net_G(z_noise, cond)

        bundle_fake = torch.cat([x_fake, cond], dim=1)

        d_fake_for_g = _raw(net_D)(bundle_fake)

        # Convention: loss_fn(pred_fake, None) gives G's adv loss (hinge/BCE etc.)
        g_loss = self.losses['critic'](d_fake_for_g, None) * self.adversarial_weight

        # Backward (scaled for accumulation)
        if args['scaler'] is not None:
            args['scaler'].scale(g_loss * args['loss_scale']).backward()
        else:
            (g_loss * args['loss_scale']).backward()

        # Step G only at the end of the accumulation window
        if args['should_step_now']:
            if args['scaler'] is not None:
                args['scaler'].step(g_opt)
                args['scaler'].update()
            else:
                g_opt.step()

        return g_loss

    def end_step(self):
        net_G = self.models['net']              # generator
        net_D = self.models['critic']           # discriminator
        g_opt = self.optimizers['net']
        d_opt = self.optimizers['critic']
        if g_opt is None or d_opt is None:
            raise RuntimeError("Expected 'net' (G) and 'critic' (D) optimizers in self.optimizers.")
        # Restore grads by default
        net_G.requires_grad_(True)
        net_D.requires_grad_(True)

    def validation_step(self, batch, batch_idx):
        x_real = batch['gt']
        cond = batch['cond']
        net_G = self.models['net']        # generator

        batch_size = x_real.shape[0]

        z_noise = torch.zeros_like(x_real)

        cond = self._maybe_resize_like(cond, z_noise)

        with torch.no_grad():
            # Use EMA net if present via self.forward
            x_fake = self.forward(z_noise, cond)

            valid_output = {'preds': x_fake, 'targets': x_real}

            # Reset Metric in the beginning iter in an epoch
            if batch_idx == 0:
                self._reset_metrics()

            metrics = self._calc_metrics(valid_output)

            named_imgs = {
                'condition': cond,
                'groundtruth': x_real,
                'generated': x_fake,
            }

        return metrics, named_imgs

    def training_calc_losses(self, batch, batch_idx): pass
    def training_backpropagation(self, losses, batch_idx, accumulate, scaler): pass
    def training_gradient_descent(self, batch_idx): pass

    @staticmethod
    def _maybe_resize_like(src, ref):
        """Resize src to match ref spatial size if needed."""
        if src is None:
            return None
        if src.shape[-2:] != ref.shape[-2:]:
            return F.interpolate(src, size=ref.shape[-2:], mode='bilinear', align_corners=False)
        return src
