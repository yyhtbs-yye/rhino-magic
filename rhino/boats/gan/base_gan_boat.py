import torch
from contextlib import nullcontext

from rhino.boats.base.base_boat import BaseBoat
from trainer.utils.build_components import build_module
from torch.nn.parallel import DistributedDataParallel as DDP

def _raw(m):
    return m.module if isinstance(m, DDP) else m

class BaseGanBoat(BaseBoat):
    def __init__(self, boat_config=None, optimization_config=None, validation_config=None):
        super().__init__()
        
        assert boat_config is not None, "boat_config must be provided"

        # Build the model
        self.models['net'] = build_module(boat_config['net'])
        self.models['critic'] = build_module(boat_config['critic'])

        # Store configurations
        self.boat_config = boat_config
        self.optimization_config = optimization_config or {}
        self.validation_config = validation_config or {}

        self.concurrent = bool(self.optimization_config.get('concurrent', False))
        self.g_interval = int(self.optimization_config.get('g_interval', 1))
        self.d_interval = int(self.optimization_config.get('d_interval', 1))
        self.adversarial_weight = float(self.optimization_config.get('adversarial_weight', 0.01))

        self.use_ema = self.optimization_config.get('use_ema', False)
        self.use_reference = validation_config.get('use_reference', False)

        # Setup EMA if enabled
        if self.use_ema:
            self._setup_ema()
            self.ema_start = self.optimization_config.get('ema_start', 0)
        else:
            self.ema_start = 0

    def forward(self, x):
        
        network_in_use = (
            self.models['net_ema'] 
            if self.use_ema and 'net_ema' in self.models 
            else self.models['net']
        )       
        return network_in_use(x)

    def d_step(self, batch, **args): # start_new_accum, scaler, loss_scale, should_step_now):

        net_G = self.models['net']          # generator
        net_D = self.models['critic']       # discriminator
        d_opt = self.optimizers['critic']   # 

        x_real = batch['gt']

        batch_size = x_real.shape[0]

        z_noise = _raw(net_G).get_noise(batch_size, x_real.device)

        # Zero grads only at the start of a new accumulation window
        if args['start_new_accum']:
            d_opt.zero_grad(set_to_none=True)

        # Freeze G, enable D
        net_G.requires_grad_(False)
        net_D.requires_grad_(True)

        # Generate fake samples with current G (no grad to G)
        with torch.no_grad():
            x_fake = net_G(z_noise)

        # Forward D on real & fake (under autocast if enabled)
        with args['autocast_ctx']():
            d_real, d_fake = net_D(x_real), net_D(x_fake)
            d_loss = self.losses['critic'](d_real, d_fake) # GPT suggest remove "* self.adversarial_weight"

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

        net_G = self.models['net']            # generator
        net_D = self.models['critic']     # discriminator
        g_opt = self.optimizers['net']

        x_real = batch['gt']

        batch_size = x_real.shape[0]

        if args['start_new_accum']:
            g_opt.zero_grad(set_to_none=True)

        # Enable G, freeze D so G doesn't update D
        net_G.requires_grad_(True)
        net_D.requires_grad_(False)

        z_noise = _raw(net_G).get_noise(batch_size, x_real.device)

        # Recompute fakes WITH grad through G
        with args['autocast_ctx']():
            x_fake = net_G(z_noise)
            d_fake_for_g = net_D(x_fake)
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
        net_G = self.models['net']            # generator
        net_D = self.models['critic']     # discriminator
        g_opt = self.optimizers['net']
        d_opt = self.optimizers['critic']
        if g_opt is None or d_opt is None:
            raise RuntimeError("Expected 'net' (G) and 'critic' (D) optimizers in self.optimizers.")
        # Restore grads by default
        net_G.requires_grad_(True)
        net_D.requires_grad_(True)

    def training_step(self, batch, batch_idx, epoch, *, 
                      scaler=None, accumulate=1, microstep=0,):

        # Schedules
        do_g = (self.g_interval > 0) and (batch_idx % self.g_interval == 0)
        do_d = (self.d_interval > 0) and (batch_idx % self.d_interval == 0)

        # Accumulation controls
        start_new_accum = (microstep % accumulate == 0)
        should_step_now = ((microstep + 1) % accumulate == 0)
        loss_scale = 1.0 / float(accumulate)

        # AMP context if scaler is provided
        autocast_ctx = torch.cuda.amp.autocast if scaler is not None else nullcontext

        g_loss, d_loss = None, None

        if do_d:
            d_loss = self.d_step(batch, start_new_accum=start_new_accum, scaler=scaler, loss_scale=loss_scale, should_step_now=should_step_now,
                                 autocast_ctx=autocast_ctx)

        if do_g:
            g_loss = self.g_step(batch, start_new_accum=start_new_accum, scaler=scaler, loss_scale=loss_scale, should_step_now=should_step_now,
                                 autocast_ctx=autocast_ctx)

            # Optional EMA just after a real G step
            if getattr(self, 'use_ema', False) and self.get_global_step() >= getattr(self, 'ema_start', 0):
                self._update_ema()

        self.end_step()

        total_loss = None
        if g_loss is not None and d_loss is not None:
            total_loss = g_loss + d_loss
        elif g_loss is not None:
            total_loss = g_loss
        elif d_loss is not None:
            total_loss = d_loss
        else:
            raise RuntimeError("No generator or discriminator loss was computed in training_step.")

        return {
            "total_loss": total_loss,
            "g_loss": g_loss,
            "d_loss": d_loss,
            "did_step": should_step_now,
        }

    def validation_step(self, batch, batch_idx):

        x_real = batch['gt']

        batch_size = x_real.shape[0]

        net_G = self.models['net']

        with torch.no_grad():

            z_noise = _raw(net_G).get_noise(batch_size, x_real.device)

            x_fake = self.forward(z_noise)

            valid_output = {'preds': x_fake, 'targets': x_real}

            # Reset Metric in the begining iter in an epoch
            if batch_idx == 0:
                self._reset_metrics()

            metrics = self._calc_metrics(valid_output)

            named_imgs = {'groundtruth': x_real, 'generated': x_fake,}

        return metrics, named_imgs

    def training_calc_losses(self, batch, batch_idx, ): pass

    def training_backpropagation(self, losses, batch_idx, accumulate, scaler): pass

    def training_gradient_descent(self, batch_idx): pass
