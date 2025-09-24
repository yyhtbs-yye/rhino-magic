import torch
import torch.nn as nn

from contextlib import nullcontext
import math

from typing import Optional, Tuple
from torch.cuda.amp.grad_scaler import GradScaler

from rhino.boats.base.base_boat import BaseBoat
from trainer.utils.build_components import build_module
from torch.nn.parallel import DistributedDataParallel as DDP

def _raw(m):
    return m.module if isinstance(m, DDP) else m

def _sample_noise_for_G(net_G, batch_size, device: torch.device) -> torch.Tensor:
    """Return z ~ N(0,1) with the right dimension for StyleGAN2Generator.

    - If the (wrapped) generator has `noise_size`, use it (StyleGAN2).
    - Else, fall back to a `get_noise` method if present (other generators).
    """
    g = _raw(net_G)
    if hasattr(g, "noise_size"):
        return torch.randn(batch_size, int(g.noise_size), device=device)
    if hasattr(g, "get_noise"):
        return g.get_noise(batch_size, device)
    raise AttributeError(
        "Cannot determine noise dimension: expected `noise_size` attr (StyleGAN2) "
        "or a `get_noise(batch_size, device)` method on the generator."
    )

class RegularizedGanBoat(BaseBoat):
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

        reg = self.optimization_config.pop('reg', {})

        self.r1_interval        = int(reg.get("r1_interval", 16))
        self.r1_gamma           = float(reg.get("r1_gamma", 10.0))

        self.pl_interval        = int(reg.get("pl_interval", 4))
        self.pl_weight          = float(reg.get("pl_weight", 2.0))
        self.pl_batch_shrink    = int(reg.get("pl_batch_shrink", 2))  # smaller batch for PL
        self.pl_beta            = float(reg.get("pl_beta", 0.99))     # running mean decay

        self.pl_mean = torch.zeros([])
        # self.register_buffer("pl_mean", )           # scalar EMA target

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

        net_G = self.models['net']            # generator
        net_D = self.models['critic']     # discriminator
        d_opt = self.optimizers['critic']

        x_real = batch['gt']

        batch_size = x_real.shape[0]

        z_noise = _sample_noise_for_G(net_G, batch_size, x_real.device)

        # Zero grads only at the start of a new accumulation window
        if args['start_new_accum']:
            d_opt.zero_grad(set_to_none=True)

        # Freeze G, enable D
        net_G.requires_grad_(False)
        net_D.requires_grad_(True)

        # Generate fake samples with current G (no grad to G)
        with torch.no_grad():
            x_fake = _raw(net_G)(z_noise)

        # Forward D on real & fake (under autocast if enabled)
        # with autocast_ctx():

        len_real = x_real.shape[0]
        len_fake = x_fake.shape[0]
        x_cat = torch.concat([x_real, x_fake.detach()], dim=0)
        d_cat = net_D(x_cat)

        d_real, d_fake = torch.split(d_cat, [len_real, len_fake], dim=0)

        d_loss = self.losses['critic'](d_real, d_fake) # GPT suggest remove "* self.adversarial_weight"

        r1_loss = None

        if self.get_global_step() % self.r1_interval == 0:
            r1_loss = r1_gradient_penalty_loss(net_D, x_real)
            d_loss = d_loss + self.r1_gamma * r1_loss

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


        return d_loss, r1_loss

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

        z_noise = _sample_noise_for_G(net_G, batch_size, x_real.device)

        # Recompute fakes WITH grad through G
        # with autocast_ctx():
        x_fake = net_G(z_noise)
        d_fake_for_g = _raw(net_D)(x_fake)
        # Convention: loss_fn(pred_fake, None) gives G's adv loss (hinge/BCE etc.)
        g_loss = self.losses['critic'](d_fake_for_g, None) * self.adversarial_weight

        pl_loss = None

        if self.get_global_step() % self.pl_interval == 0:
            pl_loss, self.pl_mean, _ =  gen_path_regularizer(net_G, batch_size, self.pl_mean)
            g_loss = g_loss + pl_loss * self.pl_weight
        
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


        return g_loss, pl_loss

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

        g_loss, d_loss, r1_loss, pl_loss = None, None, None, None

        # AMP context if scaler is provided
        autocast_ctx = torch.cuda.amp.autocast if scaler is not None else nullcontext

        g_loss, d_loss = None, None

        if do_d:
            d_loss, r1_loss = self.d_step(batch, start_new_accum=start_new_accum, scaler=scaler, 
                                 loss_scale=loss_scale, should_step_now=should_step_now)
            
        if do_g:
            g_loss, pl_loss = self.g_step(batch, start_new_accum=start_new_accum, scaler=scaler, 
                                 loss_scale=loss_scale, should_step_now=should_step_now)

            # Optional EMA just after a real G step
            if getattr(self, 'use_ema', False) and self.get_global_step() >= getattr(self, 'ema_start', 0):
                self._update_ema()

        self.end_step()

        total_loss = None
        for x in (g_loss, d_loss, r1_loss, pl_loss):
            if x is not None:
                total_loss = x if total_loss is None else total_loss + x

        if total_loss is None:
            raise RuntimeError("No loss computed in training_step.")

        return {
            "total_loss": total_loss,
            "g_loss": g_loss,
            "d_loss": d_loss,
            "r1_loss": r1_loss,
            "pl_loss": pl_loss,
            "did_step": should_step_now,
        }
    
    
    def validation_step(self, batch, batch_idx):

        x_real = batch['gt']

        batch_size = x_real.shape[0]

        net_G = self.models['net']

        with torch.no_grad():

            z_noise = _sample_noise_for_G(net_G, batch_size, x_real.device)

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

def r1_gradient_penalty_loss(discriminator: nn.Module,
                             real_data: torch.Tensor,
                             mask: Optional[torch.Tensor] = None,
                             norm_mode: str = 'pixel',
                             loss_scaler: Optional[GradScaler] = None) -> torch.Tensor:
    """Calculate R1 gradient penalty for WGAN-GP.

    R1 regularizer comes from:
    "Which Training Methods for GANs do actually Converge?" ICML'2018

    Different from original gradient penalty, this regularizer only penalized
    gradient w.r.t. real data.

    Args:
        discriminator (nn.Module): Network for the discriminator.
        real_data (Tensor): Real input data.
        mask (Tensor): Masks for inpainting. Default: None.
        norm_mode (str): This argument decides along which dimension the norm
            of the gradients will be calculated. Currently, we support ["pixel"
            , "HWC"]. Defaults to "pixel".

    Returns:
        Tensor: A tensor for gradient penalty.
    """
    batch_size = real_data.shape[0]

    real_data = real_data.clone().requires_grad_()

    disc_pred = discriminator(real_data)
    if loss_scaler:
        disc_pred = loss_scaler.scale(disc_pred)

    gradients = torch.autograd.grad(
        outputs=disc_pred,
        inputs=real_data,
        grad_outputs=torch.ones_like(disc_pred),
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    if loss_scaler:
        # unscale the gradient
        inv_scale = 1. / loss_scaler.get_scale()
        gradients = gradients * inv_scale

    if mask is not None:
        gradients = gradients * mask

    if norm_mode == 'pixel':
        gradients_penalty = ((gradients.norm(2, dim=1))**2).mean()
    elif norm_mode == 'HWC':
        gradients_penalty = gradients.pow(2).reshape(batch_size,
                                                     -1).sum(1).mean()
    else:
        raise NotImplementedError(
            'Currently, we only support ["pixel", "HWC"] '
            f'norm mode but got {norm_mode}.')
    if mask is not None:
        gradients_penalty /= torch.mean(mask)

    return gradients_penalty


def gen_path_regularizer(generator: nn.Module,
                         num_batches: int,
                         mean_path_length: torch.Tensor,
                         pl_batch_shrink: int = 1,
                         decay: float = 0.01,
                         weight: float = 1.,
                         pl_batch_size: Optional[int] = None,
                         loss_scaler: Optional[GradScaler] = None,
                         ) -> Tuple[torch.Tensor]:
    """Generator Path Regularization.

    Path regularization is proposed in StyleGAN2, which can help the improve
    the continuity of the latent space. More details can be found in:
    Analyzing and Improving the Image Quality of StyleGAN, CVPR2020.

    Args:
        generator (nn.Module): The generator module. Note that this loss
            requires that the generator contains ``return_latents`` interface,
            with which we can get the latent code of the current sample.
        num_batches (int): The number of samples used in calculating this loss.
        mean_path_length (Tensor): The mean path length, calculated by moving
            average.
        pl_batch_shrink (int, optional): The factor of shrinking the batch size
            for saving GPU memory. Defaults to 1.
        decay (float, optional): Decay for moving average of mean path length.
            Defaults to 0.01.
        weight (float, optional): Weight of this loss item. Defaults to ``1.``.
        pl_batch_size (int | None, optional): The batch size in calculating
            generator path. Once this argument is set, the ``num_batches`` will
            be overridden with this argument and won't be affected by
            ``pl_batch_shrink``. Defaults to None.
        sync_mean_buffer (bool, optional): Whether to sync mean path length
            across all of GPUs. Defaults to False.

    Returns:
        tuple[Tensor]: The penalty loss, detached mean path tensor, and \
            current path length.
    """
    # reduce batch size for conserving GPU memory
    if pl_batch_shrink > 1:
        num_batches = max(1, num_batches // pl_batch_shrink)

    # reset the batch size if pl_batch_size is not None
    if pl_batch_size is not None:
        num_batches = pl_batch_size

    # get output from different generators
    output_dict = _raw(generator)(None, num_batches=num_batches, return_latents=True)
    fake_img, latents = output_dict['fake_img'], output_dict['latent']

    noise = torch.randn_like(fake_img) / math.sqrt(fake_img.shape[2] * fake_img.shape[3])

    if loss_scaler:
        loss = loss_scaler.scale((fake_img * noise).sum())[0]
        grad = torch.autograd.grad(
            outputs=loss,
            inputs=latents,
            grad_outputs=torch.ones(()).to(loss),
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        # unscale the grad
        inv_scale = 1. / loss_scaler.get_scale()
        grad = grad * inv_scale
    else:
        grad = torch.autograd.grad(
            outputs=(fake_img * noise).sum(),
            inputs=latents,
            grad_outputs=torch.ones(()).to(fake_img),
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))
    # update mean path
    path_mean = mean_path_length + decay * (
        path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean() * weight

    return path_penalty, path_mean.detach(), path_lengths