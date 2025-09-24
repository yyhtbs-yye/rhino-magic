import torch
import torch.nn.functional as F
from contextlib import nullcontext
from torch.nn.parallel import DistributedDataParallel as DDP

from rhino.boats.base.base_boat import BaseBoat
from trainer.utils.build_components import build_module


def _raw(m):
    return m.module if isinstance(m, DDP) else m


class PgGanBoat(BaseBoat):
    """
    PGGAN training boat (compatible with your PGGANGenerator/PGGANDiscriminator).

    Key compatibility points:
      - Calls G/D with (curr_scale=self.current_res, transition_weight=self.alpha).
      - Samples 2-D noise [N, noise_size] using G.noise_size.
      - Supports PGGAN fade-in schedule via boat_config['pggan']['stages'].
      - Separates d_step() and g_step() for clarity and custom schedulers.
      - Still tolerant of alt APIs (stage_idx/alpha, set_stage/set_alpha) if present.

    Expected loss contract:
      - D loss: losses['critic'](d_real, d_fake)
      - G loss: losses['critic'](d_fake_for_g, None)

    Optional:
      - latent_encoder: if provided, we treat G output as latents and decode.
      - EMA: if enabled in optimization_config.
    """

    # -----------------------------
    # Init / wiring
    # -----------------------------
    def __init__(self, boat_config=None, optimization_config=None, validation_config=None):
        super().__init__()
        assert boat_config is not None, "boat_config must be provided"

        # Build models
        self.models['net'] = build_module(boat_config['net'])              # Generator (G)
        self.models['critic'] = build_module(boat_config['critic'])        # Discriminator (D)
        self.models['latent_encoder'] = build_module(boat_config['latent_encoder']) if 'latent_encoder' in boat_config else None

        if self.models['latent_encoder'] is not None:
            for p in self.models['latent_encoder'].parameters():
                p.requires_grad = False

        # Store configs
        self.boat_config = boat_config
        self.optimization_config = optimization_config or {}
        self.validation_config = validation_config or {}

        # Global defaults (can be overridden per-stage)
        self.concurrent = bool(self.optimization_config.get('concurrent', False))
        self.g_interval_default = int(self.optimization_config.get('g_interval', 1))
        self.d_interval_default = int(self.optimization_config.get('d_interval', 1))
        self.adversarial_weight = float(self.optimization_config.get('adversarial_weight', 1.0))

        # Active intervals (possibly overridden by stage)
        self.g_interval = self.g_interval_default
        self.d_interval = self.d_interval_default

        self.use_ema = self.optimization_config.get('use_ema', False)
        self.use_reference = self.validation_config.get('use_reference', False)
        if self.use_ema:
            self._setup_ema()
            self.ema_start = int(self.optimization_config.get('ema_start', 0))
        else:
            self.ema_start = 0

        # PGGAN schedule
        pggan_cfg = (boat_config.get('pggan') or {})
        assert 'stages' in pggan_cfg and len(pggan_cfg['stages']) > 0, \
            "Provide boat_config['pggan']['stages'] with res/start_epoch(/fade_epochs)."

        self.pggan_stages = []
        for s in pggan_cfg['stages']:
            self.pggan_stages.append({
                'res': int(s['res']),
                'start_epoch': int(s['start_epoch']),
                'fade_epochs': int(s.get('fade_epochs', 0)),
                'g_interval': int(s.get('g_interval', self.g_interval_default)),
                'd_interval': int(s.get('d_interval', self.d_interval_default)),
            })
        self.pggan_stages.sort(key=lambda s: s['start_epoch'])
        assert self.pggan_stages[0]['start_epoch'] == 0, "First stage must start at epoch 0."
        for i in range(1, len(self.pggan_stages)):
            assert self.pggan_stages[i]['start_epoch'] > self.pggan_stages[i-1]['start_epoch'], \
                "Stage start epochs must be strictly increasing."

        # Active stage state
        self.current_stage_idx = 0
        self.current_res = self.pggan_stages[0]['res']
        self.alpha = 1.0
        self.in_transition = False

        # Apply to nets (best-effort for any extra APIs)
        self._apply_stage_to_models()

    # -----------------------------
    # Stage helpers
    # -----------------------------
    def _stage_index_for_epoch(self, epoch: int) -> int:
        idx = 0
        for i, s in enumerate(self.pggan_stages):
            if epoch >= s['start_epoch']:
                idx = i
            else:
                break
        return idx

    def _alpha_for_epoch(self, epoch: int, stage_idx: int) -> float:
        if stage_idx == 0:
            return 1.0
        s = self.pggan_stages[stage_idx]
        fade = max(0, s.get('fade_epochs', 0))
        if fade == 0:
            return 1.0
        t = (epoch - s['start_epoch']) / float(fade)
        return float(max(0.0, min(1.0, t)))

    def _ensure_stage(self, epoch: int):
        new_idx = self._stage_index_for_epoch(epoch)
        new_alpha = self._alpha_for_epoch(epoch, new_idx)
        changed = (new_idx != self.current_stage_idx)

        self.current_stage_idx = new_idx
        self.current_res = self.pggan_stages[new_idx]['res']
        self.alpha = new_alpha
        self.in_transition = (new_idx > 0) and (new_alpha < 1.0)

        # per-stage overrides
        self.g_interval = self.pggan_stages[new_idx].get('g_interval', self.g_interval_default)
        self.d_interval = self.pggan_stages[new_idx].get('d_interval', self.d_interval_default)

        if changed:
            self._apply_stage_to_models()

        self._apply_alpha_to_models()  # keep pushing alpha each call

    def _apply_stage_to_models(self):
        res = self.current_res
        idx = self.current_stage_idx
        alpha = self.alpha
        for key in ('net', 'critic'):
            net = self.models.get(key)
            if net is None:
                continue
            raw = _raw(net)
            if hasattr(raw, 'set_stage'):
                try:
                    raw.set_stage(idx, res, alpha)
                except TypeError:
                    try:
                        raw.set_stage(idx, res)
                    except Exception:
                        pass
            elif hasattr(raw, 'set_resolution'):
                try:
                    raw.set_resolution(res)
                except Exception:
                    pass

    def _apply_alpha_to_models(self):
        for key in ('net', 'critic'):
            net = self.models.get(key)
            if net is None:
                continue
            raw = _raw(net)
            if hasattr(raw, 'set_alpha'):
                try:
                    raw.set_alpha(self.alpha)
                except Exception:
                    pass

    # -----------------------------
    # Core call adapters (compat layer)
    # -----------------------------
    def _call_G(self, z):
        net = self.models['net']
        # Prefer PGGAN kwargs, fall back to older (stage_idx/alpha), then plain
        try:
            return net(z, curr_scale=self.current_res, transition_weight=self.alpha)
        except TypeError:
            try:
                return net(z, stage_idx=self.current_stage_idx, alpha=self.alpha)
            except TypeError:
                return net(z)

    def _call_D(self, x):
        critic = self.models['critic']
        try:
            return critic(x, curr_scale=self.current_res, transition_weight=self.alpha)
        except TypeError:
            try:
                return critic(x, stage_idx=self.current_stage_idx, alpha=self.alpha)
            except TypeError:
                return critic(x)

    def _sample_noise(self, batch_size: int, device):
        """2-D noise for PGGAN G: [N, noise_size]."""
        g_raw = _raw(self.models['net'])
        n = getattr(g_raw, 'noise_size', None)
        if isinstance(n, int) and n > 0:
            return torch.randn(batch_size, n, device=device)
        # Fallback (shouldn't be needed with your PGGAN):
        return torch.randn(batch_size, 512, device=device)

    # -----------------------------
    # Public forward (uses EMA if enabled)
    # -----------------------------
    def forward(self, z, *, epoch: int = None):
        if epoch is not None:
            self._ensure_stage(int(epoch))
        use_ema = self.use_ema and 'net_ema' in self.models
        return self._call_G(z, use_ema=use_ema)

    # -----------------------------
    # One step of D (separate)
    # -----------------------------
    def d_step(self, x_real, **args): # start_new_accum, scaler, loss_scale, should_step_now):

        d_opt = self.optimizers['critic']   # 

        batch_size = x_real.shape[0]

        z_noise = self._sample_noise(batch_size, x_real.device)

        # Zero grads only at the start of a new accumulation window
        if args['start_new_accum']:
            d_opt.zero_grad(set_to_none=True)

        # Freeze G, enable D
        self.models['net'].requires_grad_(False)
        self.models['critic'].requires_grad_(True)

        # Generate fake samples with current G (no grad to G)
        with torch.no_grad():
            x_fake = self._call_G(z_noise)

        # Forward D on real & fake (under autocast if enabled)
        with args['autocast_ctx']():
            d_real, d_fake = self._call_D(x_real), self._call_D(x_fake)
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

    # -----------------------------
    # One step of G (separate)
    # -----------------------------
    def g_step(self, x_real, **args):

        g_opt = self.optimizers['net']

        batch_size = x_real.shape[0]

        if args['start_new_accum']:
            g_opt.zero_grad(set_to_none=True)

        # Enable G, freeze D so G doesn't update D
        self.models['net'].requires_grad_(True)
        self.models['critic'].requires_grad_(False)

        z_noise = self._sample_noise(batch_size, x_real.device)

        # Recompute fakes WITH grad through G
        with args['autocast_ctx']():
            x_fake = self._call_G(z_noise)
            d_fake_for_g = self._call_D(x_fake)
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

    # -----------------------------
    # Training (orchestrates g_step/d_step)
    # -----------------------------
    def training_step(self, batch, batch_idx, epoch, *, 
                      scaler=None, accumulate=1, microstep=0,):
        """
        Orchestrates PGGAN training:
          - Downscale real batch to current resolution.
          - Run D step at d_interval, then G step at g_interval.
        """
        # Stage/alpha for this step
        self._ensure_stage(int(epoch))

        # Prepare real images at current resolution
        x_real_full = batch['gt']
        x_real = self._downscale_to_stage(x_real_full, self.current_res)

        # AMP autocast (disabled if scaler is None)
        autocast_ctx = torch.cuda.amp.autocast if scaler is not None else nullcontext

        # Decide whether to trigger each optimizer at this batch_idx
        do_g = (self.g_interval > 0) and (batch_idx % self.g_interval == 0)
        do_d = (self.d_interval > 0) and (batch_idx % self.d_interval == 0)

        # Accumulation controls
        start_new_accum = (microstep % accumulate == 0)
        should_step_now = ((microstep + 1) % accumulate == 0)

        g_loss = None
        d_loss = None

        # Update D first (standard GAN)
        if do_d:
            d_loss = self.d_step(
                x_real,
                start_new_accum=start_new_accum,
                scaler=scaler,
                accumulate=accumulate,
                microstep=microstep,
                should_step_now=should_step_now,
                autocast_ctx=autocast_ctx,
            )

        # Then update G
        if do_g:
            g_loss = self.g_step(
                x_real,
                start_new_accum=start_new_accum,
                scaler=scaler,
                accumulate=accumulate,
                microstep=microstep,
                should_step_now=should_step_now,
                autocast_ctx=autocast_ctx,
            )

        # Restore grads
        self.models['net'].requires_grad_(True)
        self.models['critic'].requires_grad_(True)

        # Total loss for logging
        total_loss = None
        if g_loss is not None and d_loss is not None:
            total_loss = g_loss + d_loss
        elif g_loss is not None:
            total_loss = g_loss
        elif d_loss is not None:
            total_loss = d_loss
        else:
            total_loss = torch.tensor(0.0, device=x_real.device)

        return {
            "total_loss": total_loss,
            "g_loss": g_loss,
            "d_loss": d_loss,
            "did_step": should_step_now,
            "stage_idx": torch.tensor(self.current_stage_idx, device=total_loss.device),
            "stage_res": torch.tensor(self.current_res, device=total_loss.device),
            "alpha": torch.tensor(self.alpha, device=total_loss.device),
        }

    # -----------------------------
    # Validation
    # -----------------------------
    def validation_step(self, batch, batch_idx, *, epoch: int = None):
        if epoch is not None:
            self._ensure_stage(int(epoch))

        x_full = batch['gt']
        x = self._downscale_to_stage(x_full, self.current_res)

        with torch.no_grad():
            if self.models.get('latent_encoder') is not None:
                z_like = self.encode_images(x)
                z = torch.randn_like(z_like)  # keep latent shape if you chain through encoder
                preds = self.forward(z)  # EMA if available
                preds = self.decode_latents(preds)
            else:
                z = self._sample_noise(x.shape[0], x.device)
                preds = self.forward(z)

            valid_output = {'preds': preds, 'targets': x}

            if batch_idx == 0:
                self._reset_metrics()
            metrics = self._calc_metrics(valid_output)
            named_imgs = {'groundtruth': x, 'generated': preds}

        return metrics, named_imgs

    # -----------------------------
    # Latent encoder helpers (optional)
    # -----------------------------
    def decode_latents(self, z):
        return self.models['latent_encoder'].decode(z)

    def encode_images(self, x):
        return self.models['latent_encoder'].encode(x)

    # -----------------------------
    # Utils
    # -----------------------------
    @staticmethod
    def _downscale_to_stage(x: torch.Tensor, res: int) -> torch.Tensor:
        if x.shape[-1] == res and x.shape[-2] == res:
            return x
        return F.interpolate(x, size=(res, res), mode='bilinear', align_corners=False)

    # These remain unused but kept for BaseBoat completeness
    def training_calc_losses(self, batch, batch_idx): pass
    def training_backpropagation(self, losses, batch_idx, accumulate, scaler, active_keys): pass
    def training_gradient_descent(self, batch_idx, scaler, active_keys): pass
