# pg_gan_boat.py
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from rhino.boats.gan.base_gan_boat import BaseGanBoat
from rhino.boats.gan.helpers.pg_schedule import _EpochSchedule, _KimgSchedule, _Stage

from trainer.utils.ddp_utils import move_to_device

# -----------------------------------------------------------------------------------------
class PgGanBoat(BaseGanBoat):

    def __init__(self, config={}):
        super().__init__(config=config)

        pg_cfg: Dict = (config['boat'] or {}).get('progressive', {}) or {}

        # Prefer explicit epoch-based 'stages' if provided
        self._epoch_schedule: Optional[_EpochSchedule] = None
        self._kimg_schedule: Optional[_KimgSchedule] = None

        if 'stages' in pg_cfg and pg_cfg['stages']:
            stages: List[_Stage] = []
            for s in pg_cfg['stages']:
                stages.append(
                    _Stage(
                        res=int(s['res']),
                        start_epoch=int(s['start_epoch']),
                        fade_epochs=int(s.get('fade_epochs', 0)),
                    )
                )
            self._epoch_schedule = _EpochSchedule(stages)
        else:
            # Fallback: kimg-based progressive schedule
            resolutions = tuple(pg_cfg.get('resolutions', (4, 8, 16, 32, 64, 128)))
            fade_kimg = float(pg_cfg.get('fade_kimg', 600.0))
            stable_kimg = float(pg_cfg.get('stable_kimg', 600.0))
            start_stage_idx = int(pg_cfg.get('start_stage_idx', 0))
            max_stage_idx = pg_cfg.get('max_stage_idx', None)
            if max_stage_idx is not None:
                max_stage_idx = int(max_stage_idx)

            self._kimg_schedule = _KimgSchedule(
                resolutions=resolutions,
                fade_kimg=fade_kimg,
                stable_kimg=stable_kimg,
                start_stage_idx=start_stage_idx,
                max_stage_idx=max_stage_idx,
            )

        # live cache updated every step
        self._pg_res: int = self._current_res(epoch=0)
        self._pg_alpha: float = self._current_alpha(epoch=0)

    # ----------------------------- Training orchestration -----------------------------
    def training_step(self, batch, batch_idx, epoch, *, scaler=None):
        """Update progressive state (epoch/kimg), then defer to BaseGanBoat for stepping."""
        # Update progressive state:
        if self._epoch_schedule is not None:
            self._pg_res, self._pg_alpha = self._epoch_schedule.query(epoch=int(epoch))
        else:
            # kimg-based: advance by current batch size
            batch_size = self._infer_batch_size(batch)
            self._pg_res, self._pg_alpha = self._kimg_schedule.step(batch_size)

        # Hand off the actual stepping to the base implementation
        return super().training_step(batch, batch_idx, epoch, scaler=scaler)

    # ----------------------------- Loss calcs at current stage -----------------------------
    def d_step_calc_losses(self, batch):
        """Discriminator loss using current progressive (res, alpha)."""
        gt = batch['gt']
        batch_size = gt.size(0)
        gt = self._downscale_to(gt, self._pg_res)

        # noise from your generator (no helper)
        noise = self.noise_generator.next(batch_size, device=self.device)

        # Produce fakes with current G (no grad to G for D step)
        with torch.no_grad():
            x_fake = self.models['net'](
                noise,
                transition_weight=self._pg_alpha,
                curr_scale=self._pg_res,
            )

        # Forward D on real & fake
        d_real = self.models['critic'](
            gt, transition_weight=self._pg_alpha, curr_scale=self._pg_res
        )
        d_fake = self.models['critic'](
            x_fake, transition_weight=self._pg_alpha, curr_scale=self._pg_res
        )

        d_loss = self.losses['critic'](d_real, d_fake)

        return d_loss

    def g_step_calc_losses(self, batch):
        """Generator loss using current progressive (res, alpha)."""
        gt = batch['gt']  # only for batch_size
        batch_size = gt.size(0)

        noise = self.noise_generator.next(batch_size, device=self.device)
        x_fake = self.models['net'](
            noise, transition_weight=self._pg_alpha, curr_scale=self._pg_res,
        )

        d_fake = self.models['critic'](
            x_fake, transition_weight=self._pg_alpha, curr_scale=self._pg_res
        )

        g_loss = self.losses['critic'](d_fake, None) * getattr(self, 'adversarial_weight', 1.0)

        return g_loss

    # ----------------------------- Validation at current stage -----------------------------
    def validation_step(self, batch, batch_idx):
        """Validate at the *current* stage resolution; returns (metrics, named_imgs)."""
        # In epoch schedule mode, recompute for the provided epoch (if given).
        if (self.epoch is not None) and (self._epoch_schedule is not None):
            self._pg_res, self._pg_alpha = self._epoch_schedule.query(epoch=int(self.epoch))

        batch = move_to_device(batch, self.device)

        gt = batch['gt']
        gt = self._downscale_to(gt, self._pg_res)
        batch_size = gt.size(0)

        noise = self.noise_generator.next(batch_size, device=self.device)
        net = self.models.get('net_ema', self.models['net'])

        with torch.no_grad():
            x_fake = net(
                noise,
                transition_weight=self._pg_alpha,
                curr_scale=self._pg_res,
            )

        valid_output = {'preds': x_fake, 'targets': gt}

        # Reset metrics on first batch (if base defines these)
        metrics = {}
        try:
            if batch_idx == 0:
                self._reset_metrics()
            metrics = self._calc_metrics(valid_output)
        except Exception:
            pass

        named_imgs = {'groundtruth': gt, 'generated': x_fake}
        return metrics, named_imgs

    # ----------------------------- Utilities -----------------------------
    @staticmethod
    def _downscale_to(x: torch.Tensor, res: int) -> torch.Tensor:
        """Resize to (res x res) using bilinear for floats, nearest for ints."""
        if x.shape[-1] == res and x.shape[-2] == res:
            return x
        mode = 'nearest' if not torch.is_floating_point(x) else 'bilinear'
        return F.interpolate(
            x, size=(res, res), mode=mode, align_corners=False if mode == 'bilinear' else None
        )

    def _infer_batch_size(self, batch) -> int:
        try:
            return self._extract_images(batch).size(0)
        except Exception as e:
            raise RuntimeError(f"Unable to infer batch size from batch: {e}")

    # helpers to peek current state (useful for logs)
    def _current_res(self, epoch: int) -> int:
        if self._epoch_schedule is not None:
            r, _ = self._epoch_schedule.query(epoch=epoch)
            return r
        return self._kimg_schedule.resolution

    def _current_alpha(self, epoch: int) -> float:
        if self._epoch_schedule is not None:
            _, a = self._epoch_schedule.query(epoch=epoch)
            return a
        return float(self._kimg_schedule.alpha)
