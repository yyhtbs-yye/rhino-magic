import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from rhino.boats.gan.base_gan_boat import BaseGanBoat

@dataclass
class _KimgSchedule:
    resolutions: Tuple[int, ...]                # e.g. (4, 8, 16, 32, 64, 128)
    fade_kimg: float = 600.0                    # thousands of images to fade-in
    stable_kimg: float = 600.0                  # thousands of images to stay fully at stage after fade
    start_stage_idx: int = 0
    max_stage_idx: Optional[int] = None

    # mutable state
    stage_idx: int = 0
    alpha: float = 1.0
    in_transition: bool = False
    kimg_in_phase: float = 0.0

    def __post_init__(self):
        self.stage_idx = self.start_stage_idx
        if self.max_stage_idx is None:
            self.max_stage_idx = len(self.resolutions) - 1
        self.alpha = 0.0 if (self.stage_idx > 0) else 1.0
        self.in_transition = self.stage_idx > 0

    @property
    def resolution(self) -> int:
        return int(self.resolutions[self.stage_idx])

    def step(self, batch_size: int) -> Tuple[int, float]:
        """Advance by batch_size images (measured in kimg). Return (resolution, alpha)."""
        if self.is_done:
            return self.resolution, self.alpha

        self.kimg_in_phase += float(batch_size) / 1000.0

        if self.in_transition:
            if self.fade_kimg <= 0:
                self.alpha = 1.0
            else:
                self.alpha = max(0.0, min(1.0, self.kimg_in_phase / self.fade_kimg))
            if self.kimg_in_phase >= self.fade_kimg:
                self.in_transition = False
                self.kimg_in_phase = 0.0
        else:
            if self.kimg_in_phase >= self.stable_kimg:
                self.kimg_in_phase = 0.0
                if self.stage_idx < self.max_stage_idx:
                    self.stage_idx += 1
                    self.in_transition = True
                    self.alpha = 0.0

        return self.resolution, float(self.alpha)

    @property
    def is_done(self) -> bool:
        return (self.stage_idx >= self.max_stage_idx) and (not self.in_transition)


# ----------------------------- Epoch-based schedule (primary) -----------------------------
@dataclass(frozen=True)
class _Stage:
    res: int
    start_epoch: int
    fade_epochs: int = 0  # how many epochs to fade from previous stage into this one


class _EpochSchedule:
    """Simple, stateless mapping: (epoch -> resolution, alpha)."""
    def __init__(self, stages: List[_Stage]) -> None:
        assert len(stages) > 0, "At least one stage required."
        # enforce ordering and uniqueness of start epochs
        stages_sorted = sorted(stages, key=lambda s: int(s.start_epoch))
        unique_starts = {s.start_epoch for s in stages_sorted}
        assert len(unique_starts) == len(stages_sorted), "Duplicate start_epoch values in stages."
        self.stages = stages_sorted

    def query(self, epoch: int) -> Tuple[int, float]:
        """Return (resolution, alpha) for the given epoch.
        alpha=0..1 blends prev->current during current stage's fade window."""
        # find current stage: last with start_epoch <= epoch
        idx = 0
        for i, s in enumerate(self.stages):
            if epoch >= s.start_epoch:
                idx = i
            else:
                break

        cur = self.stages[idx]
        # compute alpha within fade window for the CURRENT stage
        if cur.fade_epochs <= 0:
            alpha = 1.0
        else:
            # epochs since this stage began
            delta = epoch - cur.start_epoch
            alpha = max(0.0, min(1.0, float(delta) / float(cur.fade_epochs)))
        return int(cur.res), float(alpha)

