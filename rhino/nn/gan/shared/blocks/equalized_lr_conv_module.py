import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union, Dict, Any

from rhino.nn.gan.shared.utils.equalized_lr import EqualizedLR

class EqualizedLRModule(nn.Module):
    # --- zero/low config knobs (override in subclasses only if needed) ---
    ELR_ENABLED: bool = True
    ELR_TYPES = (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)  # which modules to auto-wrap
    ELR_PARAM: str = "weight"
    ELR_GAIN: float = 2**0.5
    ELR_MODE: str = "fan_in"
    ELR_LR_MUL: float = 1.0
    ELR_INIT: str = "pggan"          # "pggan" or "none"
    EXCLUDE_NAMES = set()            # e.g., {"head"} to skip by attribute name
    EXCLUDE_TYPES = tuple()          # e.g., (nn.Conv1d,)

    # Optional: register fused hooks by attribute name:
    # {"up": "fused_nn", "down": "fused_avgpool"} or callable hooks
    FUSED_HOOKS = {}

    # ---- minimal fused hooks you can reference by name ----
    @staticmethod
    def _fused_nn_hook(module, inputs):
        if not hasattr(module, "weight"): return
        w = module.weight
        w = F.pad(w, (1, 1, 1, 1))
        module.weight = (w[..., 1:, 1:] + w[..., 1:, :-1] +
                         w[..., :-1, 1:] + w[..., :-1, :-1])

    @staticmethod
    def _fused_avgpool_hook(module, inputs):
        if not hasattr(module, "weight"): return
        w = module.weight
        w = F.pad(w, (1, 1, 1, 1))
        module.weight = (w[..., 1:, 1:] + w[..., 1:, :-1] +
                         w[..., :-1, 1:] + w[..., :-1, :-1]) * 0.25

    # ---- PGGAN-ish inits (used only if ELR_INIT == "pggan") ----
    def _pggan_init(self, m: nn.Module):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            if getattr(m, "weight", None) is not None:
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            if getattr(m, "weight", None) is not None:
                nn.init.normal_(m.weight, mean=0.0, std=(1.0 / max(self.ELR_LR_MUL, 1e-8)))
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    # ---- core: auto-wrap any module assigned as an attribute ----
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        try:
            self._maybe_autowrap(name, value)
        except Exception:
            # Be forgiving during construction; you can tighten this if desired.
            pass

    def _maybe_autowrap(self, name: str, value):
        if not self.ELR_ENABLED or not isinstance(value, nn.Module):
            return
        if name in self.EXCLUDE_NAMES:
            return

        # Register fused hook if requested for this attribute
        hook_spec = self.FUSED_HOOKS.get(name, None)
        if hook_spec is not None:
            hook = (self._fused_nn_hook if hook_spec == "fused_nn"
                    else self._fused_avgpool_hook if hook_spec == "fused_avgpool"
                    else hook_spec if callable(hook_spec) else None)
            if hook is not None:
                value.register_forward_pre_hook(hook)

        # Equalized LR for this module and its descendants
        self._wrap_module_tree(value)

    def _wrap_module_tree(self, root: nn.Module):
        param_name = self.ELR_PARAM
        cfg = dict(gain=self.ELR_GAIN, mode=self.ELR_MODE, lr_mul=self.ELR_LR_MUL)

        def should(m: nn.Module):
            return (isinstance(m, self.ELR_TYPES)
                    and not isinstance(m, self.EXCLUDE_TYPES)
                    and hasattr(m, param_name)
                    and isinstance(getattr(m, param_name), torch.nn.Parameter))

        # include root and all children
        for m in [root, *root.modules()]:
            if not should(m):
                continue
            # avoid double-wrapping
            if any(isinstance(h, EqualizedLR) for h in m._forward_pre_hooks.values()):
                continue
            EqualizedLR.apply(m, param_name, **cfg)
            if self.ELR_INIT == "pggan":
                self._pggan_init(m)
