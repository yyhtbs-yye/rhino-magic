# rhino/nn/utils/ext_loader.py
# Simple, fully-functional loader for custom Torch ops (no mmcv/parrots).

from __future__ import annotations
from typing import Callable, Dict, Iterable, Optional
import warnings
import torch

try:
    import torchvision.ops as _TVOPS  # optional
except Exception:  # noqa: BLE001
    _TVOPS = None


# Minimal alias map; extend if you need more.
_TORCHVISION_ALIAS: Dict[str, str] = {
    "nms": "nms",
    "batched_nms": "batched_nms",
    "roi_align": "roi_align",
    "roi_pool": "roi_pool",
}


def _resolve_from_torch_ops(namespace: str, fun: str):
    """Return torch.ops.<namespace>.<fun> if it exists, else None."""
    try:
        ns = getattr(torch.ops, namespace)
    except AttributeError:
        return None
    return getattr(ns, fun, None)


def _resolve_from_torchvision(fun: str):
    """Return torchvision.ops.<fun> (or alias) if available, else None."""
    if _TVOPS is None:
        return None
    if hasattr(_TVOPS, fun):
        return getattr(_TVOPS, fun)
    alias = _TORCHVISION_ALIAS.get(fun)
    if alias and hasattr(_TVOPS, alias):
        return getattr(_TVOPS, alias)
    return None


def _make_stub(namespace: str, fun: str) -> Callable:
    """Callable that raises a helpful error when invoked."""
    def _raise(*_args, **_kwargs):
        msg = (
            f"Custom op '{fun}' is not available.\n"
            f"Tried: torch.ops.{namespace}.{fun}"
            + (f", torchvision.ops.{fun}" if _TVOPS is not None else "")
            + ".\n"
              "If this is a compiled extension, make sure it is built and "
              f"registered under namespace '{namespace}'. "
              "If you don't need the custom CUDA path, pass use_custom_op=False."
        )
        raise NotImplementedError(msg)
    # Mark as stub so we can identify it later without re-creating it.
    _raise.__ext_stub__ = True  # type: ignore[attr-defined]
    return _raise


class ExtModule:
    """Lightweight module-like container with a helpful __repr__."""
    def __init__(self, namespace: str, mapping: Dict[str, Callable], missing: Iterable[str]):
        self.__namespace = namespace
        self.__missing = sorted(set(missing))
        self.__resolved = sorted([k for k in mapping.keys() if k not in self.__missing])
        for k, v in mapping.items():
            setattr(self, k, v)

    def __repr__(self) -> str:  # pretty debug print
        parts = [f"<ExtModule namespace='{self.__namespace}'>"]
        if self.__resolved:
            parts.append(f"  resolved: {self.__resolved}")
        if self.__missing:
            parts.append(f"  missing: {self.__missing} (lazy stubs)")
        parts.append("</ExtModule>")
        return "\n".join(parts)


def load_ext(
    name: str,
    funcs: Iterable[str],
    *,
    fallbacks: Optional[Dict[str, Callable]] = None,
) -> ExtModule:
    """
    Build a lightweight object exposing requested ops as attributes.

    Resolution order per function:
      1) torch.ops.<name>.<func>
      2) torchvision.ops.<func> (or alias), when present
      3) user-provided fallback in `fallbacks` (pure PyTorch callable)
      4) stub that raises NotImplementedError *when called*
    """
    fallbacks = fallbacks or {}
    mapping: Dict[str, Callable] = {}
    missing: list[str] = []

    for fun in funcs:
        op = (
            _resolve_from_torch_ops(name, fun)
            or _resolve_from_torchvision(fun)
            or fallbacks.get(fun)
        )
        if op is None:
            op = _make_stub(name, fun)
            missing.append(fun)
        mapping[fun] = op

    return ExtModule(name, mapping, missing)


def check_ops_exist(namespace: str = "_ext", funcs: Optional[Iterable[str]] = None) -> bool:
    """
    Quick availability check.
    - If `funcs` is given, returns True iff *all* can be resolved from torch.ops or torchvision.
    - If `funcs` is None, returns True iff the namespace exists on torch.ops or torchvision is present.
    """
    if funcs is None:
        has_ns = hasattr(torch.ops, namespace)
        has_tv = _TVOPS is not None
        return bool(has_ns or has_tv)

    for f in funcs:
        if _resolve_from_torch_ops(namespace, f) is None and _resolve_from_torchvision(f) is None:
            return False
    return True
