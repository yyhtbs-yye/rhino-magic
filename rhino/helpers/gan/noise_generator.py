import torch
import torch.nn as nn
from typing import Union, Sequence, Optional, Tuple

class NoiseGenerator(nn.Module):
    """
    Minimal seedable noise generator.

    Args:
        noise_size: int or sequence of ints describing the per-sample shape.
                    Examples: 3, (3, 128, 128), (2, 5), etc.
        noise_type: "gaussian" or "uniform".
        scale: multiplier applied to the noise values.
        random_state: optional integer seed for reproducibility.
    """
    def __init__(
        self,
        noise_size: Union[int, Sequence[int]],
        *,
        noise_type: str = "gaussian",
        scale: float = 1.0,
        random_state: Optional[int] = None,
    ):
        super().__init__()

        if isinstance(noise_size, int):
            size = (int(noise_size),)
        else:
            size = tuple(int(x) for x in noise_size)
            if len(size) == 0:
                raise ValueError("noise_size must be an int or a non-empty sequence of ints.")

        if any(d <= 0 for d in size):
            raise ValueError("All dimensions in noise_size must be > 0.")

        if noise_type not in ("gaussian", "uniform"):
            raise ValueError('noise_type must be "gaussian" or "uniform".')

        self.noise_size: Tuple[int, ...] = size
        self.noise_type = noise_type
        self.scale = float(scale)

        # Internal CPU generator for deterministic sequences across devices.
        self._gen = torch.Generator(device="cpu")
        if random_state is None:
            # If not provided, use a fresh random seed.
            self._gen.manual_seed(torch.seed() % (2**63))
        else:
            self._gen.manual_seed(int(random_state))

    def next(
        self,
        batch_size: int,
        *,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        B = int(batch_size)
        shape = (B, *self.noise_size)
        device = torch.device(device) if device is not None else None
        dtype = dtype if dtype is not None else torch.float32

        if self.noise_type == "gaussian":
            x = torch.randn(shape, generator=self._gen, device="cpu", dtype=dtype)
        else:  # "uniform"
            x = torch.rand(shape, generator=self._gen, device="cpu", dtype=dtype) * 2.0 - 1.0

        x = x * self.scale
        return x.to(device=device if device is not None else x.device, dtype=dtype, non_blocking=True)

    # Optional: make it usable in nn graphs (forward == next)
    def forward(
        self,
        batch_size: int,
        *,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        return self.next(batch_size, device=device, dtype=dtype)
