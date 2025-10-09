import torch
import torch.nn as nn
from typing import Tuple

from rhino.nn.autoregression.pixelcnnpp.pixelcnnpp_modules import ShiftedConv2d, GatedResidualBlock
from rhino.nn.autoregression.pixelcnnpp.pixelcnnpp_utils import get_n_out_channels_for_dmol
from rhino.nn.autoregression.utils.sample_from_discretized_mix_logistic import sample_from_discretized_mix_logistic

class PixelCNNPP(nn.Module):
    """
    PixelCNN++ with shifted convolutions (no masks), dual stacks (down & downright), and DMOL output head.

    It is output continuos space.
    

    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 120,
        n_blocks: int = 6,
        kernel_size: int = 3,
        dropout: float = 0.0,
        nr_mix: int = 10,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.n_blocks = n_blocks
        self.kernel_size = kernel_size
        self.dropout = dropout

        # Mixture components used by the DMOL head (change if desired)
        self.nr_mix = nr_mix

        # Initial shifted convs for v- and h-stacks
        self.v_init = ShiftedConv2d(in_channels, hidden_channels, kernel_size, shift="down")
        self.h_init = ShiftedConv2d(in_channels, hidden_channels, kernel_size, shift="downright")

        # Residual stacks, v->h couplings, and skip projections
        self.v_blocks = nn.ModuleList()
        self.h_blocks = nn.ModuleList()
        self.v_to_h = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        for _ in range(n_blocks):
            self.v_blocks.append(
                GatedResidualBlock(
                    channels=hidden_channels,
                    kernel_size=kernel_size,
                    shift="down",
                    dropout=dropout,
                    aux_channels=0,
                )
            )
            self.v_to_h.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1))

            self.h_blocks.append(
                GatedResidualBlock(
                    channels=hidden_channels,
                    kernel_size=kernel_size,
                    shift="downright",
                    dropout=dropout,
                    aux_channels=hidden_channels,  # receives v each block
                )
            )
            self.skip_convs.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1))

        # Output head: sum of skips -> DMOL params
        out_channels = get_n_out_channels_for_dmol(in_channels, self.nr_mix)
        self.out_head = nn.Sequential(
            nn.ELU(inplace=False),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.ELU(inplace=False),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: (B, C, H, W) in [-1, 1]
        Returns:
          DMOL params: (B, n_out, H, W)
        """
        v = self.v_init(x)
        h = self.h_init(x)
        skip = 0.0

        for v_block, to_h, h_block, skip_conv in zip(self.v_blocks, self.v_to_h, self.h_blocks, self.skip_convs):
            v = v_block(v)
            h = h_block(h, aux=to_h(v))
            skip = skip + skip_conv(h)

        return self.out_head(skip)

    @torch.no_grad()
    def sample(self, x) -> torch.Tensor:
        """
        Autoregressive ancestral sampling (top-left → bottom-right).

        Args:
          shape: (B, C, H, W) to generate
          device: torch.device to place the samples on (defaults to model's device)

        Returns:
          samples: (B, C, H, W) in [-1, 1]
        """
        B, C, H, W = x.shape
        if C != self.in_channels:
            raise ValueError(f"Requested C={C}, but model was built for C={self.in_channels}")

        for i in range(H):
            for j in range(W):
                # Forward pass on the partially-filled image to get DMOL params
                params = self.forward(x)  # (B, n_out, H, W)

                # Sample only the current pixel (i, j)
                p_ij = params[:, :, i:i+1, j:j+1]
                s_ij = sample_from_discretized_mix_logistic(
                    p_ij,
                    nr_mix=self.nr_mix,
                    in_channels=self.in_channels,
                )
                x[:, :, i, j] = s_ij[:, :, 0, 0]

        return x

from typing import List, Tuple, Dict

@torch.inference_mode(False)
def assert_pixelcnnpp_causality(
    model: torch.nn.Module,
    in_channels: int,
    H: int = 9,
    W: int = 9,
    positions: int = 0,         # 0 = test all pixels, else test this many random pixels
    seed: int = 0,
    atol: float = 1e-12,        # absolute tolerance for tiny numerical noise
    rtol: float = 1e-6,         # relative tolerance scaled by max grad magnitude per check
    verbose: bool = True,
) -> Dict:
    """
    Gradient-based causality test for PixelCNN++-style models with shifted convolutions.

    For many pixel locations (i, j), we:
      1) run a forward pass,
      2) sum the output logits/params at (i, j),
      3) backprop to the input,
      4) verify that gradients on any forbidden inputs (self + future pixels in raster order)
         are identically zero (up to small tolerances).

    Args:
        model:       Your PixelCNNPP instance (should be on the intended device).
        in_channels: Number of input channels the model expects.
        H, W:        Spatial size to test with (kept modest to keep the test fast).
        positions:   0 to test all H*W positions, or a positive integer to test a random subset.
        seed:        RNG seed when sampling a subset of positions.
        atol, rtol:  Absolute/relative tolerances for treating grads as zero.
        verbose:     Print a short human-readable summary.

    Returns:
        A dict with fields:
            passed (bool): whether all checks passed
            violations (list): each item describes a leaking (i, j) with offending positions
            tested_positions (int): number of (i, j) tested
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()  # turn off dropout etc.

    # Freeze parameter grads (we only need input grads)
    reqs = []
    for p in model.parameters():
        reqs.append(p.requires_grad)
        p.requires_grad_(False)

    # Random input
    B, C = 1, in_channels
    x = torch.randn(B, C, H, W, device=device, requires_grad=True)

    # Choose which (i, j) to test
    all_positions: List[Tuple[int, int]] = [(i, j) for i in range(H) for j in range(W)]
    if positions and positions < len(all_positions):
        g = torch.Generator(device=device).manual_seed(seed)
        perm = torch.randperm(len(all_positions), generator=g, device=device).tolist()
        test_set = [all_positions[k] for k in perm[:positions]]
    else:
        test_set = all_positions

    violations = []
    # Keep graph and reuse it; we’ll backward repeatedly from different output pixels.
    # That means calling backward with retain_graph=True and manually zeroing x.grad.
    y = model(x)  # shape: (1, n_out, H, W)

    for (i, j) in test_set:
        # Scalar objective: sum all output channels at this (i, j)
        scalar = y[0, :, i, j].sum()

        # Zero input grad from previous iteration
        if x.grad is not None:
            x.grad.zero_()

        scalar.backward(retain_graph=True)

        # Reduce grad magnitude across channels -> (H, W)
        grad_map = x.grad.detach().abs().sum(dim=1)[0]  # (H, W)
        maxg = float(grad_map.max().item())
        eps = atol + rtol * maxg

        # Forbidden region: self and all pixels "after" (i, j) in raster order
        #   allowed: (r < i) or (r == i and c < j)
        #   forbidden: everything else (including (i, j) itself)
        allowed_mask = torch.zeros(H, W, dtype=torch.bool, device=device)
        if j > 0:
            allowed_mask[i, :j] = True
        if i > 0:
            allowed_mask[:i, :] = True
        forbidden_mask = ~allowed_mask

        # Locations with suspicious (non-zero) grad in forbidden region
        bad = (grad_map > eps) & forbidden_mask
        if bad.any():
            # Collect all offending coordinates
            locs = torch.nonzero(bad, as_tuple=False).tolist()
            violations.append({
                "pixel": (i, j),
                "max_grad": maxg,
                "tolerance": eps,
                "offenders": [(int(r), int(c), float(grad_map[r, c].item())) for r, c in locs],
            })

    # Restore param requires_grad
    for p, r in zip(model.parameters(), reqs):
        p.requires_grad_(r)
    if was_training:
        model.train()

    passed = (len(violations) == 0)
    if verbose:
        if passed:
            print(f"[Causality ✓] No leakage detected across {len(test_set)} positions "
                  f"on a {H}x{W} grid (in_channels={in_channels}).")
        else:
            first = violations[0]
            pi, pj = first["pixel"]
            print(f"[Causality ✗] Leakage detected at output pixel (i={pi}, j={pj}). "
                  f"First few offending inputs (r, c, |grad|): {first['offenders'][:5]}")

    return {
        "passed": passed,
        "violations": violations,
        "tested_positions": len(test_set),
    }

# --- Optional smoke test ---
if __name__ == "__main__":
    # Example usage with your class:
    model = PixelCNNPP(in_channels=3, hidden_channels=64, n_blocks=4, kernel_size=3, dropout=0.0, nr_mix=5).to("cuda")
    res = assert_pixelcnnpp_causality(model, in_channels=3, H=9, W=9, positions=40, seed=0)
    print(res["passed"], f"violations: {len(res['violations'])}")
    # pass
