
import torch
import torch.nn.functional as F

@torch.no_grad()
def sample_from_discretized_mix_logistic(
    params: torch.Tensor,
    nr_mix: int,
    in_channels: int,
    *,
    coupled: bool = False,
) -> torch.Tensor:
    """
    Sample from a DMoL head with arbitrary channels.

    Args:
      params: (B, n_out, H, W)
      nr_mix: number of mixtures, K
      in_channels: number of channels, C
      coupled: if True, sample channels sequentially with triangular coupling:
               mean[c] <- mean[c] + sum_{j < c} coeff[c,j] * x[j]

    Returns:
      x: (B, C, H, W) in [-1, 1]
    """
    # Split params using the generalized splitter that supports `coupled`
    logit_probs, means, log_scales, coeffs = _split_params(
        params, nr_mix, in_channels, coupled=coupled
    )  # shapes: (B,K,H,W), (B,K,C,H,W), (B,K,C,H,W), (B,K,n_coeff,H,W)|None

    B, K, H, W = logit_probs.shape
    C = in_channels

    # Gumbel-max to pick mixture component per pixel
    u = torch.rand_like(logit_probs)
    g = -torch.log(-torch.log(torch.clamp(u, 1e-6, 1 - 1e-6)))
    sel = torch.argmax(logit_probs + g, dim=1)                           # (B,H,W)
    one_hot = F.one_hot(sel, num_classes=K).permute(0, 3, 1, 2).float()  # (B,K,H,W)

    def _pick(x: torch.Tensor) -> torch.Tensor:
        # If x is (B,K,H,W) -> (B,H,W); if (B,K,C,H,W) -> (B,C,H,W); if (B,K,N,H,W) -> (B,N,H,W)
        if x.dim() == 4:
            return torch.sum(x * one_hot, dim=1)
        elif x.dim() == 5:
            return torch.sum(x * one_hot.unsqueeze(2), dim=1)
        else:
            raise ValueError("Unexpected tensor rank in pick().")

    m = _pick(means)                         # (B,C,H,W)
    logs = torch.clamp(_pick(log_scales), min=-7.0)  # (B,C,H,W), clamp for stability

    def _sample_logistic(mu: torch.Tensor, log_s: torch.Tensor) -> torch.Tensor:
        u = torch.rand_like(mu)
        return mu + torch.exp(log_s) * (torch.log(torch.clamp(u, 1e-6, 1-1e-6)) -
                                        torch.log(torch.clamp(1 - u, 1e-6, 1-1e-6)))

    if not coupled:
        # Independent channels (recommended for latents)
        eps = torch.log(torch.clamp(torch.rand_like(m), 1e-6, 1-1e-6)) - \
              torch.log(torch.clamp(1 - torch.rand_like(m), 1e-6, 1-1e-6))
        # Use a single u per element for proper logistic noise:
        u = torch.rand_like(m)
        eps = torch.log(torch.clamp(u, 1e-6, 1-1e-6)) - torch.log(torch.clamp(1 - u, 1e-6, 1-1e-6))
        x = m + torch.exp(logs) * eps
    else:
        # Triangular coupling: coeffs for selected mixture -> (B, n_coeff, H, W)
        co_sel = _pick(coeffs)               # None if C==1; else (B, C*(C-1)/2, H, W)
        x = torch.zeros_like(m)
        offset = 0
        for c in range(C):
            mu_c = m[:, c]                 # (B,H,W)
            log_s_c = logs[:, c]
            if c > 0:
                n_c = c
                co_c = co_sel[:, offset:offset + n_c]  # (B,c,H,W), order: (c←0),(c←1),...,(c←c-1)
                offset += n_c
                # Sum_j< c coeff[c,j] * x_j
                adj = (co_c * x[:, :c]).sum(dim=1)     # (B,H,W)
                mu_c = mu_c + adj
            x[:, c] = _sample_logistic(mu_c, log_s_c)

    return torch.clamp(x, -1.0, 1.0)

def _split_params(params: torch.Tensor, nr_mix: int, in_channels: int, *, coupled: bool = False):
    """
    Returns:
      logit_probs: (B, K, H, W)
      means:       (B, K, C, H, W)
      log_scales:  (B, K, C, H, W)
      coeffs:      (B, K, n_coeff, H, W) if coupled else None
                   where n_coeff = C*(C-1)/2 (triangular ordering)
    """
    B, Cout, H, W = params.shape
    if in_channels < 1:
        raise ValueError("in_channels must be >= 1")

    n_coeff = (in_channels * (in_channels - 1)) // 2 if coupled else 0
    expected = nr_mix * (1 + 2 * in_channels + n_coeff)
    if Cout != expected:
        raise ValueError(f"Expected {expected} channels, got {Cout}")

    idx = 0
    # (B, K, H, W)
    logit_probs = params[:, idx:idx + nr_mix]; idx += nr_mix

    # (B, C*K, H, W) -> (B, K, C, H, W)
    m = params[:, idx:idx + in_channels * nr_mix]; idx += in_channels * nr_mix
    ls = params[:, idx:idx + in_channels * nr_mix]; idx += in_channels * nr_mix

    def _to_kc(x):
        B_, C_, H_, W_ = x.shape
        return x.view(B_, in_channels, nr_mix, H_, W_).permute(0, 2, 1, 3, 4).contiguous()

    means = _to_kc(m)
    log_scales = _to_kc(ls)

    coeffs = None
    if coupled:
        # (B, n_coeff*K, H, W) -> (B, K, n_coeff, H, W)
        co = params[:, idx:idx + n_coeff * nr_mix]; idx += n_coeff * nr_mix
        B_, Cc, H_, W_ = co.shape
        coeffs = torch.tanh(
            co.view(B_, n_coeff, nr_mix, H_, W_).permute(0, 2, 1, 3, 4).contiguous()
        )

    return logit_probs, means, log_scales, coeffs

