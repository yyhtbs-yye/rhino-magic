# run_pixelcnnpp_cpu_test.py
import time
import torch

from rhino.nn.autoregression.pixelcnnpp.pixelcnnpp import PixelCNNPP
from rhino.nn.autoregression.pixelcnnpp.pixelcnnpp_loss import discretized_mix_logistic_loss

def total_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def to_minus1_1(x_uint8: torch.Tensor) -> torch.Tensor:
    # [0,255] -> [-1,1]
    return x_uint8.float() / 127.5 - 1.0

def test_forward_and_backward_rgb():
    print("== RGB forward/backward on CPU ==")
    torch.manual_seed(0)
    device = torch.device("cpu")

    model = PixelCNNPP(
        in_channels=3,
        hidden_channels=64,
        n_blocks=2,
        kernel_size=3,
        embed_dim=0,    # unused in unconditional version
        dropout=0.1,
    ).to(device).train()

    print("Param count:", total_parameters(model))

    B, C, H, W = 2, 3, 16, 16
    x = to_minus1_1(torch.randint(0, 256, (B, C, H, W), device=device, dtype=torch.int64))

    t0 = time.time()
    params = model(x)
    fwd_time = time.time() - t0

    expected_cout = (10 if C == 3 else 3) * model.nr_mix
    print("Forward OK. Output shape:", tuple(params.shape), "Expected C_out:", expected_cout)
    print("Forward time: %.3fs" % fwd_time)

    loss = discretized_mix_logistic_loss(
        x, params, nr_mix=model.nr_mix, in_channels=C, reduction="mean"
    )
    print("Loss:", float(loss))

    model.zero_grad(set_to_none=True)
    loss.backward()

    # Quick grad sanity
    grad_norm = sum(
        (p.grad.abs().mean().item() if p.grad is not None else 0.0)
        for p in model.parameters()
    )
    print("Mean grad (aggregate):", grad_norm)

def test_sampling_small_rgb():
    print("\n== RGB sampling on CPU ==")
    torch.manual_seed(123)
    device = torch.device("cpu")

    model = PixelCNNPP(
        in_channels=3,
        hidden_channels=64,
        n_blocks=2,
        kernel_size=3,
        embed_dim=0,
        dropout=0.0,
    ).to(device).eval()

    B, C, H, W = 2, 3, 8, 8  # keep tiny; sampling is O(HW)
    t0 = time.time()
    with torch.no_grad():
        samples = model.sample((B, C, H, W), device=device)
    dt = time.time() - t0

    print("Samples shape:", tuple(samples.shape))
    print("Value range: [%.3f, %.3f]" % (samples.min().item(), samples.max().item()))
    print("Sampling time: %.3fs" % dt)

def test_forward_and_sampling_gray():
    print("\n== Grayscale forward + sampling on CPU ==")
    torch.manual_seed(7)
    device = torch.device("cpu")

    model = PixelCNNPP(
        in_channels=1,
        hidden_channels=48,
        n_blocks=2,
        kernel_size=3,
        embed_dim=0,
        dropout=0.0,
    ).to(device)

    B, C, H, W = 2, 1, 12, 12
    x = to_minus1_1(torch.randint(0, 256, (B, C, H, W), device=device))

    # Forward (eval or train both fine)
    model.train()
    params = model(x)
    print("Gray forward OK. Output shape:", tuple(params.shape))

    loss = discretized_mix_logistic_loss(x, params, nr_mix=model.nr_mix, in_channels=C, reduction="mean")
    print("Gray loss:", float(loss))
    model.zero_grad(set_to_none=True)
    loss.backward()

    # Sampling
    model.eval()
    with torch.no_grad():
        samples = model.sample((B, C, 8, 8), device=device)
    print("Gray samples shape:", tuple(samples.shape), "range [%.3f, %.3f]" %
          (samples.min().item(), samples.max().item()))

if __name__ == "__main__":
    test_forward_and_backward_rgb()
    test_sampling_small_rgb()
    test_forward_and_sampling_gray()
    print("\nAll CPU smoke tests completed.")
