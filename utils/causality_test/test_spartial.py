

# ------------------------------------------------------------------------------------------------------------------
import random

# ======= CONFIG =======
SEED = 1234
DEVICE = "cpu"                 # keep CPU to avoid nondeterministic CUDA convs
B, C, H, W = 1, 3, 6, 6        # batch, channels, height, width
VOCAB, HIDDEN, N_BLOCKS = 8, 48, 2
N_TRIALS = 8                   # how many random target positions to test
ATOL = 1e-6

torch.manual_seed(SEED)
random.seed(SEED)

def trunk_forward(model: nn.Module, x_long: torch.Tensor) -> torch.Tensor:
    """
    Runs ONLY the backbone: embedding -> MaskedConv2d(mask A) -> ResMaskedBlocks
    Returns backbone features h of shape (B, hidden_channels, H, W).
    """
    assert x_long.dtype == torch.long
    if x_long.dim() == 3:
        x_long = x_long.unsqueeze(1)

    B, C, H, W = x_long.shape
    # replicate the model.forward() pre-head steps
    eh = model.embedding(rearrange(x_long, 'b c h w -> b (h w) c'))            # (B, H*W, C, E)
    z  = rearrange(eh, 'b (h w) c e -> b (e c) h w', h=H, w=W)                 # (B, E*C, H, W)
    h0 = model.act_in(model.conv_in(z))
    h  = model.blocks(h0)
    return h

def is_future(i, j, p_i, p_j):
    """Raster-scan future: positions strictly after (p_i,p_j)."""
    return (i > p_i) or (i == p_i and j > p_j)

def randomize_positions(x_long: torch.Tensor, positions, vocab_size: int):
    """
    For each (c,i,j) in positions, reassign x[b=0,c,i,j] to a different random token in [0, V-1].
    """
    for (c, i, j) in positions:
        old = x_long[0, c, i, j].item()
        if vocab_size == 1:
            continue
        new_val = random.randrange(vocab_size - 1)
        if new_val >= old:
            new_val += 1
        x_long[0, c, i, j] = new_val

def pick_random_target(H, W):
    return random.randrange(H), random.randrange(W)

def collect_positions(H, W, C, predicate):
    """Return all (c,i,j) satisfying predicate(i,j)."""
    out = []
    for i in range(H):
        for j in range(W):
            if predicate(i, j):
                for c in range(C):
                    out.append((c, i, j))
    return out

def assert_allclose(a, b, atol=1e-6, msg=""):
    if not torch.allclose(a, b, atol=atol, rtol=0.0):
        diff = (a - b).abs().max().item()
        raise AssertionError(f"{msg} | max|Δ|={diff:.3e} (atol={atol})")

def assert_not_allclose(a, b, atol=1e-6, msg=""):
    if torch.allclose(a, b, atol=atol, rtol=0.0):
        raise AssertionError(f"{msg} | tensors unexpectedly equal within atol={atol}")

def main():
    # Build model (head is created but never used)
    model = PixelCNNMinusMinus(
        vocab_size=VOCAB,
        in_channels=C,
        hidden_channels=HIDDEN,
        n_blocks=N_BLOCKS,
        kernel_size=3,
        embed_dim=16,
        dropout=0.0,
    ).to(DEVICE).eval()

    # base input
    x = torch.randint(low=0, high=VOCAB, size=(B, C, H, W), dtype=torch.long, device=DEVICE)

    # Run multiple randomized causality checks
    for t in range(N_TRIALS):
        # choose a target spatial position (p_i,p_j)
        p_i, p_j = pick_random_target(H, W)

        # Baseline backbone features at target
        h_ref = trunk_forward(model, x)
        feat_ref = h_ref[0, :, p_i, p_j].detach().clone()

        # --- Test 1: Invariance to FUTURE pixels (strict spatial causality) ---
        x_future = x.clone()
        future_positions = collect_positions(
            H, W, C,
            predicate=lambda i, j: is_future(i, j, p_i, p_j)
        )
        randomize_positions(x_future, future_positions, VOCAB)

        h_future = trunk_forward(model, x_future)
        feat_future = h_future[0, :, p_i, p_j]

        assert_allclose(
            feat_ref, feat_future, atol=ATOL,
            msg=f"[Trial {t}] Trunk changed at ({p_i},{p_j}) when ONLY future pixels changed."
        )

        # --- Test 2 (sanity): Sensitivity to PAST pixels (should usually change) ---
        # Not a strict requirement, but good to ensure the backbone does respond to past context.
        past_positions = collect_positions(
            H, W, C,
            predicate=lambda i, j: (i < p_i) or (i == p_i and j < p_j)
        )
        if past_positions:  # skip top-left corner
            x_past = x.clone()
            # Randomize a handful (or all) past pixels to encourage a change
            random.shuffle(past_positions)
            randomize_positions(x_past, past_positions[: max(1, len(past_positions)//2)], VOCAB)

            h_past = trunk_forward(model, x_past)
            feat_past = h_past[0, :, p_i, p_j]

            # Usually this should differ; if it doesn't for a particular seed/spot, it's fine,
            # but across several trials this gives confidence that the backbone uses the past.
            try:
                assert_not_allclose(
                    feat_ref, feat_past, atol=ATOL,
                    msg=f"[Trial {t}] Trunk did NOT change at ({p_i},{p_j}) when past pixels changed."
                )
            except AssertionError as e:
                # Not fatal for causality, just report for visibility.
                print("Note:", e)

        # --- Test 3 (expected here): Invariance to the SELF pixel (center) in backbone ---
        # With Mask A at the first backbone layer and only Mask B thereafter,
        # the backbone should NOT depend on x[p_i,p_j] itself.
        self_positions = [(c, p_i, p_j) for c in range(C)]
        x_self = x.clone()
        randomize_positions(x_self, self_positions, VOCAB)

        h_self = trunk_forward(model, x_self)
        feat_self = h_self[0, :, p_i, p_j]

        assert_allclose(
            feat_ref, feat_self, atol=ATOL,
            msg=f"[Trial {t}] Trunk changed at ({p_i},{p_j}) when ONLY its own pixel changed."
        )

    print(f"✓ Passed {N_TRIALS} backbone spatial-causality trials (future & self invariance).")

if __name__ == "__main__":
    main()
