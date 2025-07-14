#!/usr/bin/env python
"""
verify_dinov2_npz_vs_pickle.py
--------------------------------
Check that the .npz files produced by ``compress_dinov2_features.py --format npz``
are numerically equivalent to the original .pkl files.

“Trivial” differences allowed
* dtype changes (e.g. float32 → float16) **if** values match within tolerance
* device / contiguity changes (CUDA → CPU, strides, etc.)

Anything else – shape mismatch, values outside tolerance, missing .npz,
extra / missing dictionary keys – is counted as **non-trivial**.

Example
-------
python verify_dinov2_npz_vs_pickle.py \
    --input_dir  /path/to/ffhq256_feats \
    --output_dir /path/to/ffhq256_feats_compressed \
    --rtol 1e-05 --atol 1e-08 \
    --num_workers 128
"""
import argparse
import pathlib
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm


def load_pickle(path: pathlib.Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_npz(path: pathlib.Path):
    data = np.load(path)
    # Stored under the key 'patch_tokens'
    return {"patch_tokens": data["patch_tokens"]}


def compare_arrays(a: np.ndarray, b: np.ndarray, rtol: float, atol: float) -> Tuple[bool, str]:
    if a.shape != b.shape:
        return False, f"shape mismatch {a.shape} vs {b.shape}"
    if not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
        max_diff = np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))
        return False, f"values differ (max Δ = {max_diff:.3e})"
    return True, ""


def check_pair(
    pkl_path: pathlib.Path,
    npz_dir: pathlib.Path,
    rtol: float,
    atol: float,
) -> Tuple[str, bool]:
    base = pkl_path.stem                     # e.g. "000123"
    npz_path = npz_dir / f"{base}.npz"

    if not npz_path.exists():
        return f"✗ {base}: .npz missing", False

    try:
        pkl_obj = load_pickle(pkl_path)
    except Exception as e:
        return f"✗ {base}: failed to load pickle ({e})", False

    try:
        npz_obj = load_npz(npz_path)
    except Exception as e:
        return f"✗ {base}: failed to load npz ({e})", False

    # --- key checks --------------------------------------------------------
    pkl_keys = set(pkl_obj.keys())
    npz_keys = set(npz_obj.keys())

    if pkl_keys != npz_keys:
        extra = pkl_keys - npz_keys
        missing = npz_keys - pkl_keys
        return (
            f"✗ {base}: key mismatch "
            f"(extra in pkl={list(extra)}, missing in npz={list(missing)})",
            False,
        )

    # --- array comparison --------------------------------------------------
    pkl_tokens = pkl_obj["patch_tokens"]
    if isinstance(pkl_tokens, torch.Tensor):
        pkl_tokens = pkl_tokens.cpu().detach().numpy()

    npz_tokens = npz_obj["patch_tokens"]

    ok, msg = compare_arrays(pkl_tokens, npz_tokens, rtol, atol)
    if ok:
        return f"✓ {base}", True
    return f"✗ {base}: {msg}", False


def main():
    parser = argparse.ArgumentParser(description="Validate npz vs pickle numerical equivalence")
    parser.add_argument("--input_dir", default="data/ffhq/ffhq_dinov2_base/ffhq_256", help="Folder with original .pkl files")
    parser.add_argument("--output_dir", default="data/ffhq/ffhq_dinov2_base_npz/ffhq_256", help="Folder with .npz files")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance for allclose")
    parser.add_argument("--atol", type=float, default=1e-8, help="Absolute tolerance for allclose")
    parser.add_argument("--num_workers", type=int, default=128, help="Thread pool size (I/O bound)")
    args = parser.parse_args()

    pkl_dir = pathlib.Path(args.input_dir).expanduser()
    npz_dir = pathlib.Path(args.output_dir).expanduser()

    pkl_files = sorted(pkl_dir.glob("*.pkl"))
    if not pkl_files:
        print(f"No pickle files found in {pkl_dir}", file=sys.stderr)
        sys.exit(1)

    total = len(pkl_files)
    ok_count = 0
    results = []

    print(f"Verifying {total} files with rtol={args.rtol}, atol={args.atol}, workers={args.num_workers}")

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(check_pair, p, npz_dir, args.rtol, args.atol)
            for p in pkl_files
        ]
        for fut in tqdm(as_completed(futures), total=total, desc="Comparing"):
            message, ok = fut.result()
            if not ok:
                results.append(message)
            else:
                ok_count += 1

    # --- summary ------------------------------------------------------------
    print("\n========== SUMMARY ==========")
    print(f"Matching files : {ok_count}/{total}")
    print(f"Non-trivial diff: {total - ok_count}")
    if results:
        print("\nFirst 20 issues:")
        for msg in results[:20]:
            print("  ", msg)
        if len(results) > 20:
            print(f"  ... and {len(results) - 20} more")

    # exit status: 0 if all OK, 1 otherwise
    sys.exit(0 if ok_count == total else 1)


if __name__ == "__main__":
    main()
