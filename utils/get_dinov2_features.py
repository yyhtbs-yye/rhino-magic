#!/usr/bin/env python
"""
extract_dinov2_patch_feats_multithread.py

Run example
-----------
python extract_dinov2_patch_feats_multithread.py \
    --input_dir  /path/to/ffhq256 \
    --output_dir /path/to/ffhq256_feats \
    --gpu_ids    0,1,2,3 \
    --batch_size_per_gpu 8
"""
import argparse, pickle, pathlib, sys, math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import torch, timm
from timm.data import resolve_model_data_config
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
def read_image(path: pathlib.Path):
    return Image.open(path).convert("RGB")


def save_pickle(obj, path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


# ──────────────────────────────────────────────────────────────────────────────
# Model & transform helpers
# ──────────────────────────────────────────────────────────────────────────────
def load_backbone(
    name: str = "vit_giant_patch14_dinov2.lvd142m", device: torch.device = "cpu"
):
    model = timm.create_model(
        name,
        pretrained=True,
        num_classes=0,        # drop classifier
        features_only=False,  # keep tokens
    ).to(device)
    model.eval()
    return model


def build_transform(model):
    cfg = resolve_model_data_config(model)
    return transforms.Compose(
        [
            transforms.Pad((0, 0, 40, 40), fill=0),   # 256→296
            transforms.Resize(cfg["input_size"][1], interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(cfg["input_size"][1]),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg["mean"], std=cfg["std"]),
        ]
    )


@torch.no_grad()
def extract_patch_tokens(model, x):
    """
    Return only spatial tokens: (B, 1369, D) for ViT-G/14.
    """
    feats = model.forward_features(x)   # dict with CLS+patch tokens
    return feats[:, 1:, :]              # drop CLS


# ──────────────────────────────────────────────────────────────────────────────
# Worker function executed in **one thread per GPU**
# ──────────────────────────────────────────────────────────────────────────────
def process_on_gpu(
    device_id: int,
    img_paths: List[pathlib.Path],
    out_dir: pathlib.Path,
    batch_size: int,
):
    device = torch.device(f"cuda:{device_id}")
    model  = load_backbone(device=device)
    transform = build_transform(model)

    n_saved = 0
    pbar = tqdm(
        total=len(img_paths),
        desc=f"GPU{device_id}",
        position=device_id,
        leave=False,
    )

    for i in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[i : i + batch_size]
        imgs = [transform(read_image(p)) for p in batch_paths]
        imgs = torch.stack(imgs, dim=0).to(device, non_blocking=True)  # (B, 3, H, W)

        patch = extract_patch_tokens(model, imgs).cpu()                # (B, 1369, D)

        # Crop 37×37 → 32×32
        B, N, D = patch.shape
        gs = int(math.sqrt(N))                                         # 37
        patch = (
            patch.view(B, gs, gs, D)[:, :32, :32, :]                   # keep 32×32
                 .reshape(B, -1, D)                                    # (B, 1024, D)
        )

        # Save per-image
        for j, path in enumerate(batch_paths):
            out_path = out_dir / path.with_suffix(".pkl").name
            save_pickle({"patch_tokens": patch[j : j + 1]}, out_path)
            n_saved += 1

        pbar.update(len(batch_paths))

    pbar.close()
    return n_saved


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main(args):
    torch.backends.cudnn.benchmark = True

    in_dir  = pathlib.Path(args.input_dir).expanduser()
    out_dir = pathlib.Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect images
    img_exts  = {".png", ".jpg", ".jpeg", ".webp"}
    img_paths = sorted([p for p in in_dir.glob("*") if p.suffix.lower() in img_exts])
    if not img_paths:
        sys.exit(f"No images with extensions {img_exts} found in {in_dir}")

    device_ids = [int(d) for d in args.gpu_ids.split(",") if d.strip() != ""]
    if not device_ids:
        sys.exit("No GPU IDs provided")

    # Split the image list evenly across GPUs
    chunks = [
        img_paths[i::len(device_ids)] for i in range(len(device_ids))
    ]

    total_expected = len(img_paths)
    total_saved = 0

    with ThreadPoolExecutor(max_workers=len(device_ids)) as pool:
        futures = [
            pool.submit(
                process_on_gpu,
                device_id,
                chunk,
                out_dir,
                args.batch_size_per_gpu,
            )
            for device_id, chunk in zip(device_ids, chunks)
        ]

        for f in tqdm(as_completed(futures), total=len(futures), desc="GPUs finished"):
            total_saved += f.result()

    print(f"Done. Saved {total_saved}/{total_expected} pickle files to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",  type=str, default="data/ffhq/ffhq_imgs/ffhq_256",
                        help="Directory with input images (e.g., FFHQ-256 PNGs)")
    parser.add_argument("--output_dir", type=str, default="data/ffhq/ffhq_dinov2/ffhq_256",
                        help="Where to write <image-name>.pkl files")
    parser.add_argument("--gpu_ids",    type=str, default="0,1,2,3",
                        help="Comma-separated GPU IDs to use (e.g. '0,1,2,3')")
    parser.add_argument("--batch_size_per_gpu", type=int, default=8,
                        help="Mini-batch size handled by **each** GPU thread")
    args = parser.parse_args()
    main(args)

