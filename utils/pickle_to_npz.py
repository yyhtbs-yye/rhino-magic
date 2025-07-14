#!/usr/bin/env python
"""
compress_dinov2_features.py

Compress DINOv2 feature pickle files to reduce storage space.
Supports multiple compression formats: gzip, bz2, lzma, and npz.

Run example
-----------
python compress_dinov2_features.py \
    --input_dir  /path/to/ffhq256_feats \
    --output_dir /path/to/ffhq256_feats_compressed \
    --format npz \
    --num_workers 8
"""
import argparse
import pickle
import pathlib
import gzip
import bz2
import lzma
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
from tqdm import tqdm
import torch


def load_pickle(path: pathlib.Path) -> Dict[str, Any]:
    """Load a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_compressed_pickle(obj: Dict[str, Any], path: pathlib.Path, compression: str):
    """Save object as compressed pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if compression == "gzip":
        with gzip.open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif compression == "bz2":
        with bz2.open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif compression == "lzma":
        with lzma.open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        raise ValueError(f"Unsupported compression format: {compression}")


def save_npz(obj: Dict[str, Any], path: pathlib.Path):
    """Save patch tokens as compressed numpy arrays (.npz)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert torch tensor to numpy if needed
    patch_tokens = obj["patch_tokens"]
    if isinstance(patch_tokens, torch.Tensor):
        patch_tokens = patch_tokens.numpy()
    
    # Save as compressed npz
    np.savez_compressed(path, patch_tokens=patch_tokens)


def get_file_size_mb(path: pathlib.Path) -> float:
    """Get file size in MB."""
    return path.stat().st_size / (1024 * 1024)


def process_file(
    input_path: pathlib.Path,
    output_dir: pathlib.Path,
    format_type: str
) -> tuple[str, float, float]:
    """Process a single pickle file and return compression stats."""
    
    # Load original pickle
    try:
        data = load_pickle(input_path)
    except Exception as e:
        return f"Error loading {input_path.name}: {e}", 0, 0
    
    # Determine output path and extension
    if format_type == "npz":
        output_path = output_dir / input_path.with_suffix(".npz").name
    else:
        extension_map = {"gzip": ".pkl.gz", "bz2": ".pkl.bz2", "lzma": ".pkl.xz"}
        output_path = output_dir / (input_path.stem + extension_map[format_type])
    
    # Get original file size
    original_size = get_file_size_mb(input_path)
    
    # Save compressed version
    try:
        if format_type == "npz":
            save_npz(data, output_path)
        else:
            save_compressed_pickle(data, output_path, format_type)
        
        compressed_size = get_file_size_mb(output_path)
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
        
        return (
            f"✓ {input_path.name} -> {output_path.name}",
            original_size,
            compressed_size
        )
        
    except Exception as e:
        return f"Error compressing {input_path.name}: {e}", original_size, 0


def process_batch(
    file_paths: List[pathlib.Path],
    output_dir: pathlib.Path,
    format_type: str
) -> tuple[List[str], float, float]:
    """Process a batch of files."""
    results = []
    total_original = 0
    total_compressed = 0
    
    for path in file_paths:
        result, orig_size, comp_size = process_file(path, output_dir, format_type)
        results.append(result)
        total_original += orig_size
        total_compressed += comp_size
    
    return results, total_original, total_compressed


def main(args):
    input_dir = pathlib.Path(args.input_dir).expanduser()
    output_dir = pathlib.Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all pickle files
    pickle_files = sorted(list(input_dir.glob("*.pkl")))
    if not pickle_files:
        print(f"No .pkl files found in {input_dir}")
        return
    
    print(f"Found {len(pickle_files)} pickle files to compress")
    print(f"Using format: {args.format}")
    print(f"Using {args.num_workers} workers")
    
    # Split files into chunks for workers
    chunk_size = max(1, len(pickle_files) // args.num_workers)
    file_chunks = [
        pickle_files[i:i + chunk_size] 
        for i in range(0, len(pickle_files), chunk_size)
    ]
    
    total_original_size = 0
    total_compressed_size = 0
    all_results = []
    
    # Process files in parallel
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        # Submit all chunks
        futures = [
            executor.submit(process_batch, chunk, output_dir, args.format)
            for chunk in file_chunks
        ]
        
        # Collect results
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
            results, orig_size, comp_size = future.result()
            all_results.extend(results)
            total_original_size += orig_size
            total_compressed_size += comp_size
    
    # Print summary
    print("\n" + "="*60)
    print("COMPRESSION SUMMARY")
    print("="*60)
    print(f"Files processed: {len(pickle_files)}")
    print(f"Original total size: {total_original_size:.2f} MB")
    print(f"Compressed total size: {total_compressed_size:.2f} MB")
    
    if total_compressed_size > 0:
        compression_ratio = total_original_size / total_compressed_size
        space_saved = total_original_size - total_compressed_size
        percent_saved = (space_saved / total_original_size) * 100
        
        print(f"Space saved: {space_saved:.2f} MB ({percent_saved:.1f}%)")
        print(f"Compression ratio: {compression_ratio:.2f}x")
    
    print(f"\nCompressed files saved to: {output_dir}")
    
    # Show any errors
    errors = [r for r in all_results if r.startswith("Error")]
    if errors:
        print(f"\nErrors encountered ({len(errors)}):")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compress DINOv2 feature pickle files")
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="data/ffhq/ffhq_dinov2_base/ffhq_256",
        help="Directory containing .pkl feature files"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/ffhq/ffhq_dinov2_base_npz/ffhq_256",
        help="Directory to save compressed files"
    )
    parser.add_argument(
        "--format", 
        type=str, 
        choices=["gzip", "bz2", "lzma", "npz"],
        default="npz",
        help="Compression format (default: npz)"
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    
    args = parser.parse_args()
    main(args)