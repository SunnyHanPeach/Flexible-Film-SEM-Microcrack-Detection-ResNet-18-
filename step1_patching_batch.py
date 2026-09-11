"""
preprocess_patches.py
Spatial Sub-window Patching & Dataset Preprocessing for SEM Micrographs.

Author: Sun Han
Description:
    Extracts overlapping spatial sub-patches (Region of Interest) from raw
    large-scale SEM micrographs to construct balanced patch-level datasets
    for convolutional neural network training and validation.
"""

import os
import cv2
import argparse
import glob
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


def extract_patches_from_image(
        img_path: str,
        output_dir: str,
        category: str,
        patch_size: int = 32,
        stride: int = 16,
) -> int:
    """
    Extract sub-patches from a single micrograph using a 2D sliding window.

    Args:
        img_path: Path to the raw SEM image.
        output_dir: Destination root directory.
        category: Class label (e.g., 'crack', 'intact').
        patch_size: Spatial dimensions of the square patch (default: 32).
        stride: Step size for window displacement (default: 16, 50% overlap).

    Returns:
        Total number of patches generated.
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"[ERROR] Failed to load image: {img_path}")
        return 0

    h, w, _ = img.shape
    target_folder = Path(output_dir) / category
    target_folder.mkdir(parents=True, exist_ok=True)

    stem = Path(img_path).stem
    patch_count = 0

    # Vectorized coordinate computation for sliding window grid
    y_coords = range(0, h - patch_size + 1, stride)
    x_coords = range(0, w - patch_size + 1, stride)

    for y in y_coords:
        for x in x_coords:
            patch = img[y: y + patch_size, x: x + patch_size]

            # Explicit filename format preserving spatial coordinate metadata
            filename = f"{category}_{stem}_y{y}_x{x}.png"
            dest_path = target_folder / filename

            cv2.imwrite(str(dest_path), patch)
            patch_count += 1

    return patch_count


def process_category_worker(args_tuple):
    """Worker function for multi-process batch execution."""
    img_path, output_dir, category, patch_size, stride = args_tuple
    return extract_patches_from_image(img_path, output_dir, category, patch_size, stride)


def batch_patch_pipeline(
        source_root: str,
        target_root: str,
        patch_size: int = 32,
        stride: int = 16,
        num_workers: int = 4,
):
    """
    Multiprocess pipeline to traverse raw SEM image directories and generate patches.
    """
    source_path = Path(source_root)
    valid_exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    categories = ["crack", "intact"]

    tasks = []
    print(f"[INFO] Initializing patching pipeline | Patch Size: {patch_size}x{patch_size} | Stride: {stride}")

    for cat in categories:
        cat_dir = source_path / cat
        if not cat_dir.is_dir():
            print(f"[WARN] Category directory not found: {cat_dir}")
            continue

        img_list = []
        for ext in valid_exts:
            img_list.extend(glob.glob(str(cat_dir / ext)))

        print(f"[INFO] Found {len(img_list)} images for category '{cat}'")
        for p in img_list:
            tasks.append((p, target_root, cat, patch_size, stride))

    if not tasks:
        print("[ERROR] No input images discovered. Aborting pipeline.")
        return

    # Multiprocess worker pool for rapid I/O and patch extraction
    total_patches = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_category_worker, t) for t in tasks]
        for f in as_completed(futures):
            total_patches += f.result()

    print(f"[SUCCESS] Pipeline complete. Total patches generated: {total_patches}")
    print(f"[SUCCESS] Dataset stored at: {Path(target_root).resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SEM Micrograph Spatial Patching Utility")
    parser.add_argument("--source_dir", type=str, default="./raw_large_images", help="Path to raw micrographs")
    parser.add_argument("--target_dir", type=str, default="./data/train", help="Output directory for patches")
    parser.add_argument("--patch_size", type=int, default=32, help="Patch dimension (default: 32)")
    parser.add_argument("--stride", type=int, default=16, help="Sliding window stride (default: 16)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel processes")

    args = parser.parse_args()

    batch_patch_pipeline(
        source_root=args.source_dir,
        target_root=args.target_dir,
        patch_size=args.patch_size,
        stride=args.stride,
        num_workers=args.workers,
    )
