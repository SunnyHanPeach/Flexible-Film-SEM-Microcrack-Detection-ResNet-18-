"""
generate_stress_heatmap.py
Dense Spatial Probability Mapping & Gaussian Continuous Stress Field Characterization.

Author: Sun Han
Description:
    Implements a dense sliding-window probability probe combined with separable 2D
    Gaussian continuous smoothing to circumvent Grad-CAM spatial resolution collapse.
    Maps discrete CNN classifications into a continuous damage probability field for
    flexible semiconductor thin films under mechanical fatigue.
"""

import os
import argparse
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models


def load_feature_extractor(model_path: str, device: torch.device, num_classes: int = 2) -> nn.Module:
    """Instantiate ResNet-18 architecture and load fine-tuned weights."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"[ERROR] Checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def apply_separable_gaussian_smoothing(
        prob_field: np.ndarray,
        kernel_size: int = 31,
        sigma: float = 0.0
) -> np.ndarray:
    """
    Apply 2D Gaussian spatial smoothing utilizing separable 1D kernels.
    Complexity: Reduces computation from O(H * W * K^2) to O(H * W * 2K).
    """
    # cv2.GaussianBlur internally optimizes via separable 1D row-column passes
    return cv2.GaussianBlur(prob_field, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)


def render_colorbar(height: int, width: int = 40) -> np.ndarray:
    """Generate a vertical JET colormap calibration bar (0.0 to 1.0)."""
    gradient = np.linspace(255, 0, height, dtype=np.uint8).reshape(-1, 1)
    gradient = np.repeat(gradient, width, axis=1)
    colorbar = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)

    # Add scalar tick marks
    cv2.putText(colorbar, "1.0", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(colorbar, "0.5", (4, height // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(colorbar, "0.0", (4, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return colorbar


def generate_stress_field_heatmap(
        image_path: str,
        model_path: str,
        output_path: str,
        patch_size: int = 32,
        stride: int = 8,
        batch_size: int = 64,
        gaussian_kernel: int = 31,
        crack_class_idx: int = 0,
        absolute_scale: bool = True
) -> np.ndarray:
    """
    Execute dense sliding-window probe across SEM FOV and project continuous stress damage field.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_feature_extractor(model_path, device)

    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"[ERROR] Failed to load image from: {image_path}")

    h, w, _ = img_bgr.shape
    stem = Path(image_path).name
    print(f"[INFO] Processing SEM Micrograph: {stem} | Resolution: {w}x{h} | Stride: {stride}")

    # Coordinate grid generation
    y_coords = list(range(0, h - patch_size + 1, stride))
    x_coords = list(range(0, w - patch_size + 1, stride))
    total_probes = len(y_coords) * len(x_coords)
    print(f"[INFO] Dense Probing Grid: {total_probes} patches | Batch Size: {batch_size}")

    # Accumulation buffers for spatial probability estimation
    prob_accumulator = np.zeros((h, w), dtype=np.float32)
    weight_accumulator = np.zeros((h, w), dtype=np.float32)

    # ImageNet normalization tensors for vector broadcasting
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1).to(device)

    t0 = time.time()
    batch_patches = []
    batch_coords = []

    # Stream dense sliding-window patches
    for y in y_coords:
        for x in x_coords:
            patch = img_bgr[y: y + patch_size, x: x + patch_size]
            # Convert BGR -> RGB and transfrom to (3, H, W)
            patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            patch_tensor = torch.from_numpy(patch_rgb).permute(2, 0, 1).float()

            batch_patches.append(patch_tensor)
            batch_coords.append((x, y))

            if len(batch_patches) >= batch_size:
                # Forward pass
                x_in = torch.stack(batch_patches, dim=0).to(device)
                x_in = (x_in / 255.0 - mean) / std
                with torch.no_grad():
                    logits = model(x_in)
                    probs = torch.softmax(logits, dim=1)[:, crack_class_idx].cpu().numpy()

                for (bx, by), p in zip(batch_coords, probs):
                    prob_accumulator[by: by + patch_size, bx: bx + patch_size] += p
                    weight_accumulator[by: by + patch_size, bx: bx + patch_size] += 1.0

                batch_patches.clear()
                batch_coords.clear()

    # Flush remaining tail batch
    if batch_patches:
        x_in = torch.stack(batch_patches, dim=0).to(device)
        x_in = (x_in / 255.0 - mean) / std
        with torch.no_grad():
            logits = model(x_in)
            probs = torch.softmax(logits, dim=1)[:, crack_class_idx].cpu().numpy()

        for (bx, by), p in zip(batch_coords, probs):
            prob_accumulator[by: by + patch_size, bx: bx + patch_size] += p
            weight_accumulator[by: by + patch_size, bx: bx + patch_size] += 1.0

    # Unbiased empirical expectation: P_avg(x, y)
    weight_accumulator = np.maximum(weight_accumulator, 1.0)
    raw_prob_field = prob_accumulator / weight_accumulator

    # Perform continuous space smoothing
    smoothed_field = apply_separable_gaussian_smoothing(raw_prob_field, kernel_size=gaussian_kernel)

    # Scientific Intensity Calibration:
    # Use absolute [0, 1] range to avoid artificially exaggerating noise in pristine control samples.
    if absolute_scale:
        calibrated_field = np.clip(smoothed_field, 0.0, 1.0)
    else:
        p_min, p_max = smoothed_field.min(), smoothed_field.max()
        calibrated_field = (smoothed_field - p_min) / (p_max - p_min + 1e-8)

    peak_prob = float(smoothed_field.max())
    mean_prob = float(smoothed_field.mean())

    # Multi-view composite rendering
    heatmap_uint8 = np.uint8(255 * calibrated_field)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Blend original morphology (55%) with stress probability field (45%)
    overlay = cv2.addWeighted(img_bgr, 0.55, colored_heatmap, 0.45, 0)

    # Attach vertical radiometric calibration scale
    cbar = render_colorbar(height=h, width=35)
    margin = np.zeros((h, 10, 3), dtype=np.uint8)

    # Triptych arrangement: Raw SEM | Continuous Stress Field | Superimposed XAI Diagnostic
    panel = np.hstack([img_bgr, margin, colored_heatmap, margin, overlay, margin, cbar])

    elapsed = time.time() - t0
    header_text = f"XAI Continuous Stress Field | Peak Damage Prob: {peak_prob:.2%} | Mean: {mean_prob:.2%}"
    cv2.putText(panel, header_text, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, panel)

    print(f"[SUCCESS] Heatmap generated in {elapsed:.2f}s | Peak: {peak_prob:.2%} | Output: {output_path}")
    return calibrated_field


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dense Continuous Probability Field Generator for SEM Microcracks")
    parser.add_argument("--model_path", type=str, default="./models/resnet18_crack_model.pth",
                        help="Model weights path")
    parser.add_argument("--crack_image", type=str, default="./raw_large_images/test_hairline_crack.png",
                        help="Bending crack sample")
    parser.add_argument("--control_image", type=str, default="./raw_large_images/test_intact_clean.png",
                        help="Intact control sample")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory for diagnostic heatmaps")
    parser.add_argument("--stride", type=int, default=8, help="Dense probing step size (default: 8)")
    parser.add_argument("--kernel_size", type=int, default=31, help="Gaussian smoothing kernel size (odd integer)")

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("   Continuous Mechanical Stress Field Reconstruction via Dense Probing")
    print("=" * 70)

    # 1. Evaluate Bending Microcrack Sample (High Stress Concentration Zone)
    generate_stress_field_heatmap(
        image_path=args.crack_image,
        model_path=args.model_path,
        output_path=str(out_dir / "heatmap_dense_crack_field.png"),
        stride=args.stride,
        gaussian_kernel=args.kernel_size,
    )

    print("-" * 70)

    # 2. Evaluate Intact Negative Control (Robustness & Specificity Verification)
    generate_stress_field_heatmap(
        image_path=args.control_image,
        model_path=args.model_path,
        output_path=str(out_dir / "heatmap_dense_control_field.png"),
        stride=args.stride,
        gaussian_kernel=args.kernel_size,
    )
