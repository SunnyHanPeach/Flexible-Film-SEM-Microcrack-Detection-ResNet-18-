"""
eval_blind_detection.py
High-Throughput Full-Field Microcrack Blind Detection and Damage Quantification.

Author: Sun Han
Description:
    Performs batch-accelerated sliding window inference on large-scale SEM micrographs.
    Quantifies surface micro-damage ratio and localizes stress-induced crack clusters
    using deep residual feature representations with high statistical specificity.
"""

import os
import argparse
import time
from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms


def build_inference_model(model_path: str, device: torch.device, num_classes: int = 2) -> nn.Module:
    """Initialize model architecture and load trained state dictionary."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"[ERROR] Model checkpoint not found at: {model_path}")

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


class PatchInferenceEngine:
    """
    High-performance batched inference pipeline for spatial scanning across large FOV.
    """

    def __init__(self, model: nn.Module, device: torch.device, batch_size: int = 64):
        self.model = model
        self.device = device
        self.batch_size = batch_size

        # Standard ImageNet normalization parameters matching training pipeline
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1).to(device)

    def process_batch(self, batch_tensors: List[torch.Tensor]) -> np.ndarray:
        """Run batched forward pass and return crack probabilities."""
        # Concatenate into (B, 3, H, W)
        x = torch.stack(batch_tensors, dim=0).to(self.device)
        x = (x / 255.0 - self.mean) / self.std

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            # Class 0 corresponds to 'crack'
            crack_probs = probs[:, 0].cpu().numpy()

        return crack_probs


def run_blind_detection(
        image_path: str,
        model_path: str,
        output_path: str,
        patch_size: int = 32,
        stride: int = 16,
        conf_threshold: float = 0.80,
        batch_size: int = 64,
) -> Tuple[int, float]:
    """
    Execute full-micrograph sliding scan with batched tensor streaming.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_inference_model(model_path, device)
    engine = PatchInferenceEngine(model, device, batch_size=batch_size)

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"[ERROR] Failed to read image from {image_path}")
        return 0, 0.0

    h, w, _ = img.shape
    stem = Path(image_path).name
    print(f"[INFO] Scanning micrograph: {stem} | Dimensions: {w}x{h} | Device: {device.type}")

    t0 = time.time()

    # Pre-collect coordinates to construct sliding-window grid
    y_steps = list(range(0, h - patch_size + 1, stride))
    x_steps = list(range(0, w - patch_size + 1, stride))
    total_patches = len(y_steps) * len(x_steps)

    batch_tensors = []
    batch_coords = []
    positive_boxes = []

    # Stream patches through batch pipeline
    for y in y_steps:
        for x in x_steps:
            patch_bgr = img[y: y + patch_size, x: x + patch_size]
            # Convert BGR (OpenCV) -> RGB and permute to (3, H, W)
            patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
            patch_t = torch.from_numpy(patch_rgb).permute(2, 0, 1).float()

            batch_tensors.append(patch_t)
            batch_coords.append((x, y))

            if len(batch_tensors) >= batch_size:
                probs = engine.process_batch(batch_tensors)
                for (cx, cy), p in zip(batch_coords, probs):
                    if p >= conf_threshold:
                        positive_boxes.append((cx, cy, cx + patch_size, cy + patch_size))
                batch_tensors.clear()
                batch_coords.clear()

    # Flush remaining patches in buffer
    if batch_tensors:
        probs = engine.process_batch(batch_tensors)
        for (cx, cy), p in zip(batch_coords, probs):
            if p >= conf_threshold:
                positive_boxes.append((cx, cy, cx + patch_size, cy + patch_size))

    elapsed = time.time() - t0
    damage_ratio = (len(positive_boxes) / total_patches) * 100.0 if total_patches > 0 else 0.0

    # Annotate spatial bounding boxes and diagnostic statistics
    annotated_img = img.copy()
    for x1, y1, x2, y2 in positive_boxes:
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 1)

    label_text = f"Damage Ratio: {damage_ratio:.2f}% | Conf >= {conf_threshold} | {stem}"
    cv2.putText(annotated_img, label_text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, annotated_img)

    print(f"[RESULT] Patches Evaluated: {total_patches} | Detections: {len(positive_boxes)}")
    print(
        f"[RESULT] Damage Ratio: {damage_ratio:.2f}% | Time Elapsed: {elapsed:.2f}s ({total_patches / elapsed:.1f} fps)")
    print(f"[OUTPUT] Visualization saved to: {output_path}")

    return len(positive_boxes), damage_ratio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SEM Microcrack Dual-Blind Detection Benchmark")
    parser.add_argument("--model_path", type=str, default="./models/resnet18_crack_model.pth", help="Trained weights")
    parser.add_argument("--crack_image", type=str, default="./raw_large_images/test_hairline_crack.png",
                        help="Damaged SEM sample")
    parser.add_argument("--control_image", type=str, default="./raw_large_images/test_intact_clean.png",
                        help="Intact control sample")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory for output visualizations")
    parser.add_argument("--conf_thresh", type=float, default=0.80, help="Damage confidence threshold")
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size")

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("   Quantitative Stress Microcrack Screening Pipeline (Batch Mode)")
    print("=" * 65)

    # Benchmark Test 1: Bending-induced crack network (QDs-OA)
    run_blind_detection(
        image_path=args.crack_image,
        model_path=args.model_path,
        output_path=str(out_dir / "eval_crack_damaged.png"),
        conf_threshold=args.conf_thresh,
        batch_size=args.batch_size,
    )

    print("-" * 65)

    # Benchmark Test 2: Mechanically robust negative control (QDs-DDTC)
    run_blind_detection(
        image_path=args.control_image,
        model_path=args.model_path,
        output_path=str(out_dir / "eval_control_intact.png"),
        conf_threshold=args.conf_thresh,
        batch_size=args.batch_size,
    )
