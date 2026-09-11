"""
benchmark_evaluation.py
Comprehensive Quantitative Evaluation & Ablation Benchmark for SEM Microcrack Models.

Author: Sun Han
Description:
    Evaluates fine-tuned neural network checkpoints on unseen test partitions.
    Computes confusion matrices and full diagnostic metrics (Accuracy, Precision,
    Recall/Sensitivity, Specificity, and F1-Score) to validate statistical reliability.
"""

import os
import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


class BaselineShallowCNN(nn.Module):
    """
    Lightweight 2-stage convolutional network used as a baseline for ablation comparison.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 2):
        super(BaselineShallowCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def load_model_for_eval(arch: str, model_path: str, device: torch.device, num_classes: int = 2) -> nn.Module:
    """Initialize specified architecture and load state dictionary."""
    if arch.lower() == "resnet18":
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )
    elif arch.lower() == "baseline":
        model = BaselineShallowCNN(in_channels=3, num_classes=num_classes)
    else:
        raise ValueError(f"[ERROR] Unsupported architecture: {arch}")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"[ERROR] Model weights not found at: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def compute_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Calculate confusion matrix and clinical/material diagnostic performance metrics.
    Index 0: crack (Positive), Index 1: intact (Negative)
    """
    tp = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 1) & (y_pred == 0))
    fn = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 1) & (y_pred == 1))

    cm = np.array([[tp, fp], [fn, tn]], dtype=int)

    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8) * 100.0
    precision = tp / (tp + fp + 1e-8) * 100.0
    recall = tp / (tp + fn + 1e-8) * 100.0  # Sensitivity
    specificity = tn / (tn + fp + 1e-8) * 100.0
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    metrics = {
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1_Score": f1,
    }
    return cm, metrics


def evaluate_test_partition(
    model_path: str,
    test_dir: str,
    arch: str = "resnet18",
    batch_size: int = 32
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print(f"      Model Benchmark Evaluation | Architecture: {arch.upper()}")
    print(f"      Compute Device: {device} | Test Source: {Path(test_dir).resolve()}")
    print("=" * 65)

    test_transforms = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"[DATA] Evaluation samples: {len(test_dataset)} | Mapping: {test_dataset.class_to_idx}")

    model = load_model_for_eval(arch, model_path, device)

    y_true_list = []
    y_pred_list = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            y_true_list.extend(labels.numpy())
            y_pred_list.extend(preds.cpu().numpy())

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    cm, metrics = compute_comprehensive_metrics(y_true, y_pred)

    # Formal Diagnostic Report Output
    print("\n" + "-" * 65)
    print("                Quantitative Diagnostic Report                 ")
    print("-" * 65)
    print(f"  Overall Accuracy (ACC)      : {metrics['Accuracy']:.2f}%")
    print(f"  Macro F1-Score (F1)         : {metrics['F1_Score']:.2f}%")
    print(f"  Crack Recall / Sensitivity  : {metrics['Recall']:.2f}% (Target: Minimize False Negatives)")
    print(f"  Pristine Specificity (TNR)  : {metrics['Specificity']:.2f}% (Target: Minimize False Alarms)")
    print(f"  Precision (PPV)             : {metrics['Precision']:.2f}%")
    print("-" * 65)
    print("Confusion Matrix Structure:")
    print("                 Pred Crack (0)   Pred Intact (1)")
    print(f"  True Crack (0)     {cm[0, 0]:6d}          {cm[1, 0]:6d}   [TP | FN]")
    print(f"  True Intact(1)     {cm[0, 1]:6d}          {cm[1, 1]:6d}   [FP | TN]")
    print("-" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Set Quantitative Benchmark Tool")
    parser.add_argument("--model_path", type=str, default="./models/resnet18_crack_model.pth", help="Path to checkpoint")
    parser.add_argument("--test_dir", type=str, default="./data/val", help="Path to unseen test directory")
    parser.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "baseline"], help="Network backbone")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")

    args = parser.parse_args()

    evaluate_test_partition(
        model_path=args.model_path,
        test_dir=args.test_dir,
        arch=args.arch,
        batch_size=args.batch_size
    )
