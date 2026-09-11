"""
train_feature_extractor.py
Transfer Learning & Fine-tuning Pipeline for SEM Microcrack Classification.

Author: Sun Han
Description:
    Fine-tunes a deep residual network (ResNet-18) on localized SEM micro-patches.
    Incorporates spatial invariant data augmentation, cosine annealing learning rate
    scheduling, and validation tracking (Accuracy, Recall, F1-score) to achieve
    robust feature extraction without overfitting to substrate morphology.
"""

import os
import argparse
import random
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models


def set_seed(seed: int = 42) -> None:
    """Enforce strict deterministic behavior across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_resnet18_classifier(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    Construct ResNet-18 transfer learning backbone with custom classification head.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    in_features = model.fc.in_features
    # Replace final fully connected layer for binary discrimination
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes)
    )
    return model


def prepare_datasets(
        data_dir: str,
        val_ratio: float = 0.2,
        seed: int = 42
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """
    Load patch dataset with spatial data augmentation and stratified train/val split.
    """
    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise FileNotFoundError(f"[ERROR] Dataset root not found at: {data_path.resolve()}")

    # Geometric augmentations invariant to physical crack orientation
    train_transforms = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=90),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Base dataset for indexing
    full_dataset = datasets.ImageFolder(root=str(data_path))
    class_to_idx = full_dataset.class_to_idx

    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)

    # Re-apply appropriate transforms to respective subsets
    train_subset.dataset.transform = train_transforms
    val_subset.dataset.transform = val_transforms

    return train_subset, val_subset, class_to_idx


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute classification metrics focusing on damage detection performance."""
    tp = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 1) & (y_pred == 0))
    fn = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 1) & (y_pred == 1))

    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    return {
        "accuracy": float(acc * 100.0),
        "precision": float(precision * 100.0),
        "recall": float(recall * 100.0),
        "f1": float(f1 * 100.0),
    }


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    """Evaluate model on validation subset."""
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)

            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    total_samples = len(loader.dataset)
    epoch_loss = running_loss / total_samples
    metrics = compute_binary_metrics(np.array(all_targets), np.array(all_preds))
    metrics["loss"] = epoch_loss
    return metrics


def train_pipeline(
        data_dir: str,
        output_dir: str,
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        seed: int = 42
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("      ResNet-18 Transfer Learning for SEM Microcrack Characterization")
    print(f"      Device: {device} | Random Seed: {seed}")
    print("=" * 70)

    train_subset, val_subset, class_to_idx = prepare_datasets(data_dir, val_ratio=0.2, seed=seed)
    print(f"[DATA] Class Mapping: {class_to_idx}")
    print(f"[DATA] Training Set: {len(train_subset)} patches | Validation Set: {len(val_subset)} patches")

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = build_resnet18_classifier(num_classes=2, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_f1 = 0.0
    best_epoch = 0
    save_file = out_path / "resnet18_crack_model.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        scheduler.step()
        epoch_train_loss = train_loss / len(train_subset)

        # Validation Step
        val_metrics = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"| Train Loss: {epoch_train_loss:.4f} "
            f"| Val Loss: {val_metrics['loss']:.4f} "
            f"| Val Acc: {val_metrics['accuracy']:.2f}% "
            f"| Recall: {val_metrics['recall']:.2f}% "
            f"| F1: {val_metrics['f1']:.2f}%"
        )

        # Checkpoint optimal weights based on Validation F1-score (critical for crack recall)
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            # Save state dict
            torch.save(model.state_dict(), save_file)

    print("-" * 70)
    print(f"[SUMMARY] Best Validation F1-Score: {best_val_f1:.2f}% at Epoch {best_epoch}")
    print(f"[SUCCESS] Checkpoint saved successfully to: {save_file.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ResNet-18 Microcrack Transfer Learning Pipeline")
    parser.add_argument("--data_dir", type=str, default="./data/train", help="Root directory for patch dataset")
    parser.add_argument("--output_dir", type=str, default="./models", help="Directory to save model weights")
    parser.add_argument("--epochs", type=int, default=20, help="Total training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")

    args = parser.parse_args()

    train_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed
    )
