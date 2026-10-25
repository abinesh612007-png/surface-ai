"""
train.py — SurfaceAI model training script.

Usage:
    python train.py --data_dir data/NEU-CLS --epochs 30 --batch_size 32

The NEU-CLS dataset should be arranged so that each class is a subdirectory
containing its images:
    data/NEU-CLS/
        Crazing/        (300 images)
        Inclusion/      (300 images)
        Patches/        (300 images)
        Pitted_surface/ (300 images)
        Rolled-in_scale/(300 images)
        Scratches/      (300 images)

Download: https://www.kaggle.com/datasets/fantacher/neu-metal-surface-defects-data
"""

import argparse
import os
import random
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

# ── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Transforms ──────────────────────────────────────────────────────────────
TRAIN_TF = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), value="random"),
])

EVAL_TF = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Model factory ────────────────────────────────────────────────────────────
def build_model(num_classes: int, arch: str = "resnet18", pretrained: bool = True) -> nn.Module:
    """Builds deep CNN backbone with customized head."""
    if arch == "resnet34":
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet34(weights=weights)
        for p in net.parameters():
            p.requires_grad = False
        net.fc = nn.Linear(net.fc.in_features, num_classes)
    elif arch == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet50(weights=weights)
        for p in net.parameters():
            p.requires_grad = False
        net.fc = nn.Linear(net.fc.in_features, num_classes)
    elif arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.efficientnet_b0(weights=weights)
        for p in net.parameters():
            p.requires_grad = False
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, num_classes)
    else:
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet18(weights=weights)
        for p in net.parameters():
            p.requires_grad = False
        net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


# ── Training ─────────────────────────────────────────────────────────────────
def train(args):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Validate dataset path
    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"ERROR: data directory not found: {data_dir}")
        print("Please download NEU-CLS and arrange as described in train.py docstring.")
        return

    # Load full dataset without augmentation first to discover classes
    full_dataset = datasets.ImageFolder(str(data_dir), transform=EVAL_TF)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")
    print(f"Total images: {len(full_dataset)}")

    # Stratified split: 70% train, 15% val, 15% test
    n = len(full_dataset)
    n_test = int(n * 0.15)
    n_val = int(n * 0.15)
    n_train = n - n_val - n_test

    # Use a generator for reproducibility
    gen = torch.Generator().manual_seed(SEED)
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test], generator=gen
    )

    # Apply augmentation only to the training subset
    class AugmentedSubset(torch.utils.data.Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            img, label = self.subset[idx]
            # img is already a tensor from EVAL_TF; re-load from path instead
            path, _ = self.subset.dataset.samples[self.subset.indices[idx]]
            from PIL import Image
            raw = Image.open(path)
            return self.transform(raw), label

    train_aug = AugmentedSubset(train_ds, TRAIN_TF)

    train_loader = DataLoader(
        train_aug, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    model = build_model(num_classes, arch=args.arch, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Phase 1: Train only the classifier head
    head_params = model.classifier.parameters() if hasattr(model, "classifier") else model.fc.parameters()
    optimizer = torch.optim.AdamW(head_params, lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs // 2
    )

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    best_path = models_dir / "best_model.pth"

    history = []
    best_val_acc = 0.0
    patience_counter = 0

    print(f"\nPhase 1 — Training head for {args.epochs // 2} epochs …")
    phase1_epochs = args.epochs // 2

    for epoch in range(1, args.epochs + 1):
        # Unfreeze backbone after phase 1
        if epoch == phase1_epochs + 1:
            print("\nPhase 2 — Fine-tuning full network …")
            for p in model.parameters():
                p.requires_grad = True
            optimizer = torch.optim.Adam(
                model.parameters(), lr=1e-4, weight_decay=1e-4
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - phase1_epochs
            )

        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            train_correct += (out.argmax(1) == labels).sum().item()
            train_total += imgs.size(0)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                loss = criterion(out, labels)
                val_loss += loss.item() * imgs.size(0)
                val_correct += (out.argmax(1) == labels).sum().item()
                val_total += imgs.size(0)

        t_acc = train_correct / train_total
        v_acc = val_correct / val_total
        t_loss = train_loss / train_total
        v_loss = val_loss / val_total

        scheduler.step()

        rec = {
            "epoch": epoch,
            "train_loss": round(t_loss, 5),
            "train_acc": round(t_acc, 5),
            "val_loss": round(v_loss, 5),
            "val_acc": round(v_acc, 5),
        }
        history.append(rec)
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss {t_loss:.4f} Acc {t_acc:.4f} | "
            f"Val Loss {v_loss:.4f} Acc {v_acc:.4f}"
        )

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ Saved best model (val_acc={v_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    # Save training history
    with open(models_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ── Test evaluation ───────────────────────────────────────────────────
    print(f"\nLoading best model from {best_path} for test evaluation …")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    from sklearn.metrics import classification_report, confusion_matrix
    report = classification_report(
        all_labels, all_preds, target_names=class_names
    )
    cm = confusion_matrix(all_labels, all_preds)
    print("\n=== Classification Report (Test Set) ===")
    print(report)
    print("=== Confusion Matrix ===")
    print(cm)

    with open(models_dir / "evaluation_report.txt", "w") as f:
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))

    # Save class mapping for inference
    mapping = {i: name for i, name in enumerate(class_names)}
    with open(models_dir / "class_mapping.json", "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")
    print(f"Model saved to: {best_path}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SurfaceAI Neural Defect Classifier")
    parser.add_argument("--arch", choices=["resnet18", "resnet34", "resnet50", "efficientnet_b0"],
                        default="resnet18", help="Deep learning architecture backbone")
    parser.add_argument("--data_dir", default="data/NEU-CLS",
                        help="Path to NEU-CLS dataset root")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Total training epochs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0,
                        help="DataLoader workers (0 = main thread, safe on Windows)")
    parser.add_argument("--patience", type=int, default=8,
                        help="Early-stopping patience epochs")
    args = parser.parse_args()
    train(args)
