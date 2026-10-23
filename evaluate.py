"""
evaluate.py — Standalone evaluation script for SurfaceAI.
Run after training to generate detailed metrics on the test set.

Usage:
    python evaluate.py --data_dir data/NEU-CLS --checkpoint models/best_model.pth
"""

import argparse
import json
from pathlib import Path

import torch
from torchvision import datasets, transforms, models
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support
)

EVAL_TF = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

SEED = 42


def build_model(num_classes):
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    checkpoint = Path(args.checkpoint)

    if not data_dir.is_dir():
        print(f"ERROR: Dataset directory not found: {data_dir}")
        return
    if not checkpoint.is_file():
        print(f"ERROR: Checkpoint not found: {checkpoint}")
        return

    # Load dataset
    full_ds = datasets.ImageFolder(str(data_dir), transform=EVAL_TF)
    class_names = full_ds.classes
    num_classes = len(class_names)
    n = len(full_ds)
    n_test = int(n * 0.15)
    n_val  = int(n * 0.15)
    n_train= n - n_val - n_test

    gen = torch.Generator().manual_seed(SEED)
    _, _, test_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val, n_test], generator=gen
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=32, shuffle=False, num_workers=0
    )

    # Load model
    model = build_model(num_classes).to(device)
    state = torch.load(str(checkpoint), map_location=device, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    print(f"Loaded model from {checkpoint} on {device}")

    # Inference
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs.tolist())

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    cm = confusion_matrix(all_labels, all_preds)
    prec, rec, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=list(range(num_classes))
    )

    print(f"\n=== Test Set Evaluation ({len(all_labels)} samples) ===")
    print(f"Overall Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("\n=== Classification Report ===")
    print(report)
    print("=== Confusion Matrix ===")
    print(cm)

    # Save JSON summary
    summary = {
        "accuracy": round(acc, 6),
        "classes": {},
        "test_size": len(all_labels),
    }
    for i, cls in enumerate(class_names):
        summary["classes"][cls] = {
            "precision": round(float(prec[i]), 6),
            "recall":    round(float(rec[i]), 6),
            "f1_score":  round(float(f1[i]), 6),
            "support":   int(support[i]),
        }
    out_path = Path("models") / "eval_summary.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nEvaluation summary saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="data/NEU-CLS")
    parser.add_argument("--checkpoint", default="models/best_model.pth")
    evaluate(parser.parse_args())
