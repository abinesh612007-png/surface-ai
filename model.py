"""
model.py — SurfaceAI AI inference module.
Loads a trained ResNet-18 checkpoint and performs defect classification.
"""

import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_NAMES = [
    "Crazing",
    "Inclusion",
    "Patches",
    "Pitted_surface",
    "Rolled-in_scale",
    "Scratches",
]

CLASS_DISPLAY = {
    "Crazing":         "Crazing",
    "Inclusion":       "Inclusion",
    "Patches":         "Patches",
    "Pitted_surface":  "Pitted Surface",
    "Rolled-in_scale": "Rolled-in Scale",
    "Scratches":       "Scratches",
}

MODEL_VERSION = "resnet18-neu-cls-v1"
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
CHECKPOINT_PATH = os.path.join(MODELS_DIR, "best_model.pth")

# NEU-CLS images are grayscale 200×200; we convert to RGB and resize to 224.
PREPROCESS = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),   # handle grayscale or RGB
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

_model = None          # singleton
_device = None
_model_loaded = False
_model_error: str = ""


# ---------------------------------------------------------------------------
# Model initialisation
# ---------------------------------------------------------------------------

def _build_model(num_classes: int = 6) -> nn.Module:
    """Return ResNet-18 with a replacement classifier head."""
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


def load_model() -> tuple[bool, str]:
    """
    Attempt to load model weights from CHECKPOINT_PATH.
    Returns (success, message).
    """
    global _model, _device, _model_loaded, _model_error

    if _model_loaded:
        return True, "Model already loaded."

    if not os.path.isfile(CHECKPOINT_PATH):
        _model_error = (
            f"Model checkpoint not found at {CHECKPOINT_PATH}. "
            "Please train the model using: python train.py"
        )
        return False, _model_error

    try:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = _build_model(num_classes=len(CLASS_NAMES))

        state = torch.load(CHECKPOINT_PATH, map_location=_device, weights_only=True)
        # Support both raw state_dict and checkpoint dicts
        if isinstance(state, dict) and "model_state_dict" in state:
            net.load_state_dict(state["model_state_dict"])
        else:
            net.load_state_dict(state)

        net.to(_device)
        net.eval()
        _model = net
        _model_loaded = True
        _model_error = ""
        return True, f"Model loaded on {_device}."
    except Exception as exc:
        _model_error = f"Failed to load model: {exc}"
        _model_loaded = False
        return False, _model_error


def get_model_status() -> dict:
    if not _model_loaded:
        load_model()
    return {
        "loaded": _model_loaded,
        "device": str(_device) if _device else "N/A",
        "version": MODEL_VERSION,
        "checkpoint": CHECKPOINT_PATH,
        "error": _model_error if not _model_loaded else None,
        "classes": [CLASS_DISPLAY.get(c, c) for c in CLASS_NAMES],
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(image_bytes: bytes) -> dict:
    """
    Run inference on raw image bytes.
    Returns a dict with predicted class, confidence, and per-class probabilities.
    Raises RuntimeError if the model is not loaded.
    """
    if not _model_loaded or _model is None:
        load_model()
    if not _model_loaded or _model is None:
        raise RuntimeError(
            "Model is not loaded. "
            "Train it first with 'python train.py', then restart the server."
        )

    # Decode image
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Cannot decode image: {exc}") from exc

    tensor = PREPROCESS(img).unsqueeze(0).to(_device)

    with torch.no_grad():
        logits = _model(tensor)                    # (1, num_classes)
        probs = torch.softmax(logits, dim=1)[0]    # (num_classes,)

    probs_np = probs.cpu().numpy().tolist()
    class_probs = {
        CLASS_DISPLAY.get(c, c): round(float(p), 6)
        for c, p in zip(CLASS_NAMES, probs_np)
    }

    top_idx = int(np.argmax(probs_np))
    predicted_raw = CLASS_NAMES[top_idx]
    predicted_display = CLASS_DISPLAY.get(predicted_raw, predicted_raw)
    confidence = probs_np[top_idx]

    # Top-3 predictions
    sorted_pairs = sorted(class_probs.items(), key=lambda x: x[1], reverse=True)
    top3 = [{"class": cls, "confidence": prob} for cls, prob in sorted_pairs[:3]]

    return {
        "predicted_class": predicted_display,
        "predicted_class_raw": predicted_raw,
        "confidence": round(confidence, 6),
        "class_probabilities": class_probs,
        "top3": top3,
        "model_version": MODEL_VERSION,
    }


# ---------------------------------------------------------------------------
# Grad-CAM (optional explainability)
# ---------------------------------------------------------------------------

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


def grad_cam(image_bytes: bytes, target_class_idx: int | None = None) -> bytes | None:
    """
    Compute a Grad-CAM heatmap overlay.
    Returns JPEG bytes of the overlay, or None if unavailable.
    This is an explanatory visualisation only — not a precise defect boundary.
    """
    if not _model_loaded:
        load_model()
    if not _model_loaded or _model is None or not _CV2_AVAILABLE:
        return None

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = PREPROCESS(img).unsqueeze(0).to(_device)
        tensor.requires_grad_(True)

        # Hook into the last conv layer
        activations = []
        gradients = []

        def fwd_hook(_, __, output):
            activations.append(output.detach())

        def bwd_hook(_, grad_in, grad_out):
            gradients.append(grad_out[0].detach())

        layer = _model.layer4
        fwd_h = layer.register_forward_hook(fwd_hook)
        bwd_h = layer.register_full_backward_hook(bwd_hook)

        output = _model(tensor)
        if target_class_idx is None:
            target_class_idx = int(output.argmax(dim=1).item())

        _model.zero_grad()
        output[0, target_class_idx].backward()

        fwd_h.remove()
        bwd_h.remove()

        act = activations[0].squeeze(0)        # (C, H, W)
        grad = gradients[0].squeeze(0)         # (C, H, W)
        weights = grad.mean(dim=(1, 2))        # (C,)
        cam = torch.relu((weights[:, None, None] * act).sum(dim=0))
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = (cam * 255).astype(np.uint8)

        import cv2 as cv
        orig = np.array(img.resize((224, 224)))
        cam_resized = cv.resize(cam, (224, 224))
        heatmap = cv.applyColorMap(cam_resized, cv.COLORMAP_JET)
        heatmap_rgb = cv.cvtColor(heatmap, cv.COLOR_BGR2RGB)
        overlay = cv.addWeighted(orig, 0.6, heatmap_rgb, 0.4, 0)
        _, buf = cv.imencode(".jpg", cv.cvtColor(overlay, cv.COLOR_RGB2BGR))
        return buf.tobytes()
    except Exception:
        return None
