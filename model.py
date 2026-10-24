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

        try:
            state = torch.load(CHECKPOINT_PATH, map_location=_device, weights_only=True)
        except Exception:
            state = torch.load(CHECKPOINT_PATH, map_location=_device, weights_only=False)
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


DEFECT_METADATA = {
    "Crazing": {
        "display": "Crazing",
        "short_label": "Crack",
        "color": "#00E5FF",       # Cyan (Crack)
        "description": "Network of fine surface fissures or fracture cracks",
    },
    "Inclusion": {
        "display": "Inclusion",
        "short_label": "Speck",
        "color": "#FF1744",       # Red / Coral (Speck)
        "description": "Foreign non-metallic particle embedded into material surface",
    },
    "Patches": {
        "display": "Patches",
        "short_label": "Stain",
        "color": "#FF9100",       # Amber / Orange (Stain)
        "description": "Surface discoloration, oxidation patch, or stain",
    },
    "Pitted_surface": {
        "display": "Pitted Surface",
        "short_label": "Pitted",
        "color": "#E040FB",       # Bright Purple
        "description": "Localized pitting cavities, pinholes, and micro-voids",
    },
    "Rolled-in_scale": {
        "display": "Rolled-in Scale",
        "short_label": "Scale",
        "color": "#00E676",       # Neon Green
        "description": "Oxide scale pressed into surface during rolling",
    },
    "Scratches": {
        "display": "Scratches",
        "short_label": "Scratch",
        "color": "#2979FF",       # Electric Blue (Scratch)
        "description": "Linear surface abrasion, scoring, or friction groove",
    },
}

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


def grad_cam(image_bytes: bytes, target_class_idx: int | None = None) -> bytes | None:
    """
    Compute a Grad-CAM heatmap overlay.
    Returns JPEG bytes of the overlay, or None if unavailable.
    """
    res = analyze_defect(image_bytes, include_gradcam=True)
    if res.get("gradcam_bytes"):
        return res["gradcam_bytes"]
    return None


def analyze_defect(image_bytes: bytes, include_gradcam: bool = True) -> dict:
    """
    Full inference and damage accuracy analysis:
    - Runs ResNet classification
    - Computes Grad-CAM activation heatmap
    - Extracts defect bounding boxes (normalized [x, y, w, h])
    - Calculates damage percentage and severity
    - Computes detection accuracy
    """
    if not _model_loaded or _model is None:
        load_model()
    if not _model_loaded or _model is None:
        raise RuntimeError(
            "Model is not loaded. Train it first with 'python train.py'."
        )

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Cannot decode image: {exc}") from exc

    tensor = PREPROCESS(img).unsqueeze(0).to(_device)
    tensor.requires_grad_(True)

    # Hook into last conv layer for Grad-CAM
    activations = []
    gradients = []

    def fwd_hook(_, __, output):
        activations.append(output)

    def bwd_hook(_, grad_in, grad_out):
        gradients.append(grad_out[0])

    layer = _model.layer4
    fwd_h = layer.register_forward_hook(fwd_hook)
    bwd_h = layer.register_full_backward_hook(bwd_hook)

    try:
        output = _model(tensor)
        probs = torch.softmax(output, dim=1)[0]
        top_idx = int(output.argmax(dim=1).item())

        _model.zero_grad()
        output[0, top_idx].backward()
    finally:
        fwd_h.remove()
        bwd_h.remove()

    probs_np = probs.detach().cpu().numpy().tolist()
    class_probs = {
        CLASS_DISPLAY.get(c, c): round(float(p), 6)
        for c, p in zip(CLASS_NAMES, probs_np)
    }

    predicted_raw = CLASS_NAMES[top_idx]
    predicted_display = CLASS_DISPLAY.get(predicted_raw, predicted_raw)
    confidence = float(probs_np[top_idx])

    sorted_pairs = sorted(class_probs.items(), key=lambda x: x[1], reverse=True)
    top3 = [{"class": cls, "confidence": prob} for cls, prob in sorted_pairs[:3]]

    meta = DEFECT_METADATA.get(predicted_raw, {
        "display": predicted_display,
        "short_label": predicted_display,
        "color": "#F5A623",
        "description": "Surface irregularity detected"
    })

    # Grad-CAM CAM map calculation
    act = activations[0].squeeze(0).detach()     # (C, H, W)
    grad = gradients[0].squeeze(0).detach()      # (C, H, W)
    weights = grad.mean(dim=(1, 2))              # (C,)
    cam = torch.relu((weights[:, None, None] * act).sum(dim=0)).cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam_u8 = (cam * 255).astype(np.uint8)

    gradcam_bytes = None
    boxes = []
    damage_percentage = 0.0

    if _CV2_AVAILABLE:
        cam_resized = cv2.resize(cam_u8, (224, 224))
        # Threshold for defect region
        thresh = cv2.threshold(cam_resized, 110, 255, cv2.THRESH_BINARY)[1]
        damage_pixels = int(np.count_nonzero(thresh))
        damage_percentage = round((damage_pixels / (224 * 224)) * 100, 1)

        # Extract defect bounding contours
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        for c in cnts[:3]:
            if cv2.contourArea(c) >= 70:
                bx, by, bw, bh = cv2.boundingRect(c)
                # Add slight padding
                pad = 4
                bx0 = max(0, bx - pad)
                by0 = max(0, by - pad)
                bx1 = min(224, bx + bw + pad)
                by1 = min(224, by + bh + pad)
                bw_pad = bx1 - bx0
                bh_pad = by1 - by0

                boxes.append({
                    "x": round(bx0 / 224.0, 4),
                    "y": round(by0 / 224.0, 4),
                    "w": round(bw_pad / 224.0, 4),
                    "h": round(bh_pad / 224.0, 4),
                    "label": meta["short_label"],
                    "accuracy": round(confidence * 100, 1),
                    "color": meta["color"]
                })

        # Fallback if no clean contour but confidence is high
        if not boxes and confidence >= 0.45:
            _, max_val, _, max_loc = cv2.minMaxLoc(cam_resized)
            if max_val > 100:
                cx, cy = max_loc
                w_px = 64
                h_px = 64
                bx = max(0, min(224 - w_px, cx - w_px // 2))
                by = max(0, min(224 - h_px, cy - h_px // 2))
                boxes.append({
                    "x": round(bx / 224.0, 4),
                    "y": round(by / 224.0, 4),
                    "w": round(w_px / 224.0, 4),
                    "h": round(h_px / 224.0, 4),
                    "label": meta["short_label"],
                    "accuracy": round(confidence * 100, 1),
                    "color": meta["color"]
                })

        if include_gradcam:
            try:
                orig = np.array(img.resize((224, 224)))
                heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
                heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                overlay = cv2.addWeighted(orig, 0.6, heatmap_rgb, 0.4, 0)
                _, buf = cv2.imencode(".jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                gradcam_bytes = buf.tobytes()
            except Exception:
                gradcam_bytes = None

    # Severity evaluation
    if damage_percentage >= 22.0 or confidence >= 0.88:
        severity = "Critical"
    elif damage_percentage >= 8.0 or confidence >= 0.65:
        severity = "Moderate"
    elif damage_percentage >= 1.5 or confidence >= 0.40:
        severity = "Minor"
    else:
        severity = "Negligible"

    defect_status = "DEFECT DETECTED" if confidence >= 0.40 else "SURFACE CLEAR"

    return {
        "predicted_class": predicted_display,
        "predicted_class_raw": predicted_raw,
        "confidence": round(confidence, 6),
        "accuracy_percent": round(confidence * 100, 1),
        "damage_percentage": damage_percentage,
        "damage_severity": severity,
        "defect_status": defect_status,
        "defect_label": meta["short_label"],
        "defect_color": meta["color"],
        "defect_description": meta["description"],
        "defect_boxes": boxes,
        "class_probabilities": class_probs,
        "top3": top3,
        "model_version": MODEL_VERSION,
        "gradcam_bytes": gradcam_bytes,
    }

