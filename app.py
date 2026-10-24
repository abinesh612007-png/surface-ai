"""
app.py — SurfaceAI Flask application entry point.
Run with: python app.py
"""

import os
import io
import csv
import json
import base64
import logging
from datetime import datetime, timezone
from pathlib import Path

from flask import (
    Flask, request, jsonify, send_file, render_template_string,
    abort, Response
)
from werkzeug.utils import secure_filename

import database as db
import analytics as an
from model import load_model, get_model_status, predict, grad_cam, analyze_defect, DEFECT_METADATA

# ── App setup ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("surfaceai")

BASE_DIR = Path(__file__).parent
INDEX_HTML = BASE_DIR / "index.html"

app = Flask(__name__, template_folder=str(BASE_DIR))
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB upload limit
app.config["SECRET_KEY"] = os.urandom(24)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
ALLOWED_MIME = {"image/jpeg", "image/png"}


def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def _extract_image_bytes(req) -> tuple[bytes | None, str | None, str | None]:
    """
    Extracts image bytes from either multipart/form-data or JSON base64.
    Returns (image_bytes, filename, error_message).
    """
    # 1. Multipart file upload
    if "image" in req.files:
        f = req.files["image"]
        if not f or not f.filename:
            return None, None, "No file selected."
        filename = secure_filename(f.filename)
        if not allowed_file(filename):
            return None, None, "Unsupported file type. Use JPG or PNG."
        b = f.read()
        if len(b) == 0:
            return None, None, "Uploaded file is empty."
        return b, filename, None

    # 2. JSON or form-encoded base64 payload
    data = req.get_json(silent=True) or req.form
    if data and "image" in data:
        raw_b64 = data["image"]
        if not raw_b64:
            return None, None, "Image data is empty."
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        try:
            b = base64.b64decode(raw_b64)
            filename = data.get("filename", f"camera_scan_{_now_utc_compact()}.jpg")
            return b, filename, None
        except Exception as exc:
            return None, None, f"Invalid base64 encoding: {exc}"

    return None, None, "No image field in request."


# ── Initialise DB and load model on startup ───────────────────────────────────
db.init_db()
ok, msg = load_model()
log.info(f"Model load: {msg}")


# ── Serve frontend ────────────────────────────────────────────────────────────
@app.route("/")
def index():
    html = INDEX_HTML.read_text(encoding="utf-8")
    return html, 200, {"Content-Type": "text/html; charset=utf-8"}


# ── Health ────────────────────────────────────────────────────────────────────
@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "timestamp": _now_utc()})


# ── Model status ──────────────────────────────────────────────────────────────
@app.route("/api/model-status")
def model_status():
    return jsonify(get_model_status())


# ── Inspect (upload + predict) ────────────────────────────────────────────────
@app.route("/api/inspect", methods=["POST"])
def inspect():
    image_bytes, filename, err = _extract_image_bytes(request)
    if err:
        return jsonify({"error": err}), 400

    # Verify the bytes are actually a decodable image
    try:
        from PIL import Image as PILImage
        PILImage.open(io.BytesIO(image_bytes)).verify()
    except Exception as exc:
        return jsonify({"error": f"File cannot be decoded as an image: {exc}"}), 400

    # Run inference & damage analysis
    try:
        result = analyze_defect(image_bytes, include_gradcam=True)
    except RuntimeError as exc:
        return jsonify({"error": str(exc), "model_unavailable": True}), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422
    except Exception as exc:
        log.exception("Inference error")
        return jsonify({"error": f"Inference failed: {exc}"}), 500

    # Persist inspection
    inspection_id = db.save_inspection(
        filename=filename,
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        class_probabilities=result["class_probabilities"],
        defect_status=result.get("defect_status"),
        model_version=result["model_version"],
        damage_percentage=result.get("damage_percentage", 0.0),
        damage_severity=result.get("damage_severity", "Minor"),
    )

    gradcam_b64 = None
    if result.get("gradcam_bytes"):
        gradcam_b64 = base64.b64encode(result["gradcam_bytes"]).decode("ascii")

    return jsonify({
        "inspection_id": inspection_id,
        "filename": filename,
        "predicted_class": result["predicted_class"],
        "confidence": result["confidence"],
        "accuracy_percent": result["accuracy_percent"],
        "damage_percentage": result["damage_percentage"],
        "damage_severity": result["damage_severity"],
        "defect_status": result["defect_status"],
        "defect_label": result["defect_label"],
        "defect_color": result["defect_color"],
        "defect_description": result["defect_description"],
        "defect_boxes": result["defect_boxes"],
        "class_probabilities": result["class_probabilities"],
        "top3": result["top3"],
        "model_version": result["model_version"],
        "gradcam": gradcam_b64,
        "timestamp": _now_utc(),
    })


# ── Live Camera Scanning Endpoint (High-Speed Frame Analysis) ─────────────────
@app.route("/api/camera/scan-frame", methods=["POST"])
def camera_scan_frame():
    """
    Analyzes a single camera video frame with bounding boxes and damage accuracy.
    Fast execution by skipping Grad-CAM rendering during continuous live feed.
    """
    image_bytes, filename, err = _extract_image_bytes(request)
    if err:
        return jsonify({"error": err}), 400

    try:
        result = analyze_defect(image_bytes, include_gradcam=False)
    except RuntimeError as exc:
        return jsonify({"error": str(exc), "model_unavailable": True}), 503
    except Exception as exc:
        return jsonify({"error": f"Camera frame scan error: {exc}"}), 500

    return jsonify({
        "predicted_class": result["predicted_class"],
        "confidence": result["confidence"],
        "accuracy_percent": result["accuracy_percent"],
        "damage_percentage": result["damage_percentage"],
        "damage_severity": result["damage_severity"],
        "defect_status": result["defect_status"],
        "defect_label": result["defect_label"],
        "defect_color": result["defect_color"],
        "defect_description": result["defect_description"],
        "defect_boxes": result["defect_boxes"],
        "class_probabilities": result["class_probabilities"],
        "top3": result["top3"],
        "timestamp": _now_utc(),
    })


# ── Camera Snapshot & Audit Endpoint ──────────────────────────────────────────
@app.route("/api/camera/snapshot", methods=["POST"])
def camera_snapshot():
    """
    Freezes a camera frame, computes full Grad-CAM explainability, damage accuracy,
    and logs the inspection to database history.
    """
    image_bytes, filename, err = _extract_image_bytes(request)
    if err:
        return jsonify({"error": err}), 400

    try:
        result = analyze_defect(image_bytes, include_gradcam=True)
    except Exception as exc:
        return jsonify({"error": f"Snapshot analysis failed: {exc}"}), 500

    snap_name = filename or f"camera_audit_{_now_utc_compact()}.jpg"
    inspection_id = db.save_inspection(
        filename=snap_name,
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        class_probabilities=result["class_probabilities"],
        defect_status=result.get("defect_status"),
        model_version=result["model_version"],
        damage_percentage=result.get("damage_percentage", 0.0),
        damage_severity=result.get("damage_severity", "Minor"),
    )

    gradcam_b64 = None
    if result.get("gradcam_bytes"):
        gradcam_b64 = base64.b64encode(result["gradcam_bytes"]).decode("ascii")

    return jsonify({
        "inspection_id": inspection_id,
        "filename": snap_name,
        "predicted_class": result["predicted_class"],
        "confidence": result["confidence"],
        "accuracy_percent": result["accuracy_percent"],
        "damage_percentage": result["damage_percentage"],
        "damage_severity": result["damage_severity"],
        "defect_status": result["defect_status"],
        "defect_label": result["defect_label"],
        "defect_color": result["defect_color"],
        "defect_description": result["defect_description"],
        "defect_boxes": result["defect_boxes"],
        "class_probabilities": result["class_probabilities"],
        "top3": result["top3"],
        "gradcam": gradcam_b64,
        "timestamp": _now_utc(),
    })


# ── Sample Surfaces for Camera Simulation ─────────────────────────────────────
@app.route("/api/camera/samples")
def camera_samples():
    """
    Returns 4 interactive sample defect materials (Tile, Fabric, Plastic part, Painted panel)
    with data URIs for camera simulator testing.
    """
    from PIL import Image as PILImage, ImageDraw

    def img_to_data_uri(img: PILImage.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    # 1. Tile with Crack (Crazing)
    img_tile = PILImage.new("RGB", (224, 224), color=(226, 222, 212))
    d_tile = ImageDraw.Draw(img_tile)
    for x in range(0, 224, 10):
        d_tile.line([(x, 0), (x, 224)], fill=(218, 214, 204), width=1)
    d_tile.rectangle([165, 145, 185, 165], fill=(205, 200, 190))
    d_tile.line([(110, 25), (107, 60), (115, 95), (112, 130), (118, 175), (114, 205)], fill=(32, 28, 24), width=3)

    # 2. Fabric with Stain (Patches)
    img_fabric = PILImage.new("RGB", (224, 224), color=(35, 55, 115))
    d_fabric = ImageDraw.Draw(img_fabric)
    for y in range(0, 224, 5):
        d_fabric.line([(0, y), (224, y)], fill=(45, 70, 135), width=1)
    for x in range(0, 224, 5):
        d_fabric.line([(x, 0), (x, 224)], fill=(45, 70, 135), width=1)
    d_fabric.ellipse([140, 25, 195, 80], fill=(16, 24, 52))
    d_fabric.line([(25, 140), (180, 128)], fill=(120, 140, 180), width=1)

    # 3. Plastic Part with Speck (Inclusion)
    img_plastic = PILImage.new("RGB", (224, 224), color=(175, 48, 28))
    d_plastic = ImageDraw.Draw(img_plastic)
    d_plastic.arc([35, 90, 115, 170], start=180, end=270, fill=(215, 140, 120), width=2)
    d_plastic.ellipse([45, 160, 75, 190], fill=(155, 40, 22))
    for sp in [(140, 60), (150, 75), (145, 90), (160, 70), (170, 85), (155, 100), (135, 80)]:
        d_plastic.ellipse([sp[0]-3, sp[1]-3, sp[0]+3, sp[1]+3], fill=(20, 15, 15))

    # 4. Painted Panel with Scratch (Scratches)
    img_painted = PILImage.new("RGB", (224, 224), color=(28, 62, 112))
    d_painted = ImageDraw.Draw(img_painted)
    d_painted.ellipse([160, 160, 185, 185], fill=(38, 75, 130))
    d_painted.ellipse([165, 165, 180, 180], fill=(22, 50, 95))
    d_painted.line([(25, 195), (170, 48)], fill=(215, 235, 255), width=3)

    return jsonify([
        {"id": "tile", "name": "Tile", "expected": "Crack", "data_uri": img_to_data_uri(img_tile)},
        {"id": "fabric", "name": "Fabric", "expected": "Stain", "data_uri": img_to_data_uri(img_fabric)},
        {"id": "plastic", "name": "Plastic part", "expected": "Speck", "data_uri": img_to_data_uri(img_plastic)},
        {"id": "painted", "name": "Painted panel", "expected": "Scratch", "data_uri": img_to_data_uri(img_painted)},
    ])


# ── Dashboard summary ─────────────────────────────────────────────────────────
@app.route("/api/summary")
def summary():
    return jsonify(db.get_summary())


# ── Inspection history ────────────────────────────────────────────────────────
@app.route("/api/history")
def history():
    search     = request.args.get("search", "").strip()
    category   = request.args.get("category", "").strip()
    date_from  = request.args.get("date_from", "").strip()
    date_to    = request.args.get("date_to", "").strip()
    sort_by    = request.args.get("sort_by", "timestamp_utc")
    sort_order = request.args.get("sort_order", "desc")
    page       = max(1, int(request.args.get("page", 1)))
    per_page   = min(100, max(1, int(request.args.get("per_page", 20))))

    result = db.get_inspections(
        search=search, category=category,
        date_from=date_from, date_to=date_to,
        sort_by=sort_by, sort_order=sort_order,
        page=page, per_page=per_page,
    )
    return jsonify(result)


@app.route("/api/history/<int:inspection_id>")
def history_item(inspection_id: int):
    record = db.get_inspection(inspection_id)
    if record is None:
        return jsonify({"error": "Inspection not found."}), 404
    return jsonify(record)


@app.route("/api/history/<int:inspection_id>", methods=["DELETE"])
def delete_history_item(inspection_id: int):
    deleted = db.delete_inspection(inspection_id)
    if not deleted:
        return jsonify({"error": "Inspection not found."}), 404
    return jsonify({"success": True, "deleted_id": inspection_id})


# ── Analytics ─────────────────────────────────────────────────────────────────
@app.route("/api/analytics/daily")
def analytics_daily():
    date_from = request.args.get("date_from", "")
    date_to   = request.args.get("date_to", "")
    return jsonify(an.daily_chart_data(date_from, date_to))


@app.route("/api/analytics/monthly")
def analytics_monthly():
    year = request.args.get("year", None)
    year = int(year) if year and year.isdigit() else None
    return jsonify(an.monthly_chart_data(year))


@app.route("/api/analytics/yearly")
def analytics_yearly():
    return jsonify(an.yearly_chart_data())


@app.route("/api/analytics/categories")
def analytics_categories():
    date_from = request.args.get("date_from", "")
    date_to   = request.args.get("date_to", "")
    return jsonify(an.category_chart_data(date_from, date_to))


@app.route("/api/analytics/confidence")
def analytics_confidence():
    category  = request.args.get("category", "")
    date_from = request.args.get("date_from", "")
    date_to   = request.args.get("date_to", "")
    return jsonify(an.confidence_chart_data(category, date_from, date_to))


# ── CSV export ────────────────────────────────────────────────────────────────
@app.route("/api/export/csv")
def export_csv():
    category  = request.args.get("category", "")
    date_from = request.args.get("date_from", "")
    date_to   = request.args.get("date_to", "")

    rows = db.get_all_for_export(category, date_from, date_to)

    def generate():
        header = [
            "id", "filename", "timestamp_utc", "predicted_class",
            "confidence", "defect_status", "model_version"
        ]
        yield ",".join(header) + "\r\n"
        for r in rows:
            yield ",".join([
                str(r.get("id", "")),
                _csv_safe(r.get("filename", "")),
                _csv_safe(r.get("timestamp_utc", "")),
                _csv_safe(r.get("predicted_class", "")),
                str(round(r.get("confidence", 0) * 100, 2)) + "%",
                _csv_safe(str(r.get("defect_status") or "N/A")),
                _csv_safe(r.get("model_version", "")),
            ]) + "\r\n"

    filename = f"surfaceai_export_{_now_utc_compact()}.csv"
    return Response(
        generate(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ── Helpers ───────────────────────────────────────────────────────────────────
def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _now_utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _csv_safe(value: str) -> str:
    """Wrap in quotes if the value contains commas, quotes, or newlines."""
    value = str(value).replace('"', '""')
    if any(c in value for c in (",", '"', "\n", "\r")):
        return f'"{value}"'
    return value


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  SurfaceAI — Intelligent Surface Defect Detection")
    print("  http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)
