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
from model import load_model, get_model_status, predict, grad_cam

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
    # Validate upload
    if "image" not in request.files:
        return jsonify({"error": "No image field in request."}), 400

    f = request.files["image"]
    if not f or not f.filename:
        return jsonify({"error": "No file selected."}), 400

    filename = secure_filename(f.filename)
    if not allowed_file(filename):
        return jsonify({"error": "Unsupported file type. Use JPG or PNG."}), 400

    image_bytes = f.read()
    if len(image_bytes) == 0:
        return jsonify({"error": "Uploaded file is empty."}), 400

    # Verify the bytes are actually a decodable image
    try:
        from PIL import Image as PILImage
        PILImage.open(io.BytesIO(image_bytes)).verify()
    except Exception as exc:
        return jsonify({"error": f"File cannot be decoded as an image: {exc}"}), 400

    # Run inference
    try:
        result = predict(image_bytes)
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
        defect_status=None,        # binary classification not yet available
        model_version=result["model_version"],
    )

    # Optional Grad-CAM
    gradcam_b64 = None
    try:
        gc_bytes = grad_cam(image_bytes)
        if gc_bytes:
            gradcam_b64 = base64.b64encode(gc_bytes).decode("ascii")
    except Exception:
        pass

    return jsonify({
        "inspection_id": inspection_id,
        "filename": filename,
        "predicted_class": result["predicted_class"],
        "confidence": result["confidence"],
        "class_probabilities": result["class_probabilities"],
        "top3": result["top3"],
        "model_version": result["model_version"],
        "gradcam": gradcam_b64,
        "timestamp": _now_utc(),
    })


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
