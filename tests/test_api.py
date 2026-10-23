"""
tests/test_api.py — API integration tests for SurfaceAI.
Run: python -m pytest tests/ -v
"""

import sys, os, io, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from PIL import Image
import app as flask_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Create a test client with an isolated SQLite database."""
    import database as db
    test_db = tmp_path / "test.db"
    monkeypatch.setattr(db, "DB_PATH", str(test_db))
    db.init_db()
    flask_app.app.config["TESTING"] = True
    with flask_app.app.test_client() as c:
        yield c


def _jpeg_bytes(width=64, height=64):
    img = Image.new("RGB", (width, height), color=(180, 180, 180))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


# ── Health ────────────────────────────────────────────────────────────────────
def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.get_json()
    assert data["status"] == "ok"
    assert "timestamp" in data


# ── Model status ──────────────────────────────────────────────────────────────
def test_model_status(client):
    r = client.get("/api/model-status")
    assert r.status_code == 200
    data = r.get_json()
    assert "loaded" in data
    assert "version" in data


# ── Summary (empty DB) ────────────────────────────────────────────────────────
def test_summary_empty(client):
    r = client.get("/api/summary")
    assert r.status_code == 200
    data = r.get_json()
    assert data["total"] == 0
    assert data["latest"] is None
    assert isinstance(data["recent"], list)


# ── History (empty DB) ────────────────────────────────────────────────────────
def test_history_empty(client):
    r = client.get("/api/history")
    assert r.status_code == 200
    data = r.get_json()
    assert data["total"] == 0
    assert data["items"] == []


# ── Inspect — missing image ───────────────────────────────────────────────────
def test_inspect_no_image(client):
    r = client.post("/api/inspect")
    assert r.status_code == 400
    assert "error" in r.get_json()


# ── Inspect — wrong file type ─────────────────────────────────────────────────
def test_inspect_wrong_type(client):
    data = {"image": (io.BytesIO(b"plain text"), "file.txt")}
    r = client.post("/api/inspect", data=data, content_type="multipart/form-data")
    assert r.status_code == 400


# ── Inspect — invalid image bytes ────────────────────────────────────────────
def test_inspect_invalid_image(client):
    data = {"image": (io.BytesIO(b"\xff\xd8\xff garbage"), "corrupt.jpg")}
    r = client.post("/api/inspect", data=data, content_type="multipart/form-data")
    # Either 400 (validation) or 503 (model not loaded) is acceptable
    assert r.status_code in (400, 503)


# ── Analytics endpoints ───────────────────────────────────────────────────────
@pytest.mark.parametrize("endpoint", [
    "/api/analytics/daily",
    "/api/analytics/monthly",
    "/api/analytics/yearly",
    "/api/analytics/categories",
    "/api/analytics/confidence",
])
def test_analytics_empty(client, endpoint):
    r = client.get(endpoint)
    assert r.status_code == 200
    data = r.get_json()
    assert "labels" in data or "datasets" in data or "data" in data


# ── Export CSV ────────────────────────────────────────────────────────────────
def test_export_csv_empty(client):
    r = client.get("/api/export/csv")
    assert r.status_code == 200
    assert "text/csv" in r.content_type


# ── Delete non-existent record ────────────────────────────────────────────────
def test_delete_nonexistent(client):
    r = client.delete("/api/history/99999")
    assert r.status_code == 404


# ── History item not found ────────────────────────────────────────────────────
def test_history_item_not_found(client):
    r = client.get("/api/history/99999")
    assert r.status_code == 404
