"""
tests/test_analytics.py — Unit tests for analytics aggregation helpers.
Run: python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import database as db
import analytics as an
from datetime import datetime, timezone


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Each test gets a fresh in-memory-like isolated DB."""
    test_db = tmp_path / "test_analytics.db"
    monkeypatch.setattr(db, "DB_PATH", str(test_db))
    db.init_db()


def _insert(cls="Crazing", conf=0.85, ts=None):
    ts = ts or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with db.get_connection() as conn:
        conn.execute(
            """INSERT INTO inspections
               (filename, timestamp_utc, predicted_class, confidence,
                class_probabilities_json, model_version)
               VALUES (?,?,?,?,?,?)""",
            ("test.jpg", ts, cls, conf, "{}", "test-v1"),
        )


# ── daily_chart_data ──────────────────────────────────────────────────────────
def test_daily_chart_data_empty():
    result = an.daily_chart_data()
    assert isinstance(result["labels"], list)
    assert len(result["labels"]) == 0


def test_daily_chart_data_with_records():
    _insert("Crazing", 0.9, "2024-06-15T10:00:00Z")
    _insert("Scratches", 0.7, "2024-06-15T11:00:00Z")
    _insert("Inclusion", 0.8, "2024-06-16T09:00:00Z")

    result = an.daily_chart_data()
    assert "2024-06-15" in result["labels"]
    assert "2024-06-16" in result["labels"]
    total_ds = next((d for d in result["datasets"] if d["label"] == "Total"), None)
    assert total_ds is not None
    idx = result["labels"].index("2024-06-15")
    assert total_ds["data"][idx] == 2


# ── monthly_chart_data ────────────────────────────────────────────────────────
def test_monthly_chart_data_empty():
    result = an.monthly_chart_data()
    assert result["labels"] == []


def test_monthly_chart_data_with_records():
    _insert("Patches", 0.6, "2024-03-10T08:00:00Z")
    _insert("Patches", 0.7, "2024-04-20T08:00:00Z")

    result = an.monthly_chart_data(year=2024)
    assert "2024-03" in result["labels"]
    assert "2024-04" in result["labels"]


# ── yearly_chart_data ─────────────────────────────────────────────────────────
def test_yearly_chart_data_empty():
    result = an.yearly_chart_data()
    assert result["labels"] == []


def test_yearly_chart_data_multiple_years():
    _insert("Crazing", 0.8, "2023-05-01T00:00:00Z")
    _insert("Inclusion", 0.9, "2024-07-01T00:00:00Z")

    result = an.yearly_chart_data()
    assert "2023" in result["labels"]
    assert "2024" in result["labels"]


# ── category_chart_data ───────────────────────────────────────────────────────
def test_category_chart_data_empty():
    result = an.category_chart_data()
    assert result["labels"] == []
    assert result["total"] == 0


def test_category_chart_data_counts():
    for _ in range(3): _insert("Crazing", 0.9)
    for _ in range(2): _insert("Scratches", 0.7)

    result = an.category_chart_data()
    idx_c = result["labels"].index("Crazing")
    idx_s = result["labels"].index("Scratches")
    assert result["data"][idx_c] == 3
    assert result["data"][idx_s] == 2
    assert result["total"] == 5
    assert result["most_common"] == "Crazing"


# ── confidence_chart_data ─────────────────────────────────────────────────────
def test_confidence_chart_data_empty():
    result = an.confidence_chart_data()
    assert sum(result["data"]) == 0


def test_confidence_buckets():
    _insert("Crazing",  0.55)
    _insert("Crazing",  0.75)
    _insert("Scratches", 0.95)

    result = an.confidence_chart_data()
    assert len(result["labels"]) == 6
    # Total count should match inserts
    assert sum(result["data"]) == 3
