"""
database.py — SurfaceAI SQLite database layer.
Handles schema creation, inspection CRUD, and analytics queries.
"""

import sqlite3
import json
import os
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), "surfaceai.db")


def get_connection():
    """Return a new SQLite connection with row_factory set."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create schema if it does not already exist."""
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS inspections (
                id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                filename                TEXT    NOT NULL,
                timestamp_utc           TEXT    NOT NULL,
                predicted_class         TEXT    NOT NULL,
                confidence              REAL    NOT NULL,
                class_probabilities_json TEXT   NOT NULL,
                defect_status           TEXT    DEFAULT NULL,
                model_version           TEXT    NOT NULL DEFAULT 'unknown'
            );

            CREATE INDEX IF NOT EXISTS idx_inspections_timestamp
                ON inspections (timestamp_utc);

            CREATE INDEX IF NOT EXISTS idx_inspections_predicted_class
                ON inspections (predicted_class);
        """)


# ---------------------------------------------------------------------------
# Write operations
# ---------------------------------------------------------------------------

def save_inspection(
    filename: str,
    predicted_class: str,
    confidence: float,
    class_probabilities: dict,
    defect_status: str | None,
    model_version: str,
) -> int:
    """Insert a new inspection record and return its id."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO inspections
                (filename, timestamp_utc, predicted_class, confidence,
                 class_probabilities_json, defect_status, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                filename,
                ts,
                predicted_class,
                round(float(confidence), 6),
                json.dumps(class_probabilities),
                defect_status,
                model_version,
            ),
        )
        return cur.lastrowid


def delete_inspection(inspection_id: int) -> bool:
    """Delete an inspection by id. Returns True if a row was deleted."""
    with get_connection() as conn:
        cur = conn.execute(
            "DELETE FROM inspections WHERE id = ?", (inspection_id,)
        )
        return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------

def get_inspection(inspection_id: int) -> dict | None:
    """Return a single inspection as a dict, or None."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM inspections WHERE id = ?", (inspection_id,)
        ).fetchone()
    if row is None:
        return None
    return _row_to_dict(row)


def get_inspections(
    search: str = "",
    category: str = "",
    date_from: str = "",
    date_to: str = "",
    sort_by: str = "timestamp_utc",
    sort_order: str = "desc",
    page: int = 1,
    per_page: int = 20,
) -> dict:
    """Return paginated, filtered inspections plus total count."""
    allowed_sort = {
        "timestamp_utc", "confidence", "predicted_class", "id", "filename"
    }
    allowed_order = {"asc", "desc"}
    if sort_by not in allowed_sort:
        sort_by = "timestamp_utc"
    if sort_order not in allowed_order:
        sort_order = "desc"

    conditions = []
    params: list = []

    if search:
        conditions.append("(filename LIKE ? OR CAST(id AS TEXT) = ?)")
        params += [f"%{search}%", search.strip()]

    if category:
        conditions.append("predicted_class = ?")
        params.append(category)

    if date_from:
        conditions.append("timestamp_utc >= ?")
        params.append(date_from)

    if date_to:
        # include the full day
        params.append(date_to if "T" in date_to else date_to + "T23:59:59Z")
        conditions.append("timestamp_utc <= ?")

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    with get_connection() as conn:
        total = conn.execute(
            f"SELECT COUNT(*) FROM inspections {where}", params
        ).fetchone()[0]

        offset = (page - 1) * per_page
        rows = conn.execute(
            f"""
            SELECT * FROM inspections
            {where}
            ORDER BY {sort_by} {sort_order}
            LIMIT ? OFFSET ?
            """,
            params + [per_page, offset],
        ).fetchall()

    return {
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": max(1, (total + per_page - 1) // per_page),
        "items": [_row_to_dict(r) for r in rows],
    }


def get_summary() -> dict:
    """Return dashboard KPI metrics."""
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM inspections").fetchone()[0]

        row = conn.execute(
            """
            SELECT predicted_class, COUNT(*) AS cnt
            FROM inspections
            GROUP BY predicted_class
            ORDER BY cnt DESC
            LIMIT 1
            """
        ).fetchone()
        most_common = row["predicted_class"] if row else None

        latest = conn.execute(
            "SELECT * FROM inspections ORDER BY timestamp_utc DESC LIMIT 1"
        ).fetchone()

        recent = conn.execute(
            """
            SELECT * FROM inspections
            ORDER BY timestamp_utc DESC
            LIMIT 10
            """
        ).fetchall()

    return {
        "total": total,
        "most_common_defect": most_common,
        "latest": _row_to_dict(latest) if latest else None,
        "recent": [_row_to_dict(r) for r in recent],
    }


# ---------------------------------------------------------------------------
# Analytics queries
# ---------------------------------------------------------------------------

def get_daily_analytics(date_from: str = "", date_to: str = "") -> list:
    where, params = _date_where(date_from, date_to)
    with get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT substr(timestamp_utc, 1, 10) AS day,
                   COUNT(*) AS total,
                   predicted_class
            FROM inspections
            {where}
            GROUP BY day, predicted_class
            ORDER BY day
            """,
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def get_monthly_analytics(year: int | None = None) -> list:
    params: list = []
    cond = ""
    if year:
        cond = "WHERE substr(timestamp_utc,1,4) = ?"
        params.append(str(year))
    with get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT substr(timestamp_utc, 1, 7) AS month,
                   COUNT(*) AS total,
                   predicted_class
            FROM inspections
            {cond}
            GROUP BY month, predicted_class
            ORDER BY month
            """,
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def get_yearly_analytics() -> list:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT substr(timestamp_utc, 1, 4) AS year,
                   COUNT(*) AS total,
                   predicted_class
            FROM inspections
            GROUP BY year, predicted_class
            ORDER BY year
            """
        ).fetchall()
    return [dict(r) for r in rows]


def get_category_analytics(date_from: str = "", date_to: str = "") -> list:
    where, params = _date_where(date_from, date_to)
    with get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT predicted_class, COUNT(*) AS count
            FROM inspections
            {where}
            GROUP BY predicted_class
            ORDER BY count DESC
            """,
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def get_confidence_analytics(
    category: str = "", date_from: str = "", date_to: str = ""
) -> list:
    conditions = []
    params: list = []
    if category:
        conditions.append("predicted_class = ?")
        params.append(category)
    if date_from:
        conditions.append("timestamp_utc >= ?")
        params.append(date_from)
    if date_to:
        conditions.append("timestamp_utc <= ?")
        params.append(date_to if "T" in date_to else date_to + "T23:59:59Z")
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    with get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT
                CASE
                    WHEN confidence < 0.5  THEN '0–50%'
                    WHEN confidence < 0.6  THEN '50–60%'
                    WHEN confidence < 0.7  THEN '60–70%'
                    WHEN confidence < 0.8  THEN '70–80%'
                    WHEN confidence < 0.9  THEN '80–90%'
                    ELSE '90–100%'
                END AS bucket,
                COUNT(*) AS count
            FROM inspections
            {where}
            GROUP BY bucket
            ORDER BY bucket
            """,
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_for_export(
    category: str = "", date_from: str = "", date_to: str = ""
) -> list:
    conditions = []
    params: list = []
    if category:
        conditions.append("predicted_class = ?")
        params.append(category)
    if date_from:
        conditions.append("timestamp_utc >= ?")
        params.append(date_from)
    if date_to:
        conditions.append("timestamp_utc <= ?")
        params.append(date_to if "T" in date_to else date_to + "T23:59:59Z")
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    with get_connection() as conn:
        rows = conn.execute(
            f"SELECT * FROM inspections {where} ORDER BY timestamp_utc DESC",
            params,
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    try:
        d["class_probabilities"] = json.loads(d.get("class_probabilities_json", "{}"))
    except (json.JSONDecodeError, TypeError):
        d["class_probabilities"] = {}
    return d


def _date_where(date_from: str, date_to: str):
    conditions = []
    params: list = []
    if date_from:
        conditions.append("timestamp_utc >= ?")
        params.append(date_from)
    if date_to:
        conditions.append("timestamp_utc <= ?")
        params.append(date_to if "T" in date_to else date_to + "T23:59:59Z")
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    return where, params


# Aliases
get_db_summary = get_summary
