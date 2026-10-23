"""
analytics.py — Higher-level analytics aggregation helpers for SurfaceAI.
These wrap the raw database queries and shape data for the frontend charts.
"""

from database import (
    get_daily_analytics,
    get_monthly_analytics,
    get_yearly_analytics,
    get_category_analytics,
    get_confidence_analytics,
)
from collections import defaultdict


# ── Class display names (kept in sync with model.py) ─────────────────────────
DISPLAY_CLASSES = [
    "Crazing",
    "Inclusion",
    "Patches",
    "Pitted Surface",
    "Rolled-in Scale",
    "Scratches",
]

CLASS_COLORS = {
    "Crazing":         "#22d3ee",
    "Inclusion":       "#818cf8",
    "Patches":         "#f472b6",
    "Pitted Surface":  "#fb923c",
    "Rolled-in Scale": "#a3e635",
    "Scratches":       "#fbbf24",
}


def daily_chart_data(date_from: str = "", date_to: str = "") -> dict:
    """
    Returns data shaped for a Chart.js multi-dataset line/bar chart.
    Each dataset is one predicted class; x-axis = unique dates.
    """
    rows = get_daily_analytics(date_from, date_to)

    days_set: set[str] = set()
    class_day: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for r in rows:
        day = r["day"]
        cls = r["predicted_class"]
        days_set.add(day)
        class_day[cls][day] += r["total"]

    labels = sorted(days_set)
    datasets = []
    totals_by_day: dict[str, int] = defaultdict(int)

    for cls in DISPLAY_CLASSES:
        data = [class_day[cls].get(d, 0) for d in labels]
        for d, v in zip(labels, data):
            totals_by_day[d] += v
        datasets.append({
            "label": cls,
            "data": data,
            "backgroundColor": CLASS_COLORS.get(cls, "#888"),
            "borderColor": CLASS_COLORS.get(cls, "#888"),
        })

    # Total per day dataset
    datasets.insert(0, {
        "label": "Total",
        "data": [totals_by_day[d] for d in labels],
        "backgroundColor": "#4f8ef7",
        "borderColor": "#4f8ef7",
        "type": "line",
        "fill": False,
        "tension": 0.3,
        "yAxisID": "y",
    })

    return {"labels": labels, "datasets": datasets}


def monthly_chart_data(year: int | None = None) -> dict:
    """
    Returns monthly totals, optionally filtered to a given year.
    """
    rows = get_monthly_analytics(year)

    months_set: set[str] = set()
    class_month: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for r in rows:
        month = r["month"]  # e.g. "2024-03"
        cls = r["predicted_class"]
        months_set.add(month)
        class_month[cls][month] += r["total"]

    labels = sorted(months_set)
    datasets = []
    totals: dict[str, int] = defaultdict(int)

    for cls in DISPLAY_CLASSES:
        data = [class_month[cls].get(m, 0) for m in labels]
        for m, v in zip(labels, data):
            totals[m] += v
        datasets.append({
            "label": cls,
            "data": data,
            "backgroundColor": CLASS_COLORS.get(cls, "#888"),
            "borderColor": CLASS_COLORS.get(cls, "#888"),
        })

    datasets.insert(0, {
        "label": "Total",
        "data": [totals[m] for m in labels],
        "backgroundColor": "#4f8ef7",
        "borderColor": "#4f8ef7",
        "type": "line",
        "fill": False,
        "tension": 0.3,
    })

    return {"labels": labels, "datasets": datasets}


def yearly_chart_data() -> dict:
    rows = get_yearly_analytics()

    years_set: set[str] = set()
    class_year: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for r in rows:
        year = r["year"]
        cls = r["predicted_class"]
        years_set.add(year)
        class_year[cls][year] += r["total"]

    labels = sorted(years_set)
    datasets = []
    totals: dict[str, int] = defaultdict(int)

    for cls in DISPLAY_CLASSES:
        data = [class_year[cls].get(y, 0) for y in labels]
        for y, v in zip(labels, data):
            totals[y] += v
        datasets.append({
            "label": cls,
            "data": data,
            "backgroundColor": CLASS_COLORS.get(cls, "#888"),
            "borderColor": CLASS_COLORS.get(cls, "#888"),
        })

    datasets.insert(0, {
        "label": "Total",
        "data": [totals[y] for y in labels],
        "backgroundColor": "#4f8ef7",
        "borderColor": "#4f8ef7",
        "type": "line",
        "fill": False,
        "tension": 0.3,
    })

    return {"labels": labels, "datasets": datasets}


def category_chart_data(date_from: str = "", date_to: str = "") -> dict:
    rows = get_category_analytics(date_from, date_to)
    total = sum(r["count"] for r in rows)

    labels = []
    data = []
    colors = []
    percentages = []

    for r in rows:
        cls = r["predicted_class"]
        cnt = r["count"]
        labels.append(cls)
        data.append(cnt)
        colors.append(CLASS_COLORS.get(cls, "#888"))
        percentages.append(round(cnt / total * 100, 1) if total else 0)

    return {
        "labels": labels,
        "data": data,
        "backgroundColor": colors,
        "percentages": percentages,
        "total": total,
        "most_common": labels[0] if labels else None,
    }


def confidence_chart_data(
    category: str = "", date_from: str = "", date_to: str = ""
) -> dict:
    rows = get_confidence_analytics(category, date_from, date_to)
    buckets = ["0–50%", "50–60%", "60–70%", "70–80%", "80–90%", "90–100%"]
    count_map = {r["bucket"]: r["count"] for r in rows}
    data = [count_map.get(b, 0) for b in buckets]
    return {
        "labels": buckets,
        "data": data,
        "backgroundColor": [
            "#ef4444", "#f97316", "#eab308",
            "#22c55e", "#06b6d4", "#6366f1"
        ],
    }
