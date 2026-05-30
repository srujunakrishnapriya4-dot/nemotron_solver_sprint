from __future__ import annotations

from typing import Any


def build_composition_shift(rows: list[dict[str, Any]], *, limit: int = 300) -> list[dict[str, Any]]:
    hard = [row for row in rows if int(row.get("parameters", {}).get("depth", 1)) >= 3 or float(row.get("difficulty_score", 0.0)) >= 0.6]
    return hard[:limit]
