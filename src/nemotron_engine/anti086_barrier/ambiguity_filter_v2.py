from __future__ import annotations

from typing import Any


def filter_ambiguity_v2(rows: list[dict[str, Any]], *, max_ambiguity: float = 0.30) -> list[dict[str, Any]]:
    return [row for row in rows if float(row.get("ambiguity_score", 1.0)) <= max_ambiguity and row.get("verifier_status") == "verified"]
