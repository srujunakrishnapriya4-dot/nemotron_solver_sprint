from __future__ import annotations

from typing import Any


def build_surface_shift(rows: list[dict[str, Any]], *, limit: int = 300) -> list[dict[str, Any]]:
    out = []
    for row in rows[:limit]:
        clone = dict(row)
        clone["id"] = f"{row.get('generation_hash', row.get('id'))}_surface_v2"
        clone["prompt"] = "Wonderland variant: " + str(row.get("prompt", ""))
        clone["surface_shift"] = True
        out.append(clone)
    return out
