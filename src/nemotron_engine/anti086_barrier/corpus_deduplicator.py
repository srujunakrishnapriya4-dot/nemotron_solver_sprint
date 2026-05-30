from __future__ import annotations

from typing import Any


def deduplicate_corpus(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for row in rows:
        key = (row.get("family"), row.get("text"), row.get("answer"))
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out
