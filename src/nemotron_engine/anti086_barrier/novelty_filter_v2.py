from __future__ import annotations

from typing import Any


def filter_novelty_v2(rows: list[dict[str, Any]], *, min_distance: float = 0.35) -> list[dict[str, Any]]:
    kept = []
    seen_templates: set[str] = set()
    for row in rows:
        template = _template(row.get("prompt", ""))
        if template in seen_templates:
            continue
        if float(row.get("nearest_train_template_distance", 1.0)) < min_distance:
            continue
        seen_templates.add(template)
        kept.append(row)
    return kept


def _template(prompt: str) -> str:
    return " ".join(token if not any(ch.isdigit() for ch in token) else "N" for token in prompt.lower().split()[:30])
