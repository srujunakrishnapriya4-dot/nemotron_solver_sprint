from __future__ import annotations

from collections import Counter
from typing import Any


TARGET_CAPS = {
    "equation_symbolic": 700,
    "bit_manipulation": 700,
    "gravity_numeric": 300,
    "unit_conversion": 300,
    "cipher_text": 300,
}


def balance_rule_space(rows: list[dict[str, Any]], caps: dict[str, int] | None = None) -> list[dict[str, Any]]:
    caps = caps or TARGET_CAPS
    counts: Counter[str] = Counter()
    out = []
    for row in sorted(rows, key=lambda item: (-float(item.get("private_like_score", 0.0)), str(item.get("family")), str(item.get("rule_id")), str(item.get("generation_hash")))):
        family = str(row.get("family", "unknown"))
        if counts[family] < caps.get(family, 0):
            out.append(row)
            counts[family] += 1
    return out
