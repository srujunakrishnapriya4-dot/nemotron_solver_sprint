from __future__ import annotations

from collections import Counter
from typing import Any


HARD = {"equation_symbolic", "bit_manipulation", "gravity_numeric", "unit_conversion", "cipher_text"}


def build_hard_family_split(rows: list[dict[str, Any]], *, limit_per_family: int = 160) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    out = []
    for row in sorted(rows, key=lambda item: (-float(item.get("private_like_score", 0.0)), str(item.get("family")))):
        family = str(row.get("family"))
        cap = limit_per_family if family in HARD else min(30, limit_per_family)
        if counts[family] < cap:
            out.append(row)
            counts[family] += 1
    return out
