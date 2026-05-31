from __future__ import annotations

from collections import Counter
from typing import Any


TARGET_MIXTURE = {
    "bit_manipulation": 0.15,
    "numeric_formula": 0.12,
    "gravity_numeric": 0.08,
    "unit_conversion": 0.12,
    "word_cipher": 0.15,
    "char_cipher": 0.10,
    "symbol_mapping": 0.12,
    "roman_numeral": 0.06,
    "format_only": 0.05,
    "other_supported": 0.05,
}


def build_mixture_report(direct_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(direct_rows)
    family_counts = Counter(str(row.get("family", "unknown")) for row in direct_rows)
    source_counts = Counter(str(row.get("source_row_id", "")) for row in direct_rows)
    actual = {family: (count / total if total else 0.0) for family, count in sorted(family_counts.items())}
    underrepresented = []
    for family, target in TARGET_MIXTURE.items():
        if family == "other_supported":
            continue
        if actual.get(family, 0.0) < target * 0.35:
            underrepresented.append(family)
    max_duplicate = max(source_counts.values(), default=0)
    return {
        "status": "PASS" if max_duplicate <= 1 else "FAIL",
        "target_mixture": TARGET_MIXTURE,
        "actual_mixture": actual,
        "family_counts": dict(sorted(family_counts.items())),
        "underrepresented_families": underrepresented,
        "oversampling_used": False,
        "max_duplicate_source_row_count": max_duplicate,
    }
