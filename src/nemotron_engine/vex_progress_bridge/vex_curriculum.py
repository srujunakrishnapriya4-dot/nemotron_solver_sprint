from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import random
from typing import Any


class VexCurriculumError(ValueError):
    pass


def build_curriculum(rows: list[dict[str, Any]], recipe: str, *, seed: int = 1337, max_rows: int | None = None) -> list[dict[str, Any]]:
    if recipe == "micro_sanity":
        allowed = {"direct_raw"}
        cap = max_rows or 64
    elif recipe == "main_public_safe":
        allowed = {"direct_raw", "family_tagged"}
        cap = max_rows or len(rows)
    elif recipe == "main_verified_trace":
        allowed = {"direct_raw", "short_rule_trace"}
        cap = max_rows or len(rows)
    elif recipe == "synthetic_verified":
        allowed = {"synthetic_verified"}
        cap = max_rows or len(rows)
    else:
        raise VexCurriculumError(f"unknown recipe: {recipe}")
    filtered = [row for row in rows if row.get("corpus_type") in allowed]
    if recipe == "main_verified_trace":
        filtered = [row for row in filtered if row.get("corpus_type") != "short_rule_trace" or row.get("metadata", {}).get("verified_status") == "verified_correct"]
    return _balanced(filtered, seed=seed, max_rows=cap)


def load_corpus(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _balanced(rows: list[dict[str, Any]], *, seed: int, max_rows: int) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("family") == "unknown" and row.get("corpus_type") == "short_rule_trace":
            continue
        buckets[str(row.get("family", "unknown"))].append(row)
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    output = []
    while len(output) < max_rows and any(buckets.values()):
        for family in sorted(buckets):
            if buckets[family] and len(output) < max_rows:
                output.append(buckets[family].pop())
    return output
