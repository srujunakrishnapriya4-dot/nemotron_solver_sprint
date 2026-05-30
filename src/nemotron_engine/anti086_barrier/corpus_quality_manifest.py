from __future__ import annotations

from collections import Counter
from typing import Any

from nemotron_engine.core.schemas import stable_hash


def build_corpus_quality_manifest(corpora: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        "file_counts": {name: len(rows) for name, rows in corpora.items()},
        "family_counts": {name: dict(Counter(row.get("family", "unknown") for row in rows)) for name, rows in corpora.items()},
        "synthetic_ratio_v2": _ratio(corpora.get("win_v2.jsonl", []), "synthetic_verified"),
        "contrastive_ratio_v3": _ratio(corpora.get("win_v3.jsonl", []), "contrastive_error"),
        "manifest_hash": stable_hash({name: [row.get("row_hash") or row.get("generation_hash") or row.get("id") for row in rows] for name, rows in corpora.items()}),
    }


def _ratio(rows: list[dict[str, Any]], corpus_type: str) -> float:
    return sum(1 for row in rows if row.get("corpus_type") == corpus_type) / max(1, len(rows))
