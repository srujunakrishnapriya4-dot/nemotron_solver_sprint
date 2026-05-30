from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from nemotron_engine.core.schemas import stable_hash

from .anti_memorization_split import exclude_validation_hashes, split_rule_holdout


def build_private_like_benchmark(
    filtered_synthetic_path: str | Path = "artifacts/anti086/adversarial_synthetic_filtered.jsonl",
    output_dir: str | Path = "artifacts/anti086",
) -> dict[str, Any]:
    rows = _read_jsonl(Path(filtered_synthetic_path))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    split = split_rule_holdout(rows, holdout_fraction=0.25) if rows else {"train": [], "rule_holdout": []}
    val_public = rows[: min(200, len(rows))]
    val_rule = split["rule_holdout"][: min(300, len(split["rule_holdout"]))]
    val_surface = [dict(row, id=f"{row['id']}_surface") for row in rows[200:400]]
    val_comp = [row for row in rows[400:700] if float(row.get("difficulty_score", 0.0)) >= 0.55]
    val_family_hard = _family_hard(rows[700:1200], limit_per_family=120)
    val_adv = rows[1200:1500]
    datasets = {
        "private_like_val_public_like.jsonl": val_public,
        "private_like_val_rule_holdout.jsonl": val_rule,
        "private_like_val_surface_shift.jsonl": val_surface,
        "private_like_val_composition_shift.jsonl": val_comp,
        "private_like_val_family_hard.jsonl": val_family_hard,
        "private_like_val_adversarial_holdout.jsonl": val_adv,
    }
    train_exclusion = []
    for vals in datasets.values():
        train_exclusion.extend(vals)
    validation_hashes = sorted({row.get("generation_hash") or row.get("row_hash") or row.get("id") for row in train_exclusion})
    train_allowed = exclude_validation_hashes(rows, train_exclusion)
    for filename, payload in datasets.items():
        _write_jsonl(out / filename, payload)
    manifest = {
        "counts": {name: len(payload) for name, payload in datasets.items()},
        "validation_hashes_excluded_from_training": True,
        "validation_hashes": validation_hashes,
        "train_allowed_after_exclusion": len(train_allowed),
        "rule_holdout_rule_ids": sorted({row.get("rule_id") for row in val_rule}),
        "manifest_hash": stable_hash({name: [row.get("generation_hash") for row in payload] for name, payload in datasets.items()}),
    }
    (out / "private_like_manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
    return manifest


def _family_hard(rows: list[dict[str, Any]], *, limit_per_family: int) -> list[dict[str, Any]]:
    counts = Counter()
    output = []
    for row in sorted(rows, key=lambda item: (-float(item.get("private_like_score", 0.0)), str(item.get("id")))):
        family = row.get("family", "unknown")
        if counts[family] < limit_per_family:
            output.append(row)
            counts[family] += 1
    return output


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")
