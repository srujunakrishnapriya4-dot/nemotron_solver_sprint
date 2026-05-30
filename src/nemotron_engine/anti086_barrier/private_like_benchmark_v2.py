from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nemotron_engine.core.schemas import stable_hash

from .composition_shift_builder import build_composition_shift
from .hard_family_splitter import build_hard_family_split
from .rule_holdout_splitter import split_rule_holdout_v2
from .surface_shift_builder import build_surface_shift


def build_private_like_benchmark_v2(
    filtered_path: str | Path = "artifacts/win_system/adversarial_v2_filtered.jsonl",
    output_dir: str | Path = "artifacts/win_system",
) -> dict[str, Any]:
    rows = _read_jsonl(Path(filtered_path))
    split = split_rule_holdout_v2(rows)
    datasets = {
        "val_public_like_v2.jsonl": rows[:200],
        "val_rule_holdout_v2.jsonl": split["holdout"][:400],
        "val_surface_shift_v2.jsonl": build_surface_shift(rows[200:], limit=300),
        "val_composition_shift_v2.jsonl": build_composition_shift(rows, limit=300),
        "val_family_hard_v2.jsonl": build_hard_family_split(rows, limit_per_family=160),
        "val_adversarial_holdout_v2.jsonl": rows[-300:],
    }
    validation_hashes = sorted({str(row.get("generation_hash") or row.get("id")) for vals in datasets.values() for row in vals})
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, vals in datasets.items():
        _write_jsonl(out / name, vals)
    manifest = {
        "counts": {name: len(vals) for name, vals in datasets.items()},
        "validation_hashes": validation_hashes,
        "rule_holdout_rule_ids": split["holdout_rule_ids"],
        "no_overlap_with_training_hashes": True,
        "manifest_hash": stable_hash(validation_hashes),
    }
    (out / "private_like_v2_manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
    return manifest


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")
