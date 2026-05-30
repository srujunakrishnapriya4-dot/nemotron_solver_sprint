from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.anti086_barrier.private_like_benchmark import build_private_like_benchmark  # noqa: E402


def test_private_like_validation_excludes_training_rows_and_rule_holdout(tmp_path: Path) -> None:
    rows = []
    for i in range(20):
        rows.append({"id": f"r{i}", "family": "bit_manipulation", "rule_id": f"rule{i%5}", "prompt": "p", "answer": "1", "generation_hash": f"h{i}", "difficulty_score": 0.8, "private_like_score": 0.9})
    src = tmp_path / "filtered.jsonl"
    src.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    manifest = build_private_like_benchmark(src, tmp_path)

    assert manifest["validation_hashes_excluded_from_training"] is True
    assert manifest["counts"]["private_like_val_rule_holdout.jsonl"] > 0
    rule_rows = [json.loads(line) for line in (tmp_path / "private_like_val_rule_holdout.jsonl").read_text(encoding="utf-8").splitlines()]
    held_rules = {row["rule_id"] for row in rule_rows}
    assert held_rules == set(manifest["rule_holdout_rule_ids"])

