from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.anti086_barrier.rule_space_coverage_optimizer import optimize_rule_space_coverage  # noqa: E402


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_rule_coverage_optimizer_removes_near_duplicates_and_holds_out_rules(tmp_path: Path) -> None:
    rows = [
        {
            "id": f"r{i}",
            "family": "bit_manipulation",
            "rule_id": f"rule{i % 4}",
            "prompt": "same template 123",
            "answer": "1",
            "generation_hash": f"h{i}",
            "ambiguity_score": 0.1,
            "novelty_score": 0.8,
            "difficulty_score": 0.8,
            "private_like_score": 0.8,
        }
        for i in range(30)
    ]
    _write_jsonl(tmp_path / "filtered.jsonl", rows)
    _write_jsonl(tmp_path / "verified.jsonl", [])
    (tmp_path / "private.json").write_text(json.dumps({"validation_hashes": []}), encoding="utf-8")

    report = optimize_rule_space_coverage(tmp_path / "filtered.jsonl", tmp_path / "verified.jsonl", tmp_path / "private.json", tmp_path, train_limit=20, holdout_limit=4, per_rule_cap=10, per_template_cap=3)

    assert report["selected_count"] <= 20
    assert report["rule_holdout_disjoint"] is True
    assert report["rejected_counts"]["near_duplicate_template_cap"] > 0

