from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.anti086_barrier.private_like_benchmark_v2 import build_private_like_benchmark_v2  # noqa: E402


def test_private_like_benchmark_v2_writes_splits(tmp_path: Path) -> None:
    rows = [{"id": str(i), "family": "bit_manipulation", "rule_id": f"r{i%4}", "prompt": "p", "answer": "1", "generation_hash": f"h{i}", "difficulty_score": 0.8, "private_like_score": 0.8, "parameters": {"depth": 3}} for i in range(20)]
    src = tmp_path / "filtered.jsonl"
    src.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    manifest = build_private_like_benchmark_v2(src, tmp_path)
    assert manifest["counts"]["val_rule_holdout_v2.jsonl"] > 0
    assert (tmp_path / "private_like_v2_manifest.json").exists()
