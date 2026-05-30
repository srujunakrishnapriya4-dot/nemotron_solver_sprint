from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.anti086_barrier.synthetic_quality_filter import filter_synthetic_quality  # noqa: E402


def test_near_duplicate_synthetic_rejected(tmp_path: Path) -> None:
    row = {"id": "a", "prompt": "Same 123", "answer": "1", "family": "bit_manipulation", "rule_id": "choice_mask", "parameters": {"composition_depth": 4}, "verification_trace": {"verified": True}, "target_prediction": "1"}
    src = tmp_path / "synth.jsonl"
    src.write_text(json.dumps(row) + "\n" + json.dumps(dict(row, id="b")) + "\n", encoding="utf-8")

    report = filter_synthetic_quality(src, tmp_path)

    assert report["kept_count"] == 1
    assert report["rejected_counts"]["near_duplicate_surface_template"] == 1

