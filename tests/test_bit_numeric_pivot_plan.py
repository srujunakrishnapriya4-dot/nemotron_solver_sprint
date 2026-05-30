from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.anti086_barrier.bit_numeric_pivot_plan import build_bit_numeric_pivot_plan  # noqa: E402


def test_bit_numeric_pivot_plan_prefers_bit_when_budget_larger(tmp_path: Path) -> None:
    coverage = tmp_path / "coverage.json"
    coverage.write_text(json.dumps({"top_abstention_reasons": {"bit_candidate_budget_exhausted": 884, "inconsistent_numeric_precision": 557}}), encoding="utf-8")
    plan = build_bit_numeric_pivot_plan(coverage, tmp_path)
    assert plan["recommended_next_sprint"] == "SPRINT-7.7_BIT_BREAKTHROUGH"
    assert (tmp_path / "BIT_NUMERIC_PIVOT_PLAN.md").exists()
