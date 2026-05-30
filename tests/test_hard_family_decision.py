from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.win_system_audit.hard_family_decision import make_hard_family_decision  # noqa: E402


def test_hard_family_decision_combines_roi_quarantine_and_pivot(tmp_path: Path) -> None:
    (tmp_path / "solver_coverage_report.json").write_text(json.dumps({"solver_verified_correct_by_family": {"equation_symbolic": 57}, "solver_wrong_by_family": {}, "top_abstention_reasons": {"unsupported_equation_transform": 1467, "bit_candidate_budget_exhausted": 884}}), encoding="utf-8")
    (tmp_path / "equation_failure_taxonomy.json").write_text(json.dumps({"cluster_counts": {"digit_symbol_arithmetic": 1}}), encoding="utf-8")
    (tmp_path / "digit_symbol_failure_taxonomy.json").write_text(json.dumps({"constraint_summary": {"enough_constraints": 1}}), encoding="utf-8")
    for name in ["solver_verified_correct.jsonl", "solver_verified_wrong.jsonl", "solver_abstained.jsonl", "digit_symbol_failure_examples.jsonl"]:
        (tmp_path / name).write_text("", encoding="utf-8")
    decision = make_hard_family_decision(tmp_path)
    assert decision["pivot_recommendation"] == "SPRINT-7.7_BIT_BREAKTHROUGH"
    assert not decision["gpu_main_allowed"]
