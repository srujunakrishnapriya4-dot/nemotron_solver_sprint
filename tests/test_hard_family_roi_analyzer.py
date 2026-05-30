from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.anti086_barrier.hard_family_roi_analyzer import analyze_hard_family_roi  # noqa: E402


def test_roi_analyzer_recommends_equation_pause_and_bit_pivot(tmp_path: Path) -> None:
    (tmp_path / "solver_coverage_report.json").write_text(json.dumps({"solver_verified_correct_by_family": {"equation_symbolic": 57, "bit_manipulation": 707}, "solver_wrong_by_family": {"equation_symbolic": 10}, "top_abstention_reasons": {"unsupported_equation_transform": 1467, "bit_candidate_budget_exhausted": 884}}), encoding="utf-8")
    (tmp_path / "equation_failure_taxonomy.json").write_text(json.dumps({"cluster_counts": {"digit_symbol_arithmetic": 282}}), encoding="utf-8")
    (tmp_path / "digit_symbol_failure_taxonomy.json").write_text(json.dumps({"constraint_summary": {"enough_constraints": 188, "underdetermined_or_unseen_target_operator": 315}}), encoding="utf-8")
    report = analyze_hard_family_roi(tmp_path, tmp_path / "roi.json")
    assert report["families"]["equation_symbolic"]["recommended_action"] == "OFFLINE_SYNTHESIS"
    assert report["families"]["bit_manipulation"]["recommended_action"] == "PIVOT_TO_THIS_FAMILY"
