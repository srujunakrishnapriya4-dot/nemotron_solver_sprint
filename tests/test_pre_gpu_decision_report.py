from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.win_system_audit.pre_gpu_decision_report import build_pre_gpu_decision_report  # noqa: E402


def test_pre_gpu_report_blocks_main_training(tmp_path: Path) -> None:
    (tmp_path / "solver_coverage_report.json").write_text(json.dumps({"sprint75_acceptance": {"success": False}}), encoding="utf-8")
    (tmp_path / "equation_quarantine_manifest.json").write_text(json.dumps({"equation_status": "OFFLINE_SYNTHESIS_REQUIRED", "equation_unsafe_excluded_count": 10}), encoding="utf-8")
    (tmp_path / "win_corpus_manifest.json").write_text(json.dumps({"equation_status": "OFFLINE_SYNTHESIS_REQUIRED"}), encoding="utf-8")
    (tmp_path / "hard_family_roi_report.json").write_text(json.dumps({"recommended_next_sprint": "SPRINT-7.7_BIT_BREAKTHROUGH"}), encoding="utf-8")
    report = build_pre_gpu_decision_report(tmp_path, tmp_path)
    assert report["decision"] == "PHASE1_INCOMPLETE_GPU_BLOCKED"
    assert not report["gpu_main_allowed"]
    assert "not a 0.95 candidate" in report["note"]


def test_pre_gpu_report_allows_partial_micro_stack_test(tmp_path: Path) -> None:
    (tmp_path / "solver_coverage_report.json").write_text(json.dumps({"sprint78_acceptance": {"success": False, "verdict": "SPRINT-7.8_FAIL_BIT_TRUTH_TABLE_MINING"}, "bit_program_synthesis": {"bit_verified_correct": 774, "bit_wrong": 9}}), encoding="utf-8")
    (tmp_path / "equation_quarantine_manifest.json").write_text(json.dumps({"equation_status": "OFFLINE_SYNTHESIS_REQUIRED", "equation_unsafe_excluded_count": 10}), encoding="utf-8")
    (tmp_path / "win_corpus_manifest.json").write_text(json.dumps({"equation_status": "OFFLINE_SYNTHESIS_REQUIRED", "bit_truth_table_trace_count": 2}), encoding="utf-8")
    (tmp_path / "hard_family_roi_report.json").write_text(json.dumps({}), encoding="utf-8")
    report = build_pre_gpu_decision_report(tmp_path, tmp_path)
    assert report["decision"] == "PHASE1_PARTIAL_GPU_MICRO_STACK_TEST_ONLY"
