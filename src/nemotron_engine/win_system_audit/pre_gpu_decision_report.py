from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nemotron_engine.core.schemas import stable_hash


def build_pre_gpu_decision_report(
    input_dir: str | Path = "artifacts/win_system",
    output_dir: str | Path = "artifacts/win_system",
) -> dict[str, Any]:
    root = Path(input_dir)
    coverage = _read_json(root / "solver_coverage_report.json")
    quarantine = _read_json(root / "equation_quarantine_manifest.json")
    corpus = _read_json(root / "win_corpus_manifest.json")
    roi = _read_json(root / "hard_family_roi_report.json")
    sprint75 = coverage.get("sprint75_acceptance", {})
    sprint77 = coverage.get("sprint77_acceptance", {})
    sprint78 = coverage.get("sprint78_acceptance", {})
    unsafe_excluded = quarantine.get("equation_unsafe_excluded_count", 0) > 0
    corpus_clean = corpus.get("equation_status") in {"QUARANTINED", "OFFLINE_SYNTHESIS_REQUIRED"}
    if sprint78.get("success") and corpus_clean and corpus.get("bit_truth_table_trace_count", 0) > 0:
        decision = "PHASE1_COMPLETE_GPU_MICRO_ALLOWED"
        note = "Micro is allowed after SPRINT-7.8; main remains blocked until parent-child eval beats hard-family gates."
    elif corpus_clean and coverage.get("bit_program_synthesis", {}).get("bit_verified_correct", 0) > 772 and coverage.get("bit_program_synthesis", {}).get("bit_wrong", 999) <= 14:
        decision = "PHASE1_PARTIAL_GPU_MICRO_STACK_TEST_ONLY"
        note = "Micro may be run only as a stack test, not as a 0.95 candidate; hard-family symbolic Phase 1 did not pass."
    else:
        decision = "PHASE1_INCOMPLETE_GPU_BLOCKED"
        note = "GPU main is blocked; micro training is not a 0.95 candidate while equation/bit/numeric gaps remain."
    report = {
        "decision": decision,
        "gpu_main_allowed": False,
        "gpu_micro_is_stack_test_only": decision == "PHASE1_PARTIAL_GPU_MICRO_STACK_TEST_ONLY",
        "unsafe_equation_traces_excluded": unsafe_excluded,
        "corpus_clean": corpus_clean,
        "equation_status": quarantine.get("equation_status"),
        "sprint77_success": bool(sprint77.get("success")),
        "sprint77_verdict": sprint77.get("verdict", "SPRINT-7.7_NOT_RUN"),
        "sprint78_success": bool(sprint78.get("success")),
        "sprint78_verdict": sprint78.get("verdict", "SPRINT-7.8_NOT_RUN"),
        "recommended_next_sprint": roi.get("recommended_next_sprint", "SPRINT-7.7_BIT_BREAKTHROUGH"),
        "note": note,
    }
    report["report_hash"] = stable_hash(report)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pre_gpu_decision_report.json").write_text(json.dumps(report, sort_keys=True, indent=2), encoding="utf-8")
    (out / "PRE_GPU_DECISION_REPORT.md").write_text(_md(report), encoding="utf-8")
    return report


def _md(report: dict[str, Any]) -> str:
    return (
        "# Pre-GPU Decision Report\n\n"
        f"Decision: {report['decision']}\n\n"
        f"Equation status: {report.get('equation_status')}\n\n"
        f"Recommended next sprint: {report['recommended_next_sprint']}\n\n"
        f"{report['note']}\n"
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
