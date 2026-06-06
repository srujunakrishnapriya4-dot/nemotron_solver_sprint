from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.v2_corpus_io import read_json, write_json_checked


def decide_candidate(
    *,
    eval_report: dict[str, Any] | None,
    error_mining_report: dict[str, Any] | None,
    package_guard_report: dict[str, Any] | None,
    public_lb_calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    eval_completed = bool(eval_report and eval_report.get("model_eval_completed"))
    package_pass = bool(package_guard_report and package_guard_report.get("status") == "PASS")
    public_recorded = bool(public_lb_calibration and public_lb_calibration.get("public_lb_calibration_recorded") is True)
    if not eval_completed:
        failures.append("adapter_eval_missing_or_incomplete")
    if not package_pass:
        failures.append("package_rehearsal_not_pass")
    if not public_recorded:
        warnings.append("public_lb_calibration_missing_submit_recommended_false")
    day10_exact = float((eval_report or {}).get("exact_match", 0.0))
    v2a_exact = float((eval_report or {}).get("v2a_exact_match", (eval_report or {}).get("baselines", {}).get("v2a_exact_match", 0.0)))
    tinker_exact = float((eval_report or {}).get("tinker_exact_match", (eval_report or {}).get("baselines", {}).get("tinker_exact_match", 0.0)))
    severe_regressions = list((eval_report or {}).get("severe_family_regressions", []))
    if eval_completed and day10_exact < v2a_exact:
        failures.append("day10_adapter_does_not_beat_v2a")
    if eval_completed and tinker_exact and day10_exact < tinker_exact:
        failures.append("day10_adapter_below_tinker_baseline")
    if severe_regressions:
        failures.append("severe_family_regression")
    submit_recommended = bool(eval_completed and package_pass and public_recorded and not failures)
    return {
        "status": "PASS" if not failures else "FAIL",
        "candidate": "day10_solver_teacher_adapter",
        "adapter_eval_completed": eval_completed,
        "adapter_exact_match": day10_exact,
        "v2a_exact_match": v2a_exact,
        "tinker_exact_match": tinker_exact,
        "error_mining_status": None if error_mining_report is None else error_mining_report.get("status"),
        "candidate_quality": "HOLD_NO_SUBMIT" if not submit_recommended else "READY_FOR_PUBLIC_LB_CALIBRATED_SUBMISSION_REVIEW",
        "submit_recommended": submit_recommended,
        "packaging_allowed": False,
        "submission_allowed": False,
        "public_lb_calibration_recorded": public_recorded,
        "no_leaderboard_evidence": not public_recorded,
        "no_0_95_evidence": True,
        "warnings": warnings,
        "failures": failures,
    }


def _read_optional(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    return read_json(p) if p.exists() else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Decide whether Day 10 adapter candidate can move toward submission.")
    parser.add_argument("--eval-report", default="artifacts/sprint11/day10_adapter_eval_report.json")
    parser.add_argument("--error-mining-report", default="artifacts/sprint11/day10_adapter_error_mining_report.json")
    parser.add_argument("--package-guard-report", default="artifacts/sprint11/day10_adapter_package_guard_report.json")
    parser.add_argument("--public-lb-calibration", default=None)
    parser.add_argument("--out", default="artifacts/sprint11/day10_candidate_decision_report.json")
    args = parser.parse_args(argv)
    report = decide_candidate(
        eval_report=_read_optional(args.eval_report),
        error_mining_report=_read_optional(args.error_mining_report),
        package_guard_report=_read_optional(args.package_guard_report),
        public_lb_calibration=_read_optional(args.public_lb_calibration),
    )
    write_json_checked(args.out, report, field_name="day10_candidate_decision_report")
    print(json.dumps({"status": report["status"], "submit_recommended": report["submit_recommended"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
