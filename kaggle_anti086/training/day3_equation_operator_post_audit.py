from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.v2_corpus_io import file_record, read_json, write_json_checked


DEFAULT_REPAIR_REPORT = "artifacts/sprint11/day3_equation_operator_repair_report.json"
DEFAULT_NUMERIC_EVIDENCE = (
    "targeted numeric_formula test fails while directly instantiating NumericFormulaSolver",
    "git diff shows no changes to kaggle_anti086/solvers/numeric_formula_solver.py",
    "git diff shows no changes to tests/test_sprint11_numeric_formula_solver.py",
    "failing code path does not call modified router, solver_ensemble, verifier, or equation_operator_solver",
)
FILE_DECISION_KEYS = (
    "kaggle_anti086/solvers/equation_operator_solver.py",
    "kaggle_anti086/solvers/router.py",
    "kaggle_anti086/solvers/solver_ensemble.py",
    "kaggle_anti086/solvers/verifier.py",
    "tests/test_day3_equation_operator_solver.py",
    "tests/test_sprint11_router.py",
    "tests/test_sprint11_router_verifier_boundaries.py",
)


def build_equation_operator_post_audit(
    *,
    repair_report_path: str | Path = DEFAULT_REPAIR_REPORT,
    numeric_formula_failure_status: str = "pre_existing_unresolved_with_evidence",
    numeric_formula_evidence: list[str] | None = None,
) -> dict[str, Any]:
    repair_path = Path(repair_report_path)
    if not repair_path.exists():
        raise FileNotFoundError(f"missing required equation_operator repair report: {repair_path}")
    measurement_report = read_json(repair_path)
    measurement = _measurement(measurement_report)
    regression = _regression_audit(numeric_formula_failure_status, numeric_formula_evidence)
    file_decisions = _file_decisions(measurement, regression)
    reclassification = _reclassification(measurement)
    revised = _revised_projection(measurement)
    status = _status(measurement, regression)
    keep_decision = _keep_decision(measurement, regression, file_decisions)
    return {
        "schema_version": 1,
        "created_by": "DAY3_EQUATION_OPERATOR_POST_AUDIT",
        "status": status,
        "inputs": {
            "equation_operator_repair_report": file_record(repair_path),
        },
        "implementation_summary": {
            "files_changed": list(FILE_DECISION_KEYS),
            "solver_added": True,
            "router_changed": True,
            "ensemble_changed": True,
            "verifier_changed": True,
            "tests_changed": True,
        },
        "diff_summary": {
            "equation_operator_solver": "added standalone verified equation/operator induction solver with parse, candidate generation, all-example verification, and ambiguity guards",
            "router": "changed equation_operator from unsupported high-risk route to supported equation_operator_solver route",
            "solver_ensemble": "imports/adds EquationOperatorSolver, treats equation_operator as numeric, and restricts high-confidence equation routes to routed solver only",
            "verifier": "treats equation_operator answers as numeric and applies numeric answer validation",
            "tests": "added Day3 equation_operator tests and updated router boundary expectations from unsupported to supported-safe-abstain behavior",
        },
        "measurement": measurement,
        "regression_audit": regression,
        "file_decisions": file_decisions,
        "reclassification": reclassification,
        "revised_recovery_projection": revised,
        "decision": {
            "keep_equation_operator_solver": keep_decision,
            "commit_authorized": False,
            "reason": _decision_reason(keep_decision, regression, measurement),
            "next_action": _next_action(keep_decision),
        },
        "blocked_actions": {
            "v2a_150": "blocked",
            "package": "blocked",
            "submission": "blocked",
            "leaderboard_claim": "blocked",
        },
        "model_results_faked": False,
        "leaderboard_claim": False,
        "no_0_93_evidence": True,
        "no_0_95_evidence": True,
    }


def _measurement(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "rows_seen": int(report.get("rows_seen", 0)),
        "newly_recovered": int(report.get("newly_recovered", 0)),
        "unsafe_answer_count": int(report.get("unsafe_answer_count", 0)),
        "projected_raw_exact_match": report.get("projected_raw_exact_match_if_applied"),
    }


def _regression_audit(status: str, evidence: list[str] | None) -> dict[str, Any]:
    allowed = {
        "passed",
        "pre_existing_unresolved_with_evidence",
        "regression_from_equation_operator",
        "unknown_blocker",
    }
    if status not in allowed:
        raise ValueError(f"invalid numeric_formula_failure_status: {status}")
    return {
        "numeric_formula_failure_status": status,
        "evidence": list(evidence or DEFAULT_NUMERIC_EVIDENCE),
    }


def _file_decisions(measurement: dict[str, Any], regression: dict[str, Any]) -> dict[str, str]:
    if measurement["unsafe_answer_count"] > 0 or regression["numeric_formula_failure_status"] == "regression_from_equation_operator":
        return {key: "REVERT" for key in FILE_DECISION_KEYS}
    if measurement["newly_recovered"] == 0:
        return {
            "kaggle_anti086/solvers/equation_operator_solver.py": "KEEP",
            "kaggle_anti086/solvers/router.py": "QUARANTINE",
            "kaggle_anti086/solvers/solver_ensemble.py": "QUARANTINE",
            "kaggle_anti086/solvers/verifier.py": "QUARANTINE",
            "tests/test_day3_equation_operator_solver.py": "KEEP",
            "tests/test_sprint11_router.py": "QUARANTINE",
            "tests/test_sprint11_router_verifier_boundaries.py": "QUARANTINE",
        }
    return {key: "KEEP" for key in FILE_DECISION_KEYS}


def _reclassification(measurement: dict[str, Any]) -> dict[str, Any]:
    return {
        "equation_operator_previous_plan_rows": 90,
        "equation_operator_actual_recovered_rows": measurement["newly_recovered"],
        "new_label": "BLOCKED_NO_LOCAL_RECOVERY" if measurement["newly_recovered"] == 0 else "RECOVERED_BY_VERIFIED_SOLVER",
        "reason": "Current Day2 equation_operator artifacts are symbolic/ABSTAIN placeholders without numeric recoverable examples.",
    }


def _revised_projection(measurement: dict[str, Any]) -> dict[str, Any]:
    current = 1383
    total = 1792
    remaining = {"sequence_pattern": 60, "custom_numeral": 112}
    recoverable = sum(remaining.values())
    projected = current + recoverable
    target = 1667
    return {
        "current_correct": current,
        "remaining_authorized_families": remaining,
        "remaining_plausible_recoverable": recoverable,
        "projected_correct_if_remaining_plausible_solved": projected,
        "projected_accuracy_if_remaining_plausible_solved": projected / total,
        "target_correct_0_93": target,
        "remaining_gap_to_0_93": max(0, target - projected),
        "can_reach_0_93_from_current_revised_plan": False,
    }


def _status(measurement: dict[str, Any], regression: dict[str, Any]) -> str:
    if measurement["unsafe_answer_count"] > 0:
        return "FAIL"
    if regression["numeric_formula_failure_status"] == "regression_from_equation_operator":
        return "FAIL"
    if regression["numeric_formula_failure_status"] == "unknown_blocker":
        return "WARN"
    if measurement["newly_recovered"] == 0:
        return "WARN"
    return "PASS"


def _keep_decision(measurement: dict[str, Any], regression: dict[str, Any], file_decisions: dict[str, str]) -> str:
    if measurement["unsafe_answer_count"] > 0 or regression["numeric_formula_failure_status"] == "regression_from_equation_operator":
        return "no"
    if "QUARANTINE" in set(file_decisions.values()) or regression["numeric_formula_failure_status"] != "passed":
        return "quarantine"
    return "yes"


def _decision_reason(keep_decision: str, regression: dict[str, Any], measurement: dict[str, Any]) -> str:
    if keep_decision == "no":
        return "unsafe answer or equation_operator-caused regression blocks keeping patch"
    if keep_decision == "quarantine":
        return "standalone solver is tested, but zero local recovery and core routing changes do not justify commit authorization"
    return "solver recovered rows without regression; commit still requires explicit user approval"


def _next_action(keep_decision: str) -> str:
    if keep_decision == "quarantine":
        return "quarantine_integration_or_revert_router_ensemble_verifier_then_start_sequence_pattern_or_custom_numeral"
    if keep_decision == "no":
        return "revert_equation_operator_patch_before_day3_continues"
    return "seek_explicit_commit_approval_or_continue_day3_recovery"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit Day3 equation_operator implementation after zero-recovery measurement.")
    parser.add_argument("--repair-report", default=DEFAULT_REPAIR_REPORT)
    parser.add_argument(
        "--numeric-formula-failure-status",
        default="pre_existing_unresolved_with_evidence",
        choices=["passed", "pre_existing_unresolved_with_evidence", "regression_from_equation_operator", "unknown_blocker"],
    )
    parser.add_argument("--numeric-evidence", action="append", default=[])
    parser.add_argument("--out", default="artifacts/sprint11/day3_equation_operator_post_audit.json")
    args = parser.parse_args(argv)
    report = build_equation_operator_post_audit(
        repair_report_path=args.repair_report,
        numeric_formula_failure_status=args.numeric_formula_failure_status,
        numeric_formula_evidence=args.numeric_evidence or None,
    )
    write_json_checked(args.out, report, field_name="day3_equation_operator_post_audit")
    print(
        json.dumps(
            {
                "status": report["status"],
                "out": args.out,
                "keep_equation_operator_solver": report["decision"]["keep_equation_operator_solver"],
                "commit_authorized": report["decision"]["commit_authorized"],
                "new_label": report["reclassification"]["new_label"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
