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


DEFAULT_REPAIR_REPORT = "artifacts/sprint11/day3_sequence_pattern_repair_report.json"
CURRENT_CORRECT = 1383
TOTAL_ROWS = 1792
TARGET_CORRECT_0_93 = 1667
PROJECTED_WITH_CUSTOM_NUMERAL = 1495


def build_sequence_pattern_post_audit(
    *,
    repair_report_path: str | Path = DEFAULT_REPAIR_REPORT,
    numeric_formula_failure_status: str = "failed",
    router_changed: bool = False,
    ensemble_changed: bool = False,
    verifier_changed: bool = False,
) -> dict[str, Any]:
    repair_path = Path(repair_report_path)
    if not repair_path.exists():
        raise FileNotFoundError(f"missing required sequence_pattern repair report: {repair_path}")
    repair = read_json(repair_path)
    measurement = _measurement(repair)
    core_changed = bool(router_changed or ensemble_changed or verifier_changed)
    status = _status(measurement, core_changed)
    keep_decision = _keep_decision(measurement, core_changed)
    return {
        "schema_version": 1,
        "created_by": "DAY3_SEQUENCE_PATTERN_POST_AUDIT",
        "status": status,
        "inputs": {
            "sequence_pattern_repair_report": file_record(repair_path),
        },
        "implementation_summary": {
            "files_changed": [
                "kaggle_anti086/solvers/sequence_pattern_solver.py",
                "kaggle_anti086/training/day3_sequence_pattern_repair_report.py",
                "tests/test_day3_sequence_pattern_solver.py",
            ],
            "standalone_solver_added": True,
            "measurement_script_added": True,
            "tests_added": True,
            "router_changed": bool(router_changed),
            "ensemble_changed": bool(ensemble_changed),
            "verifier_changed": bool(verifier_changed),
        },
        "diff_summary": {
            "sequence_pattern_solver": "added standalone verified sequence induction solver with conservative parsing, exact verification, ambiguity guards, and indexed-placeholder rejection",
            "repair_report_script": "added Day2 artifact replay measurement for sequence_pattern rows with unsafe-answer accounting",
            "tests": "added direct standalone solver tests covering rule support, rejection paths, ABSTAIN safety, and no submission/package metadata",
            "router": "changed" if router_changed else "no diff",
            "solver_ensemble": "changed" if ensemble_changed else "no diff",
            "verifier": "changed" if verifier_changed else "no diff",
        },
        "measurement": measurement,
        "artifact_pattern_summary": {
            "all_expected_abstain": True,
            "pattern": "indexed placeholder sequence prompts",
            "reason_routing_is_blocked": "Routing a verified sequence solver would answer expected-ABSTAIN local probes and create unsafe answers.",
        },
        "reclassification": {
            "sequence_pattern_previous_plan_rows": 60,
            "sequence_pattern_actual_recovered_rows": measurement["newly_recovered"],
            "new_label": "BLOCKED_NO_LOCAL_RECOVERY" if measurement["newly_recovered"] == 0 else "RECOVERED_BY_VERIFIED_SOLVER",
            "reason": "Current Day2 sequence_pattern artifacts are expected-ABSTAIN indexed placeholder prompts.",
        },
        "revised_recovery_projection": _revised_projection(),
        "known_unresolved_failures": [
            {
                "test": "tests/test_sprint11_numeric_formula_solver.py::test_numeric_formula_ambiguous_not_low_risk_verified",
                "classification": "pre_existing_unresolved_with_evidence",
                "status": numeric_formula_failure_status,
            }
        ],
        "decision": {
            "keep_sequence_pattern_solver": keep_decision,
            "commit_authorized": False,
            "reason": _decision_reason(keep_decision, measurement, core_changed),
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


def _revised_projection() -> dict[str, Any]:
    return {
        "current_correct": CURRENT_CORRECT,
        "remaining_authorized_families": {"custom_numeral": 112},
        "remaining_plausible_recoverable": 112,
        "projected_correct_if_remaining_plausible_solved": PROJECTED_WITH_CUSTOM_NUMERAL,
        "projected_accuracy_if_remaining_plausible_solved": PROJECTED_WITH_CUSTOM_NUMERAL / TOTAL_ROWS,
        "target_correct_0_93": TARGET_CORRECT_0_93,
        "remaining_gap_to_0_93": TARGET_CORRECT_0_93 - PROJECTED_WITH_CUSTOM_NUMERAL,
        "can_reach_0_93_from_current_revised_plan": False,
    }


def _status(measurement: dict[str, Any], core_changed: bool) -> str:
    if measurement["unsafe_answer_count"] > 0:
        return "FAIL"
    if core_changed:
        return "WARN"
    if measurement["newly_recovered"] == 0:
        return "WARN"
    return "PASS"


def _keep_decision(measurement: dict[str, Any], core_changed: bool) -> str:
    if measurement["unsafe_answer_count"] > 0:
        return "no"
    if core_changed:
        return "quarantine"
    return "yes"


def _decision_reason(keep_decision: str, measurement: dict[str, Any], core_changed: bool) -> str:
    if keep_decision == "no":
        return "unsafe sequence answers block keeping the solver"
    if keep_decision == "quarantine":
        return "core router/ensemble/verifier changes are not justified for zero-recovery sequence_pattern artifacts"
    if measurement["newly_recovered"] == 0:
        return "standalone solver is safe as future utility, but current Day2 artifacts recover zero rows"
    return "standalone solver recovered rows without core integration; commit still requires explicit user approval"


def _next_action(keep_decision: str) -> str:
    if keep_decision == "no":
        return "revert_sequence_pattern_solver_before_day3_continues"
    if keep_decision == "quarantine":
        return "revert_or_quarantine_core_sequence_integration_then_continue_custom_numeral_or_reaudit_extra_gap"
    return "continue_to_custom_numeral_or_reaudit_extra_gap_before_any_submission_decision"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit Day3 sequence_pattern implementation after repair measurement.")
    parser.add_argument("--repair-report", default=DEFAULT_REPAIR_REPORT)
    parser.add_argument("--numeric-formula-failure-status", default="failed", choices=["failed", "passed"])
    parser.add_argument("--router-changed", action="store_true")
    parser.add_argument("--ensemble-changed", action="store_true")
    parser.add_argument("--verifier-changed", action="store_true")
    parser.add_argument("--out", default="artifacts/sprint11/day3_sequence_pattern_post_audit.json")
    args = parser.parse_args(argv)
    report = build_sequence_pattern_post_audit(
        repair_report_path=args.repair_report,
        numeric_formula_failure_status=args.numeric_formula_failure_status,
        router_changed=args.router_changed,
        ensemble_changed=args.ensemble_changed,
        verifier_changed=args.verifier_changed,
    )
    write_json_checked(args.out, report, field_name="day3_sequence_pattern_post_audit")
    print(
        json.dumps(
            {
                "status": report["status"],
                "out": args.out,
                "new_label": report["reclassification"]["new_label"],
                "keep_sequence_pattern_solver": report["decision"]["keep_sequence_pattern_solver"],
                "commit_authorized": report["decision"]["commit_authorized"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
