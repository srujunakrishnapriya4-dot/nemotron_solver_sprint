from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.v2_corpus_io import file_record, read_json, write_json_checked


DEFAULT_INPUTS = {
    "hardening_audit": "artifacts/sprint11/day2_21_failure_hardening_audit.json",
    "final_solver_score_report": "artifacts/sprint11/day2_final_solver_score_report.json",
    "abstain_policy_gate": "artifacts/sprint11/day2_abstain_policy_gate.json",
    "abstain_resolution_report": "artifacts/sprint11/day2_abstain_resolution_report.json",
    "abstain_fallback_mining": "artifacts/sprint11/day2_abstain_fallback_mining.json",
    "unsafe_abstain_forensic_audit": "artifacts/sprint11/day2_unsafe_abstain_forensic_audit.json",
}
TARGET_ACCURACY = 0.93


def build_day2_final_decision_report(
    *,
    input_paths: dict[str, str | Path] | None = None,
    target_accuracy: float = TARGET_ACCURACY,
) -> dict[str, Any]:
    paths = {**DEFAULT_INPUTS, **(input_paths or {})}
    reports = {name: _load_required(path, name=name) for name, path in paths.items()}
    inputs = {name: file_record(path) for name, path in paths.items()}
    final_solver = reports["final_solver_score_report"]
    fallback = reports["abstain_fallback_mining"]
    forensic = reports["unsafe_abstain_forensic_audit"]

    score_state = _score_state(final_solver, target_accuracy=target_accuracy)
    recovery_state = _recovery_state(score_state, fallback, forensic)
    day3_plan = _day3_recovery_plan(recovery_state)
    status = "WARN" if not recovery_state["can_reach_0_93_from_current_authorized_recovery"] else "PASS"
    return {
        "schema_version": 1,
        "created_by": "DAY2_FINAL_DECISION_REPORT",
        "status": status,
        "inputs": inputs,
        "score_state": score_state,
        "recovery_state": recovery_state,
        "day3_recovery_plan": day3_plan,
        "blocked_actions": {
            "v2a_150": "blocked",
            "package": "blocked",
            "submission": "blocked",
            "leaderboard_claim": "blocked",
        },
        "decision": {
            "day2_status": "DAY2_COMPLETE_READY_FOR_DAY3_RECOVERY_PLAN",
            "next_action": "commit_day2_reports_then_start_day3_verified_solver_recovery",
            "authorized_day3_solver_families": ["equation_operator", "sequence_pattern", "custom_numeral"],
            "extra_gap_reaudit_required": True,
        },
        "warnings": _warnings(recovery_state, reports),
        "failures": [],
        "model_results_faked": False,
        "leaderboard_claim": False,
        "no_0_93_evidence": True,
        "no_0_95_evidence": True,
    }


def _load_required(path: str | Path, *, name: str) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"missing required Day2 decision input {name}: {target}")
    return read_json(target)


def _score_state(final_solver: dict[str, Any], *, target_accuracy: float) -> dict[str, Any]:
    overall = final_solver.get("overall", {})
    total = int(overall.get("rows", 0))
    raw = float(overall.get("raw_exact_match", 0.0))
    current = int(round(raw * total))
    target_needed = int(math.ceil(total * target_accuracy))
    return {
        "total_rows": total,
        "current_correct": current,
        "current_raw_exact_match": raw,
        "target_accuracy": target_accuracy,
        "target_correct_needed": target_needed,
        "additional_correct_needed": max(0, target_needed - current),
    }


def _recovery_state(score_state: dict[str, Any], fallback: dict[str, Any], forensic: dict[str, Any]) -> dict[str, Any]:
    fallback_classes = fallback.get("recoverability_classes", {})
    maybe_rows = int(fallback_classes.get("maybe_recoverable_requires_new_solver", {}).get("rows", 0))
    forensic_rows = sum(
        int(forensic.get("forensic_classes", {}).get(name, {}).get("rows", 0))
        for name in (
            "possibly_recoverable_with_new_verified_solver",
            "possibly_recoverable_with_better_parser",
            "possibly_recoverable_with_relaxed_but_verified_rule",
        )
    )
    total_plausible = maybe_rows + forensic_rows
    projected_correct = score_state["current_correct"] + total_plausible
    total_rows = score_state["total_rows"]
    projected_accuracy = None if total_rows == 0 else projected_correct / total_rows
    remaining_gap = max(0, score_state["target_correct_needed"] - projected_correct)
    return {
        "maybe_recoverable_rows": maybe_rows,
        "forensic_recoverable_rows": forensic_rows,
        "total_plausible_recoverable": total_plausible,
        "projected_correct_if_all_plausible_solved": projected_correct,
        "projected_accuracy_if_all_plausible_solved": projected_accuracy,
        "remaining_gap_to_0_93": remaining_gap,
        "can_reach_0_93_from_current_authorized_recovery": bool(projected_accuracy is not None and projected_accuracy >= score_state["target_accuracy"]),
    }


def _day3_recovery_plan(recovery_state: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "priority": "P0",
            "family": "equation_operator",
            "rows": 90,
            "risk": "MEDIUM",
            "required_solver": "verified equation/operator induction",
            "acceptance_gate": "all examples verified, reject multifit",
        },
        {
            "priority": "P0",
            "family": "sequence_pattern",
            "rows": 60,
            "risk": "MEDIUM",
            "required_solver": "verified sequence pattern induction",
            "acceptance_gate": "unique recurrence, reject ambiguity",
        },
        {
            "priority": "P0",
            "family": "custom_numeral",
            "rows": 112,
            "risk": "MEDIUM",
            "required_solver": "verified custom numeral alphabet/base induction",
            "acceptance_gate": "roundtrip encode/decode verified",
        },
        {
            "priority": "P1",
            "family": "low_risk_extra_gap_recovery",
            "rows_needed": int(recovery_state["remaining_gap_to_0_93"]),
            "candidate_families": ["permutation_sorting", "gravity_numeric", "unit_conversion", "bit_manipulation"],
            "risk": "HIGH_UNTIL_REAUDITED",
            "acceptance_gate": "must identify at least 22 low-risk rows before implementation",
        },
    ]


def _warnings(recovery_state: dict[str, Any], reports: dict[str, dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    if not recovery_state["can_reach_0_93_from_current_authorized_recovery"]:
        warnings.append("projected_recovery_still_below_0_93")
    if recovery_state["remaining_gap_to_0_93"] > 0:
        warnings.append("extra_gap_reaudit_required")
    if reports["abstain_resolution_report"].get("policy_decision", {}).get("decision") != "ABSTAIN_ALLOWED":
        warnings.append("abstain_policy_requires_fallback_or_blocks_submission")
    return warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build final Day2 decision report and Day3 recovery plan.")
    parser.add_argument("--hardening-audit", default=DEFAULT_INPUTS["hardening_audit"])
    parser.add_argument("--final-solver-score-report", default=DEFAULT_INPUTS["final_solver_score_report"])
    parser.add_argument("--abstain-policy-gate", default=DEFAULT_INPUTS["abstain_policy_gate"])
    parser.add_argument("--abstain-resolution-report", default=DEFAULT_INPUTS["abstain_resolution_report"])
    parser.add_argument("--abstain-fallback-mining", default=DEFAULT_INPUTS["abstain_fallback_mining"])
    parser.add_argument("--unsafe-abstain-forensic-audit", default=DEFAULT_INPUTS["unsafe_abstain_forensic_audit"])
    parser.add_argument("--out", default="artifacts/sprint11/day2_final_decision_report.json")
    args = parser.parse_args(argv)
    report = build_day2_final_decision_report(
        input_paths={
            "hardening_audit": args.hardening_audit,
            "final_solver_score_report": args.final_solver_score_report,
            "abstain_policy_gate": args.abstain_policy_gate,
            "abstain_resolution_report": args.abstain_resolution_report,
            "abstain_fallback_mining": args.abstain_fallback_mining,
            "unsafe_abstain_forensic_audit": args.unsafe_abstain_forensic_audit,
        }
    )
    write_json_checked(args.out, report, field_name="day2_final_decision_report")
    print(
        json.dumps(
            {
                "status": report["status"],
                "out": args.out,
                "day2_status": report["decision"]["day2_status"],
                "projected_accuracy_if_all_plausible_solved": report["recovery_state"]["projected_accuracy_if_all_plausible_solved"],
                "remaining_gap_to_0_93": report["recovery_state"]["remaining_gap_to_0_93"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
