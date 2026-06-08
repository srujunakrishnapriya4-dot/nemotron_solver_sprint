from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.v2_corpus_io import file_record, read_json, read_jsonl, write_json_checked
from kaggle_anti086.training.day2_failure_report_v2 import classify_row


DEFAULT_INPUTS = {
    "core_eval": "artifacts/sprint11/day2_predictions/core_eval_solver_only_predictions.jsonl",
    "family_eval": "artifacts/sprint11/day2_predictions/family_eval_solver_only_predictions.jsonl",
    "rule_holdout": "artifacts/sprint11/day2_predictions/rule_holdout_solver_only_predictions.jsonl",
    "anti_leak": "artifacts/sprint11/day2_predictions/anti_leak_solver_only_predictions.jsonl",
}
OLD_V1_FAMILIES = ("custom_numeral", "equation_operator", "sequence_pattern", "permutation_sorting")


def build_final_solver_score_report(
    *,
    input_files: dict[str, str | Path] | None = None,
    failure_report_v2_path: str | Path = "artifacts/sprint11/day2_failure_report_v2.json",
    eval_ladder_report_path: str | Path = "artifacts/sprint11/day2_eval_ladder_report.json",
    decision_report_path: str | Path = "artifacts/sprint11/day2_decision_report.json",
    hardening_audit_path: str | Path = "artifacts/sprint11/day2_21_failure_hardening_audit.json",
) -> dict[str, Any]:
    paths = input_files or DEFAULT_INPUTS
    rows: list[dict[str, Any]] = []
    inputs: dict[str, Any] = {}
    for dataset, path_value in paths.items():
        path = Path(path_value)
        if not path.exists():
            raise FileNotFoundError(f"missing required solver-only prediction file for {dataset}: {path}")
        dataset_rows = read_jsonl(path)
        for row in dataset_rows:
            item = dict(row)
            item.setdefault("dataset", dataset)
            rows.append(item)
        inputs[dataset] = file_record(path, row_count=len(dataset_rows))
    failure_v2 = _read_optional(failure_report_v2_path)
    ladder = _read_optional(eval_ladder_report_path)
    decision = _read_optional(decision_report_path)
    hardening = _read_optional(hardening_audit_path)
    optional_inputs = {
        "failure_report_v2": _optional_record(failure_report_v2_path),
        "eval_ladder_report": _optional_record(eval_ladder_report_path),
        "decision_report": _optional_record(decision_report_path),
        "hardening_audit": _optional_record(hardening_audit_path),
    }
    by_dataset: dict[str, Any] = defaultdict(_empty_bucket)
    by_family: dict[str, Any] = defaultdict(_empty_bucket)
    overall = _empty_bucket()
    for row in rows:
        classification = classify_row(row)
        _update_bucket(overall, row, classification)
        _update_bucket(by_dataset[str(row.get("dataset", "unknown"))], row, classification)
        _update_bucket(by_family[str(row.get("family") or "unknown")], row, classification)
    finalized_overall = _finalize_bucket(overall)
    finalized_by_dataset = {name: _finalize_bucket(bucket) for name, bucket in sorted(by_dataset.items())}
    finalized_by_family = {name: _finalize_bucket(bucket) for name, bucket in sorted(by_family.items())}
    old_v1_status = _old_v1_repair_status(failure_v2)
    risk = _leaderboard_risk(finalized_overall, ladder=ladder, hardening=hardening)
    status = "WARN" if risk["leaderboard_risk_reasons"] else "PASS"
    return {
        "schema_version": 1,
        "created_by": "DAY2_FINAL_SOLVER_SCORE_REPORT",
        "status": status,
        "inputs": inputs | optional_inputs,
        "overall": finalized_overall | risk,
        "by_dataset": finalized_by_dataset,
        "by_family": finalized_by_family,
        "old_v1_family_repair_status": old_v1_status,
        "submission_policy_analysis": _submission_policy_analysis(finalized_overall, risk),
        "decision": {
            "v2a_150_authorized": False,
            "package_authorized": False,
            "submission_authorized": False,
            "next_action": _next_action(finalized_overall, risk, decision),
        },
        "model_results_faked": False,
        "leaderboard_claim": False,
        "no_0_93_evidence": True,
        "no_0_95_evidence": True,
    }


def _empty_bucket() -> dict[str, Any]:
    return {
        "rows": 0,
        "raw_correct": 0,
        "raw_exact_match": None,
        "answerable_rows": 0,
        "answerable_correct": 0,
        "answerable_wrong": 0,
        "answerable_accuracy": None,
        "abstain_rows": 0,
        "abstain_correct": 0,
        "abstain_unsafe_answer": 0,
        "correct_abstain_rate": None,
        "unsafe_abstain_answer_rate": 0.0,
    }


def _update_bucket(bucket: dict[str, Any], row: dict[str, Any], classification: str) -> None:
    bucket["rows"] += 1
    bucket["raw_correct"] += int(row.get("correct") is True)
    if _is_abstain(_expected(row)):
        bucket["abstain_rows"] += 1
        if classification == "abstain_correct":
            bucket["abstain_correct"] += 1
        elif classification == "abstain_unsafe_answer":
            bucket["abstain_unsafe_answer"] += 1
        return
    bucket["answerable_rows"] += 1
    if classification == "answerable_correct":
        bucket["answerable_correct"] += 1
    else:
        bucket["answerable_wrong"] += 1


def _finalize_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    result = dict(bucket)
    rows = int(result["rows"])
    answerable = int(result["answerable_rows"])
    abstain = int(result["abstain_rows"])
    result["raw_exact_match"] = None if rows == 0 else result.pop("raw_correct") / rows
    result["answerable_accuracy"] = None if answerable == 0 else result["answerable_correct"] / answerable
    result["correct_abstain_rate"] = None if abstain == 0 else result["abstain_correct"] / abstain
    result["unsafe_abstain_answer_rate"] = 0.0 if abstain == 0 else result["abstain_unsafe_answer"] / abstain
    return result


def _old_v1_repair_status(failure_v2: dict[str, Any] | None) -> dict[str, str]:
    output: dict[str, str] = {}
    for family in OLD_V1_FAMILIES:
        if not failure_v2:
            output[family] = "unknown"
            continue
        bucket = failure_v2.get("by_family", {}).get(family)
        if bucket is None:
            output[family] = "not_present"
            continue
        output[family] = "true_answerable_failures_remain" if int(bucket.get("answerable_wrong", 0)) > 0 else "resolved_by_abstain_policy"
    return output


def _leaderboard_risk(overall: dict[str, Any], *, ladder: dict[str, Any] | None, hardening: dict[str, Any] | None) -> dict[str, Any]:
    reasons: list[str] = []
    severity = 0.0
    if overall["abstain_rows"] > 0:
        reasons.append("abstain_policy_unresolved_for_final_leaderboard")
        severity = max(severity, 0.8)
    if overall["unsafe_abstain_answer_rate"] > 0:
        reasons.append("unsafe_abstain_answer_rate_nonzero")
        severity = max(severity, 0.95)
    if overall["answerable_accuracy"] is None or overall["answerable_accuracy"] < 0.95:
        reasons.append("answerable_accuracy_below_0_95")
        severity = max(severity, 0.9)
    ladder_decision = {} if not ladder else ladder.get("decision", {})
    if "combined_not_better_than_solver" in ladder_decision.get("reason_codes", []):
        reasons.append("combined_mode_not_measured_or_not_better_than_solver")
        severity = max(severity, 0.6)
    if "adapter_not_better_than_base" in ladder_decision.get("reason_codes", []):
        reasons.append("adapter_worse_or_equal_to_base")
        severity = max(severity, 0.6)
    if hardening and hardening.get("summary", {}).get("blocked_count", 0):
        reasons.append("hardening_audit_has_blocked_modes")
        severity = max(severity, 0.6)
    if not reasons and overall["answerable_accuracy"] is not None and overall["answerable_accuracy"] >= 0.99 and overall["correct_abstain_rate"] is not None and overall["correct_abstain_rate"] >= 0.99:
        reasons.append("low_local_solver_risk_but_not_leaderboard_evidence")
        severity = max(severity, 0.2)
    return {"leaderboard_risk_score": severity, "leaderboard_risk_reasons": reasons}


def _submission_policy_analysis(overall: dict[str, Any], risk: dict[str, Any]) -> dict[str, Any]:
    abstain_present = overall["abstain_rows"] > 0
    return {
        "if_abstain_is_allowed": {
            "expected_strength": "strong_answerable_solver_behavior" if overall["answerable_accuracy"] and overall["answerable_accuracy"] >= 0.99 else "limited",
            "blocking_issues": [reason for reason in risk["leaderboard_risk_reasons"] if reason != "abstain_policy_unresolved_for_final_leaderboard"],
        },
        "if_abstain_is_not_allowed": {
            "expected_risk": "HIGH" if abstain_present else "MEDIUM",
            "blocking_issues": ["expected_abstain_rows_may_not_score_as_correct"] if abstain_present else [],
        },
        "recommended_next_action": "resolve_abstain_policy_and_combined_measurement_before_submission_decision",
    }


def _next_action(overall: dict[str, Any], risk: dict[str, Any], decision: dict[str, Any] | None) -> str:
    if overall["answerable_wrong"] > 0:
        return "patch_remaining_answerable_solver_failures_then_rerun_solver_only"
    if "combined_mode_not_measured_or_not_better_than_solver" in risk["leaderboard_risk_reasons"]:
        return "measure_or_fix_combined_mode_before_adapter_scaling"
    if overall["abstain_rows"] > 0:
        return "resolve_final_competition_abstain_policy_before_submission"
    return "rerun_full_day2_evidence_ladder"


def _expected(row: dict[str, Any]) -> str | None:
    for key in ("expected", "gold_answer", "answer"):
        if row.get(key) is not None:
            return str(row[key]).strip()
    return None


def _is_abstain(value: str | None) -> bool:
    return value is not None and value.strip().upper() == "ABSTAIN"


def _read_optional(path: str | Path) -> dict[str, Any] | None:
    target = Path(path)
    if not target.exists():
        return None
    try:
        return read_json(target)
    except Exception:
        return None


def _optional_record(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    return file_record(target) if target.exists() else {"path": str(target), "exists": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build final Day2 solver-only score decomposition report.")
    parser.add_argument("--core-predictions", default=DEFAULT_INPUTS["core_eval"])
    parser.add_argument("--family-predictions", default=DEFAULT_INPUTS["family_eval"])
    parser.add_argument("--rule-holdout-predictions", default=DEFAULT_INPUTS["rule_holdout"])
    parser.add_argument("--anti-leak-predictions", default=DEFAULT_INPUTS["anti_leak"])
    parser.add_argument("--failure-report-v2", default="artifacts/sprint11/day2_failure_report_v2.json")
    parser.add_argument("--eval-ladder-report", default="artifacts/sprint11/day2_eval_ladder_report.json")
    parser.add_argument("--decision-report", default="artifacts/sprint11/day2_decision_report.json")
    parser.add_argument("--hardening-audit", default="artifacts/sprint11/day2_21_failure_hardening_audit.json")
    parser.add_argument("--out", default="artifacts/sprint11/day2_final_solver_score_report.json")
    args = parser.parse_args(argv)
    report = build_final_solver_score_report(
        input_files={
            "core_eval": args.core_predictions,
            "family_eval": args.family_predictions,
            "rule_holdout": args.rule_holdout_predictions,
            "anti_leak": args.anti_leak_predictions,
        },
        failure_report_v2_path=args.failure_report_v2,
        eval_ladder_report_path=args.eval_ladder_report,
        decision_report_path=args.decision_report,
        hardening_audit_path=args.hardening_audit,
    )
    write_json_checked(args.out, report, field_name="day2_final_solver_score_report")
    print(json.dumps({"status": report["status"], "out": args.out, "overall": report["overall"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
