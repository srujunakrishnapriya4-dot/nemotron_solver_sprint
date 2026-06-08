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


DEFAULT_FINAL_SOLVER_REPORT = "artifacts/sprint11/day2_final_solver_score_report.json"
DEFAULT_FAILURE_REPORT_V2 = "artifacts/sprint11/day2_failure_report_v2.json"


def build_abstain_policy_gate(
    *,
    final_solver_report_path: str | Path = DEFAULT_FINAL_SOLVER_REPORT,
    failure_report_v2_path: str | Path = DEFAULT_FAILURE_REPORT_V2,
) -> dict[str, Any]:
    final_path = Path(final_solver_report_path)
    if not final_path.exists():
        raise FileNotFoundError(f"missing required final solver score report: {final_path}")
    final_report = read_json(final_path)
    overall = final_report.get("overall", {})
    raw_exact_match = _float(overall.get("raw_exact_match"))
    answerable_accuracy = _optional_float(overall.get("answerable_accuracy"))
    abstain_rows = int(overall.get("abstain_rows", 0))
    correct_abstain_rate = _optional_float(overall.get("correct_abstain_rate"))
    unsafe_abstain_answer_rate = _float(overall.get("unsafe_abstain_answer_rate", 0.0))
    metrics = {
        "raw_exact_match": raw_exact_match,
        "answerable_accuracy": answerable_accuracy,
        "abstain_rows": abstain_rows,
        "correct_abstain_rate": correct_abstain_rate,
        "unsafe_abstain_answer_rate": unsafe_abstain_answer_rate,
    }
    final_policy_known = _final_submission_policy_known()
    strong_answerable = answerable_accuracy is not None and answerable_accuracy >= 0.99 and correct_abstain_rate is not None and correct_abstain_rate >= 0.99
    policy_modes = _policy_modes(metrics, final_policy_known=final_policy_known, strong_answerable=strong_answerable)
    failures: list[str] = []
    warnings: list[str] = []
    if unsafe_abstain_answer_rate > 0:
        failures.append("unsafe_abstain_answer_rate_nonzero")
    if abstain_rows > 0 and not final_policy_known:
        warnings.append("final_submission_abstain_policy_unknown")
    if raw_exact_match < 0.93:
        warnings.append("raw_exact_match_below_0_93")
    status = "FAIL" if failures else ("WARN" if warnings else "PASS")
    return {
        "schema_version": 1,
        "created_by": "DAY2_ABSTAIN_POLICY_GATE",
        "status": status,
        "inputs": {
            "final_solver_score_report": file_record(final_path),
            "failure_report_v2": _optional_record(failure_report_v2_path),
        },
        "metrics": metrics,
        "policy_modes": policy_modes,
        "decision": {
            "final_submission_policy_known": final_policy_known,
            "safe_to_package": False,
            "safe_to_submit": False,
            "next_action": _next_action(metrics, final_policy_known=final_policy_known, failures=failures),
        },
        "warnings": warnings,
        "failures": failures,
        "leaderboard_claim": False,
        "no_0_93_evidence": raw_exact_match < 0.93,
        "no_0_95_evidence": True,
    }


def _policy_modes(metrics: dict[str, Any], *, final_policy_known: bool, strong_answerable: bool) -> dict[str, Any]:
    allowed_blocks = []
    if metrics["unsafe_abstain_answer_rate"] > 0:
        allowed_blocks.append("unsafe_abstain_answer_rate_nonzero")
    if metrics["raw_exact_match"] < 0.93:
        allowed_blocks.append("raw_exact_match_below_0_93_even_with_current_local_policy")
    allowed_interpretation = (
        "answerable solver is strong and abstain behavior is clean under local policy; leaderboard score still depends on final scoring policy"
        if strong_answerable
        else "answerable or abstain behavior is not strong enough for score claims"
    )
    not_allowed_blocks = []
    if metrics["abstain_rows"] > 0:
        not_allowed_blocks.append("expected_abstain_rows_would_not_receive_credit_if_blank_abstain_disallowed")
    if metrics["raw_exact_match"] < 0.93:
        not_allowed_blocks.append("raw_exact_match_below_0_93")
    return {
        "abstain_allowed": {
            "candidate_score_interpretation": allowed_interpretation,
            "risk": "HIGH" if metrics["unsafe_abstain_answer_rate"] > 0 else ("MEDIUM" if not final_policy_known or metrics["raw_exact_match"] < 0.93 else "LOW"),
            "blocking_issues": allowed_blocks,
        },
        "abstain_not_allowed": {
            "candidate_score_interpretation": "raw exact-match is the relevant conservative view if ABSTAIN/blank does not score as correct",
            "risk": "HIGH" if metrics["abstain_rows"] > 0 or metrics["unsafe_abstain_answer_rate"] > 0 else ("MEDIUM" if metrics["raw_exact_match"] < 0.93 else "LOW"),
            "blocking_issues": not_allowed_blocks,
        },
    }


def _next_action(metrics: dict[str, Any], *, final_policy_known: bool, failures: list[str]) -> str:
    if failures:
        return "fix_abstain_safety_before_any_submission_decision"
    if metrics["abstain_rows"] > 0 and not final_policy_known:
        return "resolve_final_abstain_scoring_policy_before_package_or_submit"
    if metrics["raw_exact_match"] < 0.93:
        return "do_not_claim_0_93_fix_raw_exact_match_or_policy_gap"
    return "continue_day2_evidence_review_without_packaging"


def _final_submission_policy_known() -> bool:
    # There is no trusted repository artifact proving final competition scoring
    # accepts ABSTAIN/blank behavior. Keep this false until explicit policy
    # evidence is added and audited.
    return False


def _float(value: Any) -> float:
    return float(0.0 if value is None else value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_record(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    return file_record(target) if target.exists() else {"path": str(target), "exists": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Day2 ABSTAIN policy decision gate.")
    parser.add_argument("--final-solver-report", default=DEFAULT_FINAL_SOLVER_REPORT)
    parser.add_argument("--failure-report-v2", default=DEFAULT_FAILURE_REPORT_V2)
    parser.add_argument("--out", default="artifacts/sprint11/day2_abstain_policy_gate.json")
    args = parser.parse_args(argv)
    report = build_abstain_policy_gate(final_solver_report_path=args.final_solver_report, failure_report_v2_path=args.failure_report_v2)
    write_json_checked(args.out, report, field_name="day2_abstain_policy_gate")
    print(json.dumps({"status": report["status"], "out": args.out, "decision": report["decision"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
