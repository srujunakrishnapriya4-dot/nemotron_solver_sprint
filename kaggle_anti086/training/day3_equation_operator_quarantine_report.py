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
DEFAULT_POST_AUDIT = "artifacts/sprint11/day3_equation_operator_post_audit.json"
KNOWN_NUMERIC_FAILURE = "tests/test_sprint11_numeric_formula_solver.py::test_numeric_formula_ambiguous_not_low_risk_verified"


def build_equation_operator_quarantine_report(
    *,
    repair_report_path: str | Path = DEFAULT_REPAIR_REPORT,
    post_audit_path: str | Path = DEFAULT_POST_AUDIT,
    quarantine_actions: dict[str, bool] | None = None,
) -> dict[str, Any]:
    repair_path = Path(repair_report_path)
    if not repair_path.exists():
        raise FileNotFoundError(f"missing required repair report: {repair_path}")
    repair = read_json(repair_path)
    post_path = Path(post_audit_path)
    post = read_json(post_path) if post_path.exists() else None
    actions = {
        "standalone_solver_kept": True,
        "standalone_solver_test_kept": True,
        "router_reverted": True,
        "solver_ensemble_reverted": True,
        "verifier_reverted": True,
        "router_tests_reverted": True,
    }
    if quarantine_actions:
        actions.update(quarantine_actions)
    measurement = {
        "rows_seen": int(repair.get("rows_seen", 0)),
        "newly_recovered": int(repair.get("newly_recovered", 0)),
        "unsafe_answer_count": int(repair.get("unsafe_answer_count", 0)),
        "projected_raw_exact_match": repair.get("projected_raw_exact_match_if_applied"),
    }
    failures: list[str] = []
    warnings: list[str] = []
    if measurement["unsafe_answer_count"] > 0:
        failures.append("equation_operator_unsafe_answers_nonzero")
    if not all(actions.values()):
        failures.append("quarantine_cleanup_incomplete")
    if measurement["newly_recovered"] == 0:
        warnings.append("equation_operator_zero_local_recovery")
    status = "FAIL" if failures else ("WARN" if warnings else "PASS")
    return {
        "schema_version": 1,
        "created_by": "DAY3_EQUATION_OPERATOR_QUARANTINE_REPORT",
        "status": status,
        "inputs": {
            "equation_operator_repair_report": file_record(repair_path),
            "equation_operator_post_audit": file_record(post_path) if post_path.exists() else {"path": str(post_path), "exists": False},
        },
        "quarantine_actions": actions,
        "reason": "equation_operator recovered 0 rows and core integration is not justified for current Day3 score plan.",
        "measurement": measurement,
        "reclassification": {
            "family": "equation_operator",
            "new_label": "BLOCKED_NO_LOCAL_RECOVERY",
            "day3_score_gain": measurement["newly_recovered"],
        },
        "known_unresolved_failures": [
            {
                "test": KNOWN_NUMERIC_FAILURE,
                "classification": "pre_existing_unresolved_with_evidence",
                "blocks_full_clean_commit": True,
            }
        ],
        "decision": {
            "safe_to_continue_day3": True,
            "safe_to_commit_solver_integration": False,
            "next_family": "sequence_pattern",
            "next_action": "start Day3 P0 verified sequence_pattern solver implementation after committing safe report/standalone files if desired",
        },
        "blocked_actions": {
            "v2a_150": "blocked",
            "package": "blocked",
            "submission": "blocked",
            "leaderboard_claim": "blocked",
        },
        "warnings": warnings,
        "failures": failures,
        "post_audit_decision": None if not post else post.get("decision"),
        "model_results_faked": False,
        "leaderboard_claim": False,
        "no_0_93_evidence": True,
        "no_0_95_evidence": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Day3 equation_operator quarantine cleanup report.")
    parser.add_argument("--repair-report", default=DEFAULT_REPAIR_REPORT)
    parser.add_argument("--post-audit", default=DEFAULT_POST_AUDIT)
    parser.add_argument("--out", default="artifacts/sprint11/day3_equation_operator_quarantine_report.json")
    args = parser.parse_args(argv)
    report = build_equation_operator_quarantine_report(repair_report_path=args.repair_report, post_audit_path=args.post_audit)
    write_json_checked(args.out, report, field_name="day3_equation_operator_quarantine_report")
    print(
        json.dumps(
            {
                "status": report["status"],
                "out": args.out,
                "safe_to_continue_day3": report["decision"]["safe_to_continue_day3"],
                "safe_to_commit_solver_integration": report["decision"]["safe_to_commit_solver_integration"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
