from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.v2_corpus_io import read_json, write_json_checked


def build_day8_decision(
    collator_path: str | Path = "artifacts/sprint11/day8_real_collator_audit_report.json",
    train_summary_path: str | Path = "artifacts/sprint11/day8_v2a_train_summary.json",
    eval_report_path: str | Path = "artifacts/sprint11/day8_v2a_eval_report.json",
    candidate_report_path: str | Path = "artifacts/sprint11/day8_candidate_report.json",
) -> dict:
    collator = read_json(collator_path) if Path(collator_path).exists() else {"status": "MISSING"}
    train = read_json(train_summary_path) if Path(train_summary_path).exists() else {"status": "MISSING"}
    evaluation = read_json(eval_report_path) if Path(eval_report_path).exists() else {"status": "MISSING"}
    candidate = read_json(candidate_report_path) if Path(candidate_report_path).exists() else {"candidate_quality": "MISSING"}
    if collator.get("status") != "PASS":
        decision = "BLOCK_TRAINING_BAD_LABEL_MASK"
    elif train.get("status") != "PASS" or train.get("trained") is not True:
        decision = "BLOCK_TRAINING_RUNTIME_FAILURE"
    elif evaluation.get("status") == "NEEDS_KAGGLE_MODEL_EVAL":
        decision = "NEEDS_KAGGLE_MODEL_EVAL"
    elif candidate.get("candidate_quality") == "PROMOTE_TO_DAY9_FAILURE_MINING":
        decision = "ALLOW_DAY9_FAILURE_MINING"
    elif candidate.get("candidate_quality") == "REJECT":
        decision = "REJECT_V2A"
    else:
        decision = "BLOCK_DAY9_REPAIR"
    return {
        "status": "PASS" if decision in {"ALLOW_DAY9_FAILURE_MINING", "NEEDS_KAGGLE_MODEL_EVAL"} else "FAIL",
        "decision": decision,
        "packaging_allowed": False,
        "submission_allowed": False,
        "no_0_95_evidence": True,
        "no_leaderboard_evidence": True,
        "no_training_beyond_v2a": True,
        "collator_status": collator.get("status"),
        "training_status": train.get("status"),
        "trained": bool(train.get("trained")),
        "eval_status": evaluation.get("status"),
        "candidate_quality": candidate.get("candidate_quality"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--collator-audit", default="artifacts/sprint11/day8_real_collator_audit_report.json")
    parser.add_argument("--train-summary", default="artifacts/sprint11/day8_v2a_train_summary.json")
    parser.add_argument("--eval-report", default="artifacts/sprint11/day8_v2a_eval_report.json")
    parser.add_argument("--candidate-report", default="artifacts/sprint11/day8_candidate_report.json")
    args = parser.parse_args(argv)
    report = build_day8_decision(args.collator_audit, args.train_summary, args.eval_report, args.candidate_report)
    write_json_checked(args.out, report, field_name="day8_decision_report")
    print(json.dumps({"status": report["status"], "decision": report["decision"], "out": args.out}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
