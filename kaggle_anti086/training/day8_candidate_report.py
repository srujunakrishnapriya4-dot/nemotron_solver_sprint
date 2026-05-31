from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.v2_corpus_io import read_json, write_json_checked


def build_candidate_report(
    train_summary_path: str | Path = "artifacts/sprint11/day8_v2a_train_summary.json",
    eval_report_path: str | Path = "artifacts/sprint11/day8_v2a_eval_report.json",
) -> dict:
    train = read_json(train_summary_path) if Path(train_summary_path).exists() else {"status": "MISSING", "trained": False}
    evaluation = read_json(eval_report_path) if Path(eval_report_path).exists() else {"status": "MISSING", "model_eval_completed": False}
    quality = "NEEDS_KAGGLE_EVAL"
    status = "WARN"
    adapter_files = bool(train.get("adapter_config_exists") and train.get("adapter_model_exists"))
    if train.get("status") == "FAIL" or (train.get("trained") and not adapter_files):
        quality = "REJECT"
        status = "FAIL"
    elif evaluation.get("status") == "PASS":
        regressions = evaluation.get("regressions", [])
        deltas = [float(item.get("delta", 0.0)) for item in evaluation.get("evals", {}).values()]
        if regressions or any(delta < 0 for delta in deltas):
            quality = "REJECT"
            status = "FAIL"
        elif deltas and all(delta >= -0.000001 for delta in deltas[:2]):
            quality = "PROMOTE_TO_DAY9_FAILURE_MINING"
            status = "PASS"
    return {
        "status": status,
        "candidate": "v2a_base_lora",
        "adapter_dir": train.get("adapter_dir") or evaluation.get("adapter_dir"),
        "trained": bool(train.get("trained")),
        "eval_completed": bool(evaluation.get("model_eval_completed")),
        "private_like_delta": _delta(evaluation, "private_like_answerable_512"),
        "rule_holdout_delta": _delta(evaluation, "rule_holdout_answerable_512"),
        "family_hard_delta": _delta(evaluation, "family_hard_answerable_512"),
        "anti_leak_delta": _delta(evaluation, "anti_leak"),
        "family_regressions": evaluation.get("regressions", []),
        "candidate_quality": quality,
        "packaging_allowed": False,
        "submission_allowed": False,
        "no_leaderboard_evidence": True,
        "no_0_95_evidence": True,
    }


def _delta(report: dict, name: str) -> float | None:
    value = report.get("evals", {}).get(name, {}).get("delta")
    return None if value is None else float(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-summary", default="artifacts/sprint11/day8_v2a_train_summary.json")
    parser.add_argument("--eval-report", default="artifacts/sprint11/day8_v2a_eval_report.json")
    args = parser.parse_args(argv)
    report = build_candidate_report(args.train_summary, args.eval_report)
    write_json_checked(args.out, report, field_name="day8_candidate_report")
    print(json.dumps({"status": report["status"], "candidate_quality": report["candidate_quality"], "out": args.out}, sort_keys=True))
    return 0 if report["status"] in {"PASS", "WARN"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
