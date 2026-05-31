from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.v2_corpus_io import read_json, write_json_checked


DEFAULT_REPORTS = {
    "config_validation": "artifacts/sprint11/day7_config_validation_report.json",
    "tokenization": "artifacts/sprint11/day7_tokenization_dry_run_report.json",
    "loss_mask": "artifacts/sprint11/day7_loss_mask_report.json",
    "sampling_policy_consumption": "artifacts/sprint11/day7_sampling_policy_consumption_report.json",
    "adapter_constraint": "artifacts/sprint11/day7_adapter_constraint_report.json",
    "source_leakage": "artifacts/sprint11/train_v2_source_leakage_audit.json",
    "day6_sampling_policy": "artifacts/sprint11/train_v2_sampling_policy.json",
}


def build_day7_readiness(report_paths: dict[str, str | Path] | None = None) -> dict:
    report_paths = report_paths or DEFAULT_REPORTS
    statuses = {}
    failures = []
    for name, path in report_paths.items():
        target = Path(path)
        if not target.exists():
            statuses[name] = "MISSING"
            failures.append(f"{name}_missing")
            continue
        data = read_json(target)
        status = str(data.get("status", "FAIL"))
        statuses[name] = status
        if status == "FAIL":
            failures.append(f"{name}_failed")
    config = read_json(report_paths["config_validation"]) if Path(report_paths["config_validation"]).exists() else {}
    if config.get("runnable_count", 0) < 1:
        failures.append("no_runnable_v2_config")
    day7_status = "PASS" if not failures else "FAIL"
    return {
        "day7_status": day7_status,
        "status": day7_status,
        "decision": "ALLOW_DAY8_V2_TRAINING" if day7_status == "PASS" else "BLOCK_TRAINING",
        "training_allowed_today": False,
        "training_allowed_next_stage": day7_status == "PASS",
        "packaging_allowed": False,
        "submission_allowed": False,
        "no_leaderboard_evidence": True,
        "no_0_95_evidence": True,
        "no_adapter_packaged": True,
        "no_training_performed": True,
        "reports": statuses,
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = build_day7_readiness()
    write_json_checked(args.out, report, field_name="day7_training_readiness_report")
    print(json.dumps({"status": report["status"], "decision": report["decision"], "out": args.out}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
