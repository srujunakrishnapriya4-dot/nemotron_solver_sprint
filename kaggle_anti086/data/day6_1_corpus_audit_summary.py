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


DEFAULT_REPORTS = {
    "source_leakage_audit": "artifacts/sprint11/train_v2_source_leakage_audit.json",
    "independence_report": "artifacts/sprint11/train_v2_independence_report.json",
    "prompt_diversity_report": "artifacts/sprint11/train_v2_prompt_diversity_report.json",
    "sampling_policy": "artifacts/sprint11/train_v2_sampling_policy.json",
    "hard_negative_taxonomy": "artifacts/sprint11/train_v2_hard_negative_taxonomy.json",
}


def build_day6_1_summary(report_paths: dict[str, str | Path] | None = None) -> dict[str, Any]:
    report_paths = report_paths or DEFAULT_REPORTS
    reports: dict[str, str] = {}
    warnings: list[str] = []
    blocking: list[str] = []
    for name, path in report_paths.items():
        target = Path(path)
        if not target.exists():
            reports[name] = "MISSING"
            blocking.append(f"{name}_missing")
            continue
        status = str(read_json(target).get("status", "FAIL"))
        reports[name] = status
        if status == "FAIL":
            blocking.append(f"{name}_failed")
        elif status == "WARN":
            warnings.append(f"{name}_warn")
    hard_warn_only = set(warnings) <= {"hard_negative_taxonomy_warn"}
    decision = "ALLOW_DAY7_TRAINING_CONFIG_PREP" if not blocking and (not warnings or hard_warn_only or all(report != "FAIL" for report in reports.values())) else "BLOCK_DAY7_TRAINING_CONFIG_PREP"
    status = "FAIL" if blocking else ("WARN" if warnings else "PASS")
    return {
        "status": status,
        "decision": decision,
        "training_allowed": False,
        "packaging_allowed": False,
        "submission_allowed": False,
        "no_leaderboard_evidence": True,
        "no_0_95_evidence": True,
        "reports": reports,
        "blocking_failures": blocking,
        "warnings": warnings,
        "remaining_blockers_before_training": [
            "Day 7 training configs not validated yet.",
            "Base/parent/child training configs not prepared yet.",
            "No v2 adapter trained.",
            "No post-training eval.",
            "No public leaderboard calibration.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    summary = build_day6_1_summary()
    write_json_checked(args.out, summary, field_name="day6_1_corpus_audit_summary")
    print(json.dumps({"status": summary["status"], "decision": summary["decision"], "out": args.out}, sort_keys=True))
    return 0 if summary["decision"] == "ALLOW_DAY7_TRAINING_CONFIG_PREP" else 2


if __name__ == "__main__":
    raise SystemExit(main())
