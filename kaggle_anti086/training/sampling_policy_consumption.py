from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.v2_corpus_io import read_json, read_jsonl, write_json_checked
from kaggle_anti086.training.prepare_tokenization_dry_run import UNSUPPORTED_SFT
from kaggle_anti086.training.training_config_schema import load_training_config


def build_sampling_policy_consumption(config: dict[str, Any], policy_path: str | Path = "artifacts/sprint11/train_v2_sampling_policy.json") -> dict[str, Any]:
    policy = read_json(policy_path)
    row_policy = policy.get("row_policy", {})
    direct = read_jsonl(config["train_direct_path"])
    corrected = read_jsonl(config["train_solver_corrected_path"])
    abstain = read_jsonl(config["train_abstain_safety_path"])
    hardneg = read_jsonl(config["train_hard_negative_path"])
    failures = []
    family_weights: dict[str, float] = defaultdict(float)
    source_weights: dict[str, float] = defaultdict(float)
    corrected_weight = 0.0
    unsupported = 0
    unverified = 0
    for source, rows in (("direct", direct), ("solver_corrected", corrected), ("abstain_safety", abstain), ("hard_negative", hardneg)):
        for row in rows:
            rp = row_policy.get(str(row.get("id")), {})
            allowed = bool(rp.get("allowed_for_sft", False))
            weight = float(rp.get("sampling_weight", 0.0))
            if source == "direct" and (not allowed or weight != 1.0):
                failures.append("direct_policy_invalid")
            if source == "solver_corrected":
                if not allowed:
                    failures.append("solver_corrected_weight_missing")
                if weight > 1.0 or weight > float(config.get("direct_answer_weight", 1.0)):
                    failures.append("solver_corrected_weight_gt_direct")
                corrected_weight += weight
            if source == "abstain_safety" and allowed:
                failures.append("abstain_safety_enters_sft")
            if source == "hard_negative" and allowed:
                failures.append("hard_negative_enters_sft")
            if allowed and row.get("family") in UNSUPPORTED_SFT:
                unsupported += 1
            if allowed and row.get("verification_status", "verified") != "verified":
                unverified += 1
            if allowed:
                family_weights[str(row.get("family", "unknown"))] += weight
                source_weights[source] += weight
    if unsupported:
        failures.append("unsupported_family_enters_sft")
    if unverified:
        failures.append("unverified_row_enters_sft")
    return {
        "status": "PASS" if not failures else "FAIL",
        "direct_answer_rows": len(direct),
        "solver_corrected_rows": len(corrected),
        "solver_corrected_effective_weight": corrected_weight,
        "abstain_safety_rows": len(abstain),
        "abstain_safety_sft_rows": sum(1 for row in abstain if row_policy.get(str(row.get("id")), {}).get("allowed_for_sft")),
        "hard_negative_rows": len(hardneg),
        "hard_negative_sft_rows": sum(1 for row in hardneg if row_policy.get(str(row.get("id")), {}).get("allowed_for_sft")),
        "unsupported_sft_rows": unsupported,
        "unverified_sft_rows": unverified,
        "effective_sft_rows": sum(source_weights.values()),
        "family_effective_weights": dict(sorted(family_weights.items())),
        "source_effective_weights": dict(sorted(source_weights.items())),
        "warnings": [],
        "failures": sorted(set(failures)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--policy", default="artifacts/sprint11/train_v2_sampling_policy.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = build_sampling_policy_consumption(load_training_config(args.config), args.policy)
    write_json_checked(args.out, report, field_name="day7_sampling_policy_consumption_report")
    print(json.dumps({"status": report["status"], "effective_sft_rows": report["effective_sft_rows"], "out": args.out}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
