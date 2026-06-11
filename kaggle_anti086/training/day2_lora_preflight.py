from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

try:
    from kaggle_anti086.training.day2_dataset_loader import (
        discover_day2_datasets,
        read_json,
        summarize_dpo_file,
        summarize_sft_file,
        write_json,
    )
    from kaggle_anti086.training.day2_lora_config_builder import (
        write_default_configs,
    )
except Exception:
    from day2_dataset_loader import (
        discover_day2_datasets,
        read_json,
        summarize_dpo_file,
        summarize_sft_file,
        write_json,
    )
    from day2_lora_config_builder import (
        write_default_configs,
    )


REQUIRED_FINAL_REPORTS = {
    "day1x_final_manifest": "day1x_final_manifest.json",
    "day1x_final_decision": "day1x_final_decision_report.json",
}

MIN_EXPECTED_ROWS = {
    "direct": 50000,
    "short_trace": 50000,
    "mixed_curriculum": 100000,
    "format_heavy": 50000,
    "composed_heavy": 50000,
    "public_style_heavy": 50000,
    "hard_p2_heavy": 30000,
    "dpo_train": 20000,
    "dpo_eval": 2000,
}


def validate_day1x_final_reports(out_dir: Path) -> Dict[str, Any]:
    blocked: List[str] = []
    reports: Dict[str, Any] = {}

    for key, filename in REQUIRED_FINAL_REPORTS.items():
        path = out_dir / filename
        if not path.exists():
            blocked.append(f"missing_report:{filename}")
            reports[key] = {"status": "MISSING", "path": str(path)}
            continue
        try:
            reports[key] = read_json(path)
        except Exception as exc:
            blocked.append(f"broken_report:{filename}:{repr(exc)}")
            reports[key] = {"status": "BROKEN", "path": str(path), "error": repr(exc)}

    final_manifest = reports.get("day1x_final_manifest", {})
    final_decision = reports.get("day1x_final_decision", {})

    if final_manifest.get("status") != "PASS":
        blocked.append(f"day1x_final_manifest_not_pass:{final_manifest.get('status')}")
    if final_decision.get("status") != "PASS":
        blocked.append(f"day1x_final_decision_not_pass:{final_decision.get('status')}")

    if final_manifest.get("ready_for_day2_lora_training") is not True:
        blocked.append("final_manifest_not_ready_for_day2_lora_training")
    if final_decision.get("ready_for_day2_lora_training") is not True:
        blocked.append("final_decision_not_ready_for_day2_lora_training")

    for name, report in reports.items():
        if report.get("package_authorized") is True:
            blocked.append(f"{name}.package_authorized_true")
        if report.get("submission_authorized") is True:
            blocked.append(f"{name}.submission_authorized_true")
        if report.get("leaderboard_claim") is True:
            blocked.append(f"{name}.leaderboard_claim_true")

    safety = final_manifest.get("safety_counters", {})
    for key, value in safety.items():
        try:
            ivalue = int(value)
        except Exception:
            ivalue = 999999
        if ivalue != 0:
            blocked.append(f"final_manifest_safety_counter_nonzero:{key}={value}")

    return {
        "status": "PASS" if not blocked else "FAIL",
        "reports": reports,
        "blocked_reasons": blocked,
    }


def validate_dataset_artifacts(out_dir: Path, sample_limit: int | None = None) -> Dict[str, Any]:
    discovered = discover_day2_datasets(out_dir)
    summaries: Dict[str, Any] = {}
    blocked: List[str] = []

    for key, info in discovered.items():
        if not info["exists"]:
            blocked.append(f"missing_dataset:{key}")
            summaries[key] = info
            continue

        path = Path(info["path"])
        if info["kind"] == "dpo":
            summary = summarize_dpo_file(path, sample_limit=sample_limit)
        else:
            summary = summarize_sft_file(path, sample_limit=sample_limit)

        summaries[key] = {**info, **summary}

        expected = MIN_EXPECTED_ROWS.get(key, 1)
        if not sample_limit and summary.get("row_count", 0) < expected:
            blocked.append(f"dataset_under_expected_count:{key}:{summary.get('row_count')}<{expected}")

        if summary.get("row_count", 0) <= 0:
            blocked.append(f"dataset_empty:{key}")

        if summary.get("problem_count", 0) != 0:
            blocked.append(f"dataset_validation_problems:{key}:{summary.get('problem_count')}")

    return {
        "status": "PASS" if not blocked else "FAIL",
        "datasets": summaries,
        "blocked_reasons": blocked,
    }


def build_lora_preflight_report(out_dir: Path, sample_limit: int | None = None) -> Dict[str, Any]:
    day1x = validate_day1x_final_reports(out_dir)
    datasets = validate_dataset_artifacts(out_dir, sample_limit=sample_limit)
    configs = write_default_configs(out_dir)

    variant_matrix = configs["variant_matrix"]
    training_config_manifest = configs["training_config_manifest"]

    blocked: List[str] = []
    if day1x["status"] != "PASS":
        blocked.append("day1x_final_reports_not_pass")
        blocked.extend(day1x["blocked_reasons"])
    if datasets["status"] != "PASS":
        blocked.append("dataset_artifacts_not_pass")
        blocked.extend(datasets["blocked_reasons"])
    if variant_matrix.get("status") != "PASS":
        blocked.append("variant_matrix_not_pass")
        blocked.extend(variant_matrix.get("blocked_reasons", []))
    if training_config_manifest.get("status") != "PASS":
        blocked.append("training_config_manifest_not_pass")
        blocked.extend(training_config_manifest.get("blocked_reasons", []))

    dpo_variants = [
        v for v in variant_matrix.get("variants", [])
        if v.get("uses_dpo")
    ]
    for v in dpo_variants:
        if not v.get("requires_sft_baseline_adapter"):
            blocked.append(f"dpo_variant_does_not_require_sft_baseline:{v.get('adapter_name')}")

    status = "PASS" if not blocked else "FAIL"

    return {
        "schema_version": 1,
        "created_by": "DAY2_LORA_PREFLIGHT_AND_CONFIG_LOCK",
        "status": status,
        "day1x_final_reports_status": day1x["status"],
        "dataset_artifacts_status": datasets["status"],
        "variant_matrix_status": variant_matrix.get("status"),
        "training_config_manifest_status": training_config_manifest.get("status"),
        "sample_limited": sample_limit is not None,
        "sample_limit": sample_limit,
        "day1x": day1x,
        "datasets": datasets,
        "variant_matrix_file": str(out_dir / "day2_lora_variant_matrix.json"),
        "training_config_manifest_file": str(out_dir / "day2_training_config_manifest.json"),
        "training_authorized": status == "PASS",
        "package_authorized": False,
        "submission_authorized": False,
        "leaderboard_claim": False,
        "no_0_93_evidence": True,
        "no_0_95_evidence": True,
        "next_phase": "PHASE8_ADAPTER_SMOKE_TRAINING_AND_DELTA_EVAL",
        "blocked_reasons": blocked,
    }


def run(out_dir: Path, sample_limit: int | None = None) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_lora_preflight_report(out_dir, sample_limit=sample_limit)
    write_json(out_dir / "day2_lora_preflight_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="artifacts/sprint11")
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Optional fast validation limit per file. Omit for full validation.",
    )
    args = parser.parse_args()

    report = run(Path(args.out_dir), sample_limit=args.sample_limit)

    summary = {
        "status": report["status"],
        "day1x_final_reports_status": report["day1x_final_reports_status"],
        "dataset_artifacts_status": report["dataset_artifacts_status"],
        "variant_matrix_status": report["variant_matrix_status"],
        "training_config_manifest_status": report["training_config_manifest_status"],
        "training_authorized": report["training_authorized"],
        "package_authorized": report["package_authorized"],
        "submission_authorized": report["submission_authorized"],
        "leaderboard_claim": report["leaderboard_claim"],
        "next_phase": report["next_phase"],
        "blocked_reasons": report["blocked_reasons"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
