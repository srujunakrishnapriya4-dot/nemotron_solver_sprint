from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.v2_corpus_io import read_json, write_json_checked
from kaggle_anti086.kaggle_path_safety import require_writable_output_dir
from kaggle_anti086.training.package_submission_guard import build_package_submission_guard
from kaggle_anti086.training.train_v2a_lora import main as train_v2a_main
from kaggle_anti086.training.training_config_schema import load_training_config


def validate_full_train_gate(config: dict[str, Any], *, smoke_report_path: str | Path | None) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    failures: list[str] = []
    if config.get("stage") != "v2a_base_lora":
        failures.append("non_v2a_stage_rejected")
    if not smoke_report_path:
        failures.append("smoke_report_missing")
    else:
        path = Path(smoke_report_path)
        if not path.exists():
            failures.append("smoke_report_missing")
        else:
            smoke = read_json(path)
            if smoke.get("decision") != "ALLOW_DAY8_3_FULL_V2A_TRAINING":
                failures.append("smoke_report_decision_not_allow")
            if smoke.get("full_training_allowed") is not True:
                failures.append("smoke_report_full_training_not_allowed")
            guard = smoke.get("gates", {}).get("package_submission_guard", {})
            if isinstance(guard, dict) and guard.get("status") != "PASS":
                failures.append("package_submission_guard_not_pass")
    package_guard = build_package_submission_guard()
    if package_guard["status"] != "PASS":
        failures.append("package_submission_guard_runtime_fail")
    try:
        require_writable_output_dir(str(config.get("output_adapter_dir", "")), field_name="output_adapter_dir")
    except SystemExit as exc:
        failures.append(f"unsafe_output_adapter_dir:{exc}")
    configured_output = Path(str(config.get("output_adapter_dir", "")))
    if configured_output.exists() and (
        (configured_output / "adapter_config.json").exists()
        or (configured_output / "adapter_model.safetensors").exists()
    ):
        failures.append("configured_output_adapter_dir_already_contains_adapter")
    return warnings, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke-report", required=True)
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument("--out-summary", required=True)
    parser.add_argument("--kaggle-mode", action="store_true")
    args = parser.parse_args(argv)
    config = load_training_config(args.config)
    warnings, failures = validate_full_train_gate(config, smoke_report_path=args.smoke_report)
    if failures:
        summary = {
            "status": "FAIL",
            "stage": config.get("stage"),
            "trained": False,
            "packaging_allowed": False,
            "submission_allowed": False,
            "warnings": warnings,
            "failures": failures,
        }
        write_json_checked(args.out_manifest, {"status": "FAIL", "failures": failures}, field_name="day8_3_v2a_train_manifest")
        write_json_checked(args.out_summary, summary, field_name="day8_3_v2a_train_summary")
        print(json.dumps({"status": "FAIL", "out": args.out_summary}, sort_keys=True))
        return 2
    return train_v2a_main(
        [
            "--config",
            args.config,
            "--collator-audit",
            "artifacts/sprint11/day8_2b_real_collator_audit_report.json",
            "--out-manifest",
            args.out_manifest,
            "--out-summary",
            args.out_summary,
            "--smoke-report",
            args.smoke_report,
            "--kaggle-mode" if args.kaggle_mode else "--dry-run",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
