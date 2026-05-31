from __future__ import annotations

import argparse, json, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.v2_corpus_io import write_json_checked
from kaggle_anti086.training.train_v2a_lora import main as train_main
from kaggle_anti086.training.training_config_schema import load_training_config


def build_local_smoke_summary(steps: int) -> dict:
    return {
        "status": "NEEDS_KAGGLE_SMOKE_TRAIN",
        "steps_completed": 0,
        "requested_steps": steps,
        "adapter_dir": None,
        "adapter_valid": False,
        "loss_start": None,
        "loss_end": None,
        "loss_nan_detected": False,
        "artifact_audit_status": "MISSING",
        "smoke_eval_status": "SKIPPED",
        "warnings": ["local_smoke_training_not_run"],
        "failures": [],
    }


def build_smoke_summary(train_summary: dict, artifact_report: dict, smoke_eval_report: dict) -> dict:
    failures = []
    if train_summary.get("status") != "PASS":
        failures.append("smoke_train_not_pass")
    if int(train_summary.get("steps_completed", 0) or 0) <= 0:
        failures.append("smoke_steps_zero")
    if train_summary.get("loss_nan_detected"):
        failures.append("loss_nan_detected")
    if artifact_report.get("status") != "PASS":
        failures.append("adapter_artifact_not_pass")
    if smoke_eval_report.get("status") == "FAIL":
        failures.append("smoke_eval_failed")
    return {
        "status": "PASS" if not failures else "FAIL",
        "steps_completed": int(train_summary.get("steps_completed", 0) or 0),
        "adapter_dir": train_summary.get("adapter_dir"),
        "adapter_valid": artifact_report.get("status") == "PASS",
        "loss_start": train_summary.get("loss_start"),
        "loss_end": train_summary.get("loss_end"),
        "loss_nan_detected": bool(train_summary.get("loss_nan_detected", False)),
        "artifact_audit_status": artifact_report.get("status", "MISSING"),
        "smoke_eval_status": smoke_eval_report.get("status", "MISSING"),
        "warnings": [],
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--collator-audit", required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--out-summary", required=True)
    parser.add_argument("--kaggle-mode", action="store_true")
    args = parser.parse_args(argv)
    config = load_training_config(args.config)
    if config.get("stage") != "v2a_base_lora":
        report = {
            "status": "FAIL",
            "steps_completed": 0,
            "requested_steps": args.steps,
            "adapter_dir": None,
            "adapter_valid": False,
            "loss_start": None,
            "loss_end": None,
            "loss_nan_detected": False,
            "artifact_audit_status": "MISSING",
            "smoke_eval_status": "SKIPPED",
            "warnings": [],
            "failures": ["non_v2a_stage_rejected"],
        }
        write_json_checked(args.out_summary, report, field_name="day8_2_smoke_train_summary")
        print(json.dumps({"status": report["status"], "out": args.out_summary}, sort_keys=True))
        return 2
    if not args.kaggle_mode:
        report = build_local_smoke_summary(args.steps)
        write_json_checked(args.out_summary, report, field_name="day8_2_smoke_train_summary")
        print(json.dumps({"status": report["status"], "out": args.out_summary}, sort_keys=True))
        return 0
    # Reuse the v2a trainer in smoke mode. The trainer itself enforces all gates.
    manifest = str(Path(args.out_summary).with_name("day8_2_smoke_train_manifest.json"))
    return train_main([
        "--config", args.config,
        "--collator-audit", args.collator_audit,
        "--out-manifest", manifest,
        "--out-summary", args.out_summary,
        "--kaggle-mode",
        "--smoke-steps", str(args.steps),
    ])


if __name__ == "__main__":
    raise SystemExit(main())
