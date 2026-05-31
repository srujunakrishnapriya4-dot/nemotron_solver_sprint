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
from kaggle_anti086.kaggle_path_safety import require_writable_output_dir
from kaggle_anti086.training.day8_1_backend_readiness import build_backend_readiness
from kaggle_anti086.training.gpu_memory_audit import capture_gpu_memory_snapshot
from kaggle_anti086.training.lora_backend import prepare_model_for_v2a_training, run_lora_training
from kaggle_anti086.training.model_lifecycle import free_objects
from kaggle_anti086.training.model_loader import load_base_model, load_tokenizer
from kaggle_anti086.training.run_provenance import build_run_provenance
from kaggle_anti086.training.sft_dataset import Sprint11SFTDataset, build_sft_dataset_report, load_weighted_sft_rows
from kaggle_anti086.training.training_config_schema import STATUS_PASS_RUNNABLE, _target_modules, load_training_config, validate_training_config
from kaggle_anti086.training.training_log_history import build_training_log_history
from kaggle_anti086.training.training_run_manifest import build_run_manifest
from kaggle_anti086.training.weighted_sft_sampler import build_weighted_sample, build_weighted_sampling_report


def preflight_v2a_training(
    config: dict[str, Any],
    *,
    config_path: str | Path,
    collator_audit_path: str | Path,
    smoke_steps: int | None = None,
    require_day8_2_readiness: bool = False,
) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    failures: list[str] = []
    day7 = Path("artifacts/sprint11/day7_training_readiness_report.json")
    if not day7.exists():
        failures.append("day7_readiness_missing")
    else:
        report = read_json(day7)
        if report.get("decision") != "ALLOW_DAY8_V2_TRAINING" or report.get("training_allowed_next_stage") is not True:
            failures.append("day7_not_allowing_day8")
    if require_day8_2_readiness and not smoke_steps:
        day8_2 = Path("artifacts/sprint11/day8_2_readiness_report.json")
        if not day8_2.exists():
            failures.append("day8_2_readiness_missing")
        else:
            readiness = read_json(day8_2)
            if readiness.get("decision") != "ALLOW_DAY8_3_FULL_V2A_TRAINING" or readiness.get("full_training_allowed") is not True:
                failures.append("day8_2_not_allowing_full_training")
    validation = validate_training_config(config, path=config_path)
    if validation.status != STATUS_PASS_RUNNABLE or config.get("stage") != "v2a_base_lora":
        failures.append("config_not_runnable_v2a")
    audit = read_json(collator_audit_path) if Path(collator_audit_path).exists() else {"status": "MISSING"}
    if audit.get("status") != "PASS":
        failures.append("real_collator_audit_not_pass")
    for path, key in (
        ("artifacts/sprint11/day7_sampling_policy_consumption_report.json", "sampling_policy_not_pass"),
        ("artifacts/sprint11/train_v2_source_leakage_audit.json", "source_leakage_not_pass"),
        ("artifacts/sprint11/day7_adapter_constraint_report.json", "adapter_constraints_not_pass"),
    ):
        if not Path(path).exists() or read_json(path).get("status") != "PASS":
            failures.append(key)
    try:
        require_writable_output_dir(str(config.get("output_adapter_dir", "")), field_name="output_adapter_dir")
    except SystemExit as exc:
        failures.append(f"unsafe_output_adapter_dir:{exc}")
    if int(config.get("rank", 0)) > 32:
        failures.append("rank_gt_32")
    if set(_target_modules(config)) != {"q_proj", "v_proj", "o_proj"}:
        failures.append("invalid_target_modules")
    if config.get("full_prompt_loss") is not False:
        failures.append("full_prompt_loss_enabled")
    if config.get("assistant_only_loss") is not True:
        failures.append("assistant_only_loss_disabled")
    if config.get("train_on_user") is not False:
        failures.append("train_on_user_enabled")
    return warnings, failures


def build_training_summary(config: dict[str, Any], manifest: dict[str, Any], *, dry_run: bool, failures: list[str], warnings: list[str], backend_summary: dict[str, Any] | None = None) -> dict[str, Any]:
    backend_summary = backend_summary or {}
    trained = bool(backend_summary.get("status") == "PASS") and not dry_run and not failures
    return {
        "status": "DRY_RUN_PASS" if dry_run and not failures else ("PASS" if trained else "FAIL"),
        "stage": "v2a_base_lora",
        "run_id": manifest.get("run_id"),
        "trained": trained,
        "dry_run": dry_run,
        "steps_completed": int(backend_summary.get("steps_completed", 0) or 0),
        "loss_start": backend_summary.get("loss_start"),
        "loss_end": backend_summary.get("loss_end"),
        "loss_min": backend_summary.get("loss_min"),
        "loss_max": backend_summary.get("loss_max"),
        "loss_nan_detected": bool(backend_summary.get("loss_nan_detected", False)),
        "grad_nan_detected": bool(backend_summary.get("grad_nan_detected", False)),
        "adapter_dir": manifest.get("output_adapter_dir"),
        "adapter_config_exists": bool(backend_summary.get("adapter_config_exists", False)),
        "adapter_model_exists": bool(backend_summary.get("adapter_model_exists", False)),
        "rank": int(config.get("rank", 0)),
        "target_modules": _target_modules(config),
        "dataset_report": backend_summary.get("dataset_report"),
        "smoke_steps": int(config.get("num_steps", 0)) if int(config.get("num_steps", 0) or 0) <= 5 else None,
        "memory_snapshots": backend_summary.get("memory_snapshots", []),
        "training_log_history_path": backend_summary.get("training_log_history_path"),
        "run_provenance_path": backend_summary.get("run_provenance_path"),
        "packaging_allowed": False,
        "submission_allowed": False,
        "warnings": warnings + (["dry_run_no_training_performed"] if dry_run else []),
        "failures": failures + list(backend_summary.get("failures", [])),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--collator-audit", required=True)
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument("--out-summary", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--kaggle-mode", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=None)
    parser.add_argument("--require-day8-2-readiness", action="store_true")
    parser.add_argument("--smoke-report", default="artifacts/sprint11/day8_2b_smoke_orchestrator_report.json")
    args = parser.parse_args(argv)
    config = load_training_config(args.config)
    if args.smoke_steps is not None:
        if args.smoke_steps < 1 or args.smoke_steps > 5:
            raise SystemExit("--smoke-steps must be between 1 and 5")
        config = dict(config)
        config["num_steps"] = args.smoke_steps
    warnings, failures = preflight_v2a_training(
        config,
        config_path=args.config,
        collator_audit_path=args.collator_audit,
        smoke_steps=args.smoke_steps,
        require_day8_2_readiness=args.require_day8_2_readiness,
    )
    if not args.dry_run and args.smoke_steps is None:
        smoke_report = Path(args.smoke_report)
        if not smoke_report.exists():
            failures.append("smoke_report_missing_for_full_training")
        else:
            smoke = read_json(smoke_report)
            if smoke.get("decision") != "ALLOW_DAY8_3_FULL_V2A_TRAINING" or smoke.get("full_training_allowed") is not True:
                failures.append("smoke_report_not_allowing_full_training")
            guard = smoke.get("gates", {}).get("package_submission_guard", {})
            if isinstance(guard, dict) and guard.get("status") not in {"PASS", None}:
                failures.append("package_submission_guard_not_pass")
    try:
        manifest = build_run_manifest(
            config,
            config_path=args.config,
            collator_audit_path=args.collator_audit,
            run_kind="smoke" if args.smoke_steps is not None else "full",
        )
    except Exception as exc:
        manifest = {"run_id": None, "stage": config.get("stage"), "output_adapter_dir": None, "packaging_allowed": False, "submission_allowed": False, "failures": [str(exc)]}
        failures.append(f"manifest_failed:{type(exc).__name__}")
    backend_summary: dict[str, Any] | None = None
    memory_snapshots: list[dict[str, Any]] = []
    if args.kaggle_mode and not args.dry_run and not failures:
        readiness = build_backend_readiness(args.config, args.collator_audit)
        if readiness["status"] != "PASS":
            failures.append("backend_readiness_not_pass")
            failures.extend(readiness.get("failures", []))
        else:
            tokenizer = load_tokenizer(str(config["base_model_path"]))
            all_rows = load_weighted_sft_rows(config)
            rows = build_weighted_sample(all_rows, seed=int(config.get("seed", 42)))
            sampling_report = build_weighted_sampling_report(all_rows, rows, seed=int(config.get("seed", 42)))
            dataset_report = build_sft_dataset_report(config, rows)
            dataset_report["weighted_sampling_report"] = sampling_report
            if dataset_report["status"] != "PASS":
                failures.append("sft_dataset_not_pass")
            if sampling_report["status"] != "PASS":
                failures.append("weighted_sampling_not_pass")
            if dataset_report["status"] == "PASS" and sampling_report["status"] == "PASS":
                dataset = Sprint11SFTDataset(rows, tokenizer, int(config.get("max_seq_len", 1024)))
                memory_snapshots.append(capture_gpu_memory_snapshot("train_before_model_load", kaggle_mode=True))
                model = load_base_model(str(config["base_model_path"]), load_in_4bit=bool(config.get("load_in_4bit", False)), bf16=True)
                memory_snapshots.append(capture_gpu_memory_snapshot("train_after_model_load", kaggle_mode=True))
                model = prepare_model_for_v2a_training(model, config)
                memory_snapshots.append(capture_gpu_memory_snapshot("train_after_lora_prepare", kaggle_mode=True))
                memory_snapshots.append(capture_gpu_memory_snapshot("train_before_train", kaggle_mode=True))
                backend_summary = run_lora_training(model, tokenizer, dataset, config, manifest["output_adapter_dir"])
                memory_snapshots.append(capture_gpu_memory_snapshot("train_after_train_and_save", kaggle_mode=True))
                free_objects(model, tokenizer, reason="after_train_v2a")
                memory_snapshots.append(capture_gpu_memory_snapshot("train_after_cleanup", kaggle_mode=True))
                backend_summary["dataset_report"] = dataset_report
                backend_summary["memory_snapshots"] = memory_snapshots
                log_path = Path(args.out_summary).with_name(Path(args.out_summary).stem + "_log_history.json")
                log_report = build_training_log_history(backend_summary)
                write_json_checked(log_path, log_report, field_name="day8_train_log_history")
                backend_summary["training_log_history_path"] = str(log_path)
                provenance_path = Path(args.out_summary).with_name(Path(args.out_summary).stem + "_run_provenance.json")
                provenance = build_run_provenance(args.config, collator_audit=args.collator_audit)
                write_json_checked(provenance_path, provenance, field_name="day8_train_run_provenance")
                backend_summary["run_provenance_path"] = str(provenance_path)
    elif not args.dry_run and not failures:
        failures.append("real_training_requires_kaggle_mode")
    write_json_checked(args.out_manifest, manifest, field_name="day8_v2a_train_manifest")
    summary = build_training_summary(config, manifest, dry_run=args.dry_run, failures=failures, warnings=warnings, backend_summary=backend_summary)
    write_json_checked(args.out_summary, summary, field_name="day8_v2a_train_summary")
    print(json.dumps({"status": summary["status"], "trained": summary["trained"], "dry_run": summary["dry_run"], "out": args.out_summary}, sort_keys=True))
    return 0 if summary["status"] in {"PASS", "DRY_RUN_PASS"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
