from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.v2_corpus_io import write_json_checked
from kaggle_anti086.training.training_config_schema import load_training_config


SMALL_TARGETS = {"q_proj", "v_proj", "o_proj"}
WIDE_TARGETS = SMALL_TARGETS | {"k_proj", "up_proj", "down_proj"}
FORBIDDEN_TARGETS = {"lm_head", "embed_tokens"}


def audit_configs(config_paths: list[str | Path], *, available_modules: set[str] | None = None) -> dict[str, Any]:
    reports = []
    failures = []
    small_viable = False
    for path in config_paths:
        report = audit_config(path, available_modules=available_modules)
        reports.append(report)
        if report["stage"] == "v4_solver_teacher_lora_small" and report["status"] in {"PASS", "PASS_TARGETS_UNVERIFIED"}:
            small_viable = True
        if report["status"] == "FAIL":
            failures.append(f"{Path(path).name}:fail")
    status = "PASS" if small_viable and not failures else "FAIL"
    return {
        "status": status,
        "small_config_viable": small_viable,
        "wide_config_status": next((item["status"] for item in reports if item["stage"] == "v4_solver_teacher_lora_wide"), "MISSING"),
        "configs": reports,
        "failures": failures if not small_viable else [],
        "warnings": [] if any(item["target_modules_confirmed"] for item in reports) else ["target_modules_unverified_without_real_model"],
        "packaging_allowed": False,
        "submission_allowed": False,
        "no_training_performed": True,
    }


def audit_config(path: str | Path, *, available_modules: set[str] | None = None) -> dict[str, Any]:
    config = load_training_config(path)
    failures: list[str] = []
    warnings: list[str] = []
    stage = str(config.get("stage", ""))
    rank = int(config.get("rank", 0) or 0)
    targets = _targets(config)
    if stage not in {"v4_solver_teacher_lora_small", "v4_solver_teacher_lora_wide"}:
        failures.append("not_day10_v4_stage")
    if rank <= 0 or rank > 32:
        failures.append("rank_gt_32_or_invalid")
    if FORBIDDEN_TARGETS & set(targets):
        failures.append("forbidden_target_module")
    if stage.endswith("_small") and set(targets) != SMALL_TARGETS:
        failures.append("small_targets_must_be_q_v_o")
    if stage.endswith("_wide") and not set(targets).issubset(WIDE_TARGETS):
        failures.append("wide_targets_outside_allowed_set")
    if config.get("full_prompt_loss") is not False:
        failures.append("full_prompt_loss_must_be_false")
    if config.get("assistant_only_loss") is not True:
        failures.append("assistant_only_loss_must_be_true")
    if config.get("train_on_user") is not False:
        failures.append("train_on_user_must_be_false")
    if float(config.get("abstain_weight", 1.0)) != 0.0:
        failures.append("abstain_weight_must_be_zero")
    size_estimate = estimate_adapter_size_mb(targets, rank)
    trainable_estimate = estimate_trainable_params(targets, rank)
    if size_estimate > float(config.get("adapter_size_limit_mb", 0) or 0):
        failures.append("adapter_size_estimate_exceeds_limit")
    target_modules_confirmed = False
    missing_targets: list[str] = []
    if available_modules is not None:
        missing_targets = sorted(set(targets) - available_modules)
        target_modules_confirmed = not missing_targets
        if missing_targets:
            failures.append("target_modules_missing")
    else:
        warnings.append("target_modules_unverified_without_real_model")
    status = "FAIL" if failures else ("PASS" if target_modules_confirmed else "PASS_TARGETS_UNVERIFIED")
    if stage.endswith("_wide") and not target_modules_confirmed and not failures:
        status = "BLOCKED_TARGETS_UNCONFIRMED"
    return {
        "path": str(path),
        "stage": stage,
        "status": status,
        "target_modules": targets,
        "target_modules_confirmed": target_modules_confirmed,
        "missing_target_modules": missing_targets,
        "rank": rank,
        "adapter_size_estimate_mb": size_estimate,
        "trainable_parameter_estimate": trainable_estimate,
        "adapter_size_limit_mb": float(config.get("adapter_size_limit_mb", 0) or 0),
        "failures": failures,
        "warnings": warnings,
    }


def estimate_adapter_size_mb(targets: list[str], rank: int) -> float:
    # Conservative placeholder until real model dimensions are inspected in Kaggle.
    per_module_mb = 18.0
    return round(len(targets) * max(rank, 1) / 32.0 * per_module_mb, 3)


def estimate_trainable_params(targets: list[str], rank: int) -> int:
    per_module_params = 4_194_304
    return int(len(targets) * max(rank, 1) / 32.0 * per_module_params)


def _targets(config: dict[str, Any]) -> list[str]:
    value = config.get("target_modules", [])
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit Day 10 LoRA capacity/configs before training.")
    parser.add_argument("--configs", nargs="+", default=[
        "kaggle_anti086/training/configs/v4_solver_teacher_lora_small.yaml",
        "kaggle_anti086/training/configs/v4_solver_teacher_lora_wide.yaml",
    ])
    parser.add_argument("--available-modules", default="", help="Comma-separated module names from a real model inspection.")
    parser.add_argument("--out", default="artifacts/sprint11/day10_lora_capacity_audit.json")
    args = parser.parse_args(argv)
    modules = {item.strip() for item in args.available_modules.split(",") if item.strip()} or None
    report = audit_configs(args.configs, available_modules=modules)
    write_json_checked(args.out, report, field_name="day10_lora_capacity_audit")
    print(json.dumps({"status": report["status"], "small_config_viable": report["small_config_viable"], "out": args.out}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
