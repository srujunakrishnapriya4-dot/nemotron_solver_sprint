from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.v2_corpus_io import write_json_checked
from kaggle_anti086.training.day8_2b_smoke_orchestrator import PATHS, build_gate_entry
from kaggle_anti086.training.final_sft_leakage_audit import audit_final_sft_leakage
from kaggle_anti086.training.gpu_memory_audit import capture_gpu_memory_snapshot, write_gpu_memory_report
from kaggle_anti086.training.lora_target_inspector import verify_lora_targets
from kaggle_anti086.training.model_environment_report import build_model_environment_report
from kaggle_anti086.training.model_lifecycle import free_objects
from kaggle_anti086.training.model_loader import load_base_model
from kaggle_anti086.training.package_submission_guard import build_package_submission_guard
from kaggle_anti086.training.real_tokenizer_collator_audit import build_real_collator_audit
from kaggle_anti086.training.sft_dataset import load_weighted_sft_rows
from kaggle_anti086.training.supervised_label_decode_audit import audit_supervised_label_decode
from kaggle_anti086.training.trainable_parameter_audit import audit_trainable_parameters
from kaggle_anti086.training.training_config_schema import _target_modules, load_training_config
from kaggle_anti086.training.truncation_audit import build_truncation_audit
from kaggle_anti086.training.weighted_sft_sampler import build_weighted_sample, build_weighted_sampling_report


ART = Path("artifacts/sprint11")
PREFLIGHT_MEMORY = ART / "day8_2c_gpu_memory_report.json"


def build_smoke_preflight_report(config_path: str | Path, *, kaggle_mode: bool = False) -> dict[str, Any]:
    config = load_training_config(config_path)
    gates: dict[str, dict[str, str]] = {}
    warnings: list[str] = []
    failures: list[str] = []
    memory_snapshots: list[dict[str, Any]] = []
    if config.get("stage") != "v2a_base_lora":
        return _report("FAIL", "BLOCK_PREFLIGHT_RUNTIME", gates, warnings, ["non_v2a_stage_rejected"], memory_snapshots)

    def write_gate(name: str, report: dict[str, Any]) -> dict[str, Any]:
        path = PATHS.get(name, ART / f"day8_2c_{name}.json")
        write_json_checked(path, report, field_name=f"day8_2c_{name}")
        gates[name] = build_gate_entry(str(report.get("status", "FAIL")), path)
        return report

    collator = write_gate("collator_audit", build_real_collator_audit(config, sample_size=128, kaggle_mode=kaggle_mode))
    if collator.get("status") != "PASS":
        if kaggle_mode or collator.get("status") != "WARN_LOCAL_TOKENIZER_UNAVAILABLE":
            failures.append("collator")
    items = load_weighted_sft_rows(config)
    selected = build_weighted_sample(items, seed=int(config.get("seed", 42)))
    if write_gate("weighted_sampling", build_weighted_sampling_report(items, selected, seed=int(config.get("seed", 42)))).get("status") != "PASS":
        failures.append("runtime")
    if write_gate("truncation", build_truncation_audit(config)).get("status") != "PASS":
        failures.append("labels")
    if write_gate("label_decode", audit_supervised_label_decode(config)).get("status") != "PASS":
        failures.append("labels")
    if write_gate("leakage", audit_final_sft_leakage(selected)).get("status") != "PASS":
        failures.append("leakage")
    env = write_gate("model_environment", build_model_environment_report(config, kaggle_mode=kaggle_mode))
    if env.get("status") != "PASS":
        if kaggle_mode:
            failures.append("runtime")
        else:
            warnings.append("needs_kaggle_model_stack")
    if write_gate("package_submission_guard", build_package_submission_guard()).get("status") != "PASS":
        failures.append("runtime")
    if failures:
        return _report("FAIL", _decision(failures[0]), gates, warnings, failures, memory_snapshots)
    if not kaggle_mode:
        memory_snapshots.append(capture_gpu_memory_snapshot("local_preflight_no_model_stack"))
        write_gpu_memory_report(PREFLIGHT_MEMORY, memory_snapshots)
        return _report("WARN", "NEEDS_KAGGLE_MODEL_STACK", gates, warnings, failures, memory_snapshots)

    try:
        memory_snapshots.append(capture_gpu_memory_snapshot("before_preflight_model_load", kaggle_mode=True))
        model = load_base_model(str(config["base_model_path"]), load_in_4bit=bool(config.get("load_in_4bit", False)), bf16=True)
        memory_snapshots.append(capture_gpu_memory_snapshot("after_preflight_model_load", kaggle_mode=True))
        target = write_gate("lora_target", verify_lora_targets(model, _target_modules(config)))
        if target.get("status") != "PASS":
            failures.append("lora_target")
        from kaggle_anti086.training.lora_backend import prepare_model_for_v2a_training

        lora_model = prepare_model_for_v2a_training(model, config)
        memory_snapshots.append(capture_gpu_memory_snapshot("after_preflight_lora_prepare", kaggle_mode=True))
        trainable = write_gate("trainable_params", audit_trainable_parameters(lora_model))
        if trainable.get("status") != "PASS":
            failures.append("trainable_params")
        free_objects(lora_model, model, reason="preflight_cleanup")
        memory_snapshots.append(capture_gpu_memory_snapshot("after_preflight_cleanup", kaggle_mode=True))
    except Exception as exc:
        failures.append(f"runtime:{type(exc).__name__}")
    memory_report = write_gpu_memory_report(PREFLIGHT_MEMORY, memory_snapshots)
    if memory_report.get("status") == "FAIL":
        failures.append("memory")
    if failures:
        return _report("FAIL", _decision(failures[0]), gates, warnings, failures, memory_snapshots)
    return _report("PASS", "PASS_PREFLIGHT_READY_FOR_SMOKE_TRAIN", gates, warnings, failures, memory_snapshots)


def _decision(reason: str) -> str:
    if reason == "collator":
        return "BLOCK_PREFLIGHT_COLLATOR"
    if reason == "lora_target":
        return "BLOCK_PREFLIGHT_LORA_TARGET"
    if reason == "trainable_params":
        return "BLOCK_PREFLIGHT_TRAINABLE_PARAMS"
    if reason == "memory":
        return "BLOCK_PREFLIGHT_MEMORY"
    if reason == "leakage":
        return "BLOCK_PREFLIGHT_LEAKAGE"
    if reason == "labels":
        return "BLOCK_PREFLIGHT_LABELS"
    return "BLOCK_PREFLIGHT_RUNTIME"


def _report(status: str, decision: str, gates: dict[str, Any], warnings: list[str], failures: list[str], memory_snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "status": status,
        "decision": decision,
        "gates": gates,
        "memory_snapshots": memory_snapshots,
        "warnings": warnings,
        "failures": failures,
        "packaging_allowed": False,
        "submission_allowed": False,
        "no_leaderboard_evidence": True,
        "no_0_95_evidence": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--kaggle-mode", action="store_true")
    args = parser.parse_args(argv)
    report = build_smoke_preflight_report(args.config, kaggle_mode=args.kaggle_mode)
    write_json_checked(args.out, report, field_name="day8_2c_smoke_preflight_report")
    print(json.dumps({"status": report["status"], "decision": report["decision"], "out": args.out}, sort_keys=True))
    return 0 if report["status"] in {"PASS", "WARN"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
