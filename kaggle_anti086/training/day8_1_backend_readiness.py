from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.v2_corpus_io import read_json, write_json_checked
from kaggle_anti086.training.model_loader import check_training_dependencies
from kaggle_anti086.training.sft_dataset import build_sft_dataset_report
from kaggle_anti086.training.training_config_schema import STATUS_PASS_RUNNABLE, load_training_config, validate_training_config


def build_backend_readiness(config_path: str | Path = "kaggle_anti086/training/configs/v2a_base_lora.yaml", collator_path: str | Path = "artifacts/sprint11/day8_1_real_collator_audit_report.json") -> dict:
    config = load_training_config(config_path)
    failures = []
    warnings = []
    validation = validate_training_config(config, path=config_path)
    if validation.status != STATUS_PASS_RUNNABLE:
        failures.append("config_not_ready")
    collator = read_json(collator_path) if Path(collator_path).exists() else {"status": "MISSING"}
    if collator.get("status") != "PASS":
        failures.append("collator_audit_not_pass")
    day7 = read_json("artifacts/sprint11/day7_training_readiness_report.json") if Path("artifacts/sprint11/day7_training_readiness_report.json").exists() else {}
    if day7.get("decision") != "ALLOW_DAY8_V2_TRAINING":
        failures.append("day7_not_ready")
    for path, key in (
        ("artifacts/sprint11/day7_sampling_policy_consumption_report.json", "sampling_policy_not_pass"),
        ("artifacts/sprint11/train_v2_source_leakage_audit.json", "source_leakage_not_pass"),
        ("artifacts/sprint11/day7_adapter_constraint_report.json", "adapter_constraint_not_pass"),
    ):
        if not Path(path).exists() or read_json(path).get("status") != "PASS":
            failures.append(key)
    dataset_report = build_sft_dataset_report(config)
    if dataset_report["status"] != "PASS":
        failures.append("dataset_not_ready")
    deps = check_training_dependencies(str(config.get("base_model_path", "")))
    for field, failure in (
        ("torch_available", "torch_missing"),
        ("transformers_available", "transformers_missing"),
        ("peft_available", "peft_missing"),
        ("cuda_available", "cuda_missing"),
        ("tokenizer_loadable", "tokenizer_not_loadable"),
    ):
        if not deps.get(field):
            failures.append(failure)
    if not deps.get("bitsandbytes_available"):
        warnings.append("bitsandbytes_unavailable_qlora_disabled")
    return {
        "status": "PASS" if not failures else "FAIL",
        "training_backend_ready": not failures,
        "tokenizer_ready": bool(deps.get("tokenizer_loadable")),
        "model_ready": bool(deps.get("transformers_available") and deps.get("torch_available")),
        "collator_audit_pass": collator.get("status") == "PASS",
        "dataset_ready": dataset_report["status"] == "PASS",
        "config_ready": validation.status == STATUS_PASS_RUNNABLE,
        "gpu_ready": bool(deps.get("cuda_available")),
        "dependency_report": deps,
        "dataset_report": dataset_report,
        "failures": sorted(set(failures)),
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="kaggle_anti086/training/configs/v2a_base_lora.yaml")
    parser.add_argument("--collator-audit", default="artifacts/sprint11/day8_1_real_collator_audit_report.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = build_backend_readiness(args.config, args.collator_audit)
    write_json_checked(args.out, report, field_name="day8_1_backend_readiness_report")
    print(json.dumps({"status": report["status"], "out": args.out}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
