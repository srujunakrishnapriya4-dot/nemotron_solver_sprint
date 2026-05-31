from __future__ import annotations

from typing import Any
import argparse, json, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.v2_corpus_io import write_json_checked
from kaggle_anti086.training.lora_backend import prepare_model_for_v2a_training
from kaggle_anti086.training.model_loader import load_base_model
from kaggle_anti086.training.training_config_schema import load_training_config


def audit_trainable_parameters(model: Any) -> dict[str, Any]:
    total = trainable = 0
    trainable_names = []
    base_bad = False
    lora_good = False
    lm_head_trainable = False
    for name, param in getattr(model, "named_parameters", lambda: [])():
        n = int(getattr(param, "numel", lambda: 1)())
        total += n
        if getattr(param, "requires_grad", False):
            trainable += n
            if len(trainable_names) < 20:
                trainable_names.append(name)
            lname = name.lower()
            if "lora" in lname:
                lora_good = True
            elif "lm_head" in lname:
                lm_head_trainable = True
            else:
                base_bad = True
    ratio = trainable / total if total else 0.0
    failures = []
    if trainable == 0:
        failures.append("trainable_params_zero")
    if base_bad:
        failures.append("base_params_not_frozen")
    if not lora_good:
        failures.append("lora_params_not_trainable")
    if lm_head_trainable:
        failures.append("lm_head_trainable")
    if ratio > 0.10:
        failures.append("trainable_ratio_too_high")
    return {
        "status": "PASS" if not failures else "FAIL",
        "total_params": total,
        "trainable_params": trainable,
        "trainable_ratio": ratio,
        "base_params_frozen": not base_bad,
        "lora_params_trainable": lora_good,
        "lm_head_trainable": lm_head_trainable,
        "trainable_param_names_sample": trainable_names,
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="kaggle_anti086/training/configs/v2a_base_lora.yaml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--kaggle-mode", action="store_true")
    args = parser.parse_args(argv)
    if args.kaggle_mode:
        try:
            config = load_training_config(args.config)
            model = load_base_model(str(config["base_model_path"]), load_in_4bit=bool(config.get("load_in_4bit", False)), bf16=True)
            model = prepare_model_for_v2a_training(model, config)
            report = audit_trainable_parameters(model)
        except Exception as exc:
            report = {
                "status": "FAIL",
                "total_params": 0,
                "trainable_params": 0,
                "trainable_ratio": 0.0,
                "base_params_frozen": False,
                "lora_params_trainable": False,
                "lm_head_trainable": False,
                "trainable_param_names_sample": [],
                "failures": [f"trainable_audit_failed:{type(exc).__name__}"],
            }
    else:
        report = {
            "status": "FAIL",
            "total_params": 0,
            "trainable_params": 0,
            "trainable_ratio": 0.0,
            "base_params_frozen": False,
            "lora_params_trainable": False,
            "lm_head_trainable": False,
            "trainable_param_names_sample": [],
            "failures": ["model_unavailable_local"],
        }
    write_json_checked(args.out, report, field_name="day8_2_trainable_parameter_report")
    print(json.dumps({"status": report["status"], "out": args.out}, sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
