from __future__ import annotations

from collections import Counter
import argparse, json, sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.v2_corpus_io import write_json_checked
from kaggle_anti086.training.model_loader import load_base_model
from kaggle_anti086.training.training_config_schema import _target_modules, load_training_config


def inspect_linear_modules(model: Any) -> dict[str, Any]:
    names = []
    for name, module in getattr(model, "named_modules", lambda: [])():
        cls = module.__class__.__name__.lower()
        if "linear" in cls or cls in {"linear4bit", "linear8bitlt"}:
            names.append(name)
    return {"linear_module_names": names, "candidate_linear_module_names_sample": names[:50]}


def verify_lora_targets(model: Any, targets: list[str]) -> dict[str, Any]:
    failures = []
    if "lm_head" in targets:
        failures.append("lm_head_target_forbidden")
    info = inspect_linear_modules(model)
    counts = Counter()
    for name in info["linear_module_names"]:
        leaf = name.split(".")[-1]
        if leaf in targets:
            counts[leaf] += 1
    for target in targets:
        if counts[target] == 0:
            failures.append(f"target_missing:{target}")
    if sum(counts.values()) == 0:
        failures.append("total_matched_modules_zero")
    return {
        "status": "PASS" if not failures else "FAIL",
        "target_modules": targets,
        "matched_target_counts": {target: counts[target] for target in targets},
        "total_matched_modules": sum(counts.values()),
        "candidate_linear_module_names_sample": info["candidate_linear_module_names_sample"],
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="kaggle_anti086/training/configs/v2a_base_lora.yaml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--kaggle-mode", action="store_true")
    args = parser.parse_args(argv)
    config = load_training_config(args.config)
    if args.kaggle_mode:
        try:
            model = load_base_model(str(config["base_model_path"]), load_in_4bit=bool(config.get("load_in_4bit", False)), bf16=True)
            report = verify_lora_targets(model, _target_modules(config))
        except Exception as exc:
            report = {
                "status": "FAIL",
                "target_modules": _target_modules(config),
                "matched_target_counts": {target: 0 for target in _target_modules(config)},
                "total_matched_modules": 0,
                "candidate_linear_module_names_sample": [],
                "failures": [f"model_inspection_failed:{type(exc).__name__}"],
            }
    else:
        report = {
            "status": "FAIL",
            "target_modules": _target_modules(config),
            "matched_target_counts": {target: 0 for target in _target_modules(config)},
            "total_matched_modules": 0,
            "candidate_linear_module_names_sample": [],
            "failures": ["model_unavailable_local"],
        }
    write_json_checked(args.out, report, field_name="day8_2_lora_target_report")
    print(json.dumps({"status": report["status"], "out": args.out}, sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
