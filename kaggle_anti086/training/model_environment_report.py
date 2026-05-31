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
from kaggle_anti086.training.model_loader import check_training_dependencies
from kaggle_anti086.training.training_config_schema import load_training_config


def build_model_environment_report(config: dict[str, Any], *, kaggle_mode: bool = False) -> dict[str, Any]:
    deps = check_training_dependencies(str(config.get("base_model_path", "")), kaggle_mode=kaggle_mode, config=config)
    warnings = []
    failures = list(deps.get("failures", []))
    if not kaggle_mode and (not deps.get("cuda_available") or not deps.get("tokenizer_loadable")):
        warnings.append("local_model_or_gpu_unavailable")
    status = "FAIL" if failures and kaggle_mode else ("WARN" if warnings or failures else "PASS")
    return {
        "status": status,
        "base_model_path": str(config.get("base_model_path", "")),
        "base_model_path_exists": bool(deps.get("base_model_path_exists", False)),
        "tokenizer_loadable": bool(deps.get("tokenizer_loadable", False)),
        "torch_available": bool(deps.get("torch_available", False)),
        "transformers_available": bool(deps.get("transformers_available", False)),
        "peft_available": bool(deps.get("peft_available", False)),
        "bitsandbytes_available": bool(deps.get("bitsandbytes_available", False)),
        "cuda_available": bool(deps.get("cuda_available", False)),
        "gpu_name": deps.get("device_name"),
        "gpu_count": _gpu_count(),
        "gpu_total_memory_gb": _gpu_memory_gb(),
        "bf16_supported": bool(deps.get("bf16_supported", False)),
        "load_in_4bit": bool(config.get("load_in_4bit", False)),
        "device_map": config.get("device_map", "auto"),
        "gradient_checkpointing": bool(config.get("gradient_checkpointing", False)),
        "failures": failures if kaggle_mode else [],
        "warnings": warnings + ([] if kaggle_mode else failures),
    }


def _gpu_count() -> int:
    try:
        import torch  # type: ignore

        return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:
        return 0


def _gpu_memory_gb() -> float:
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return 0.0
        props = torch.cuda.get_device_properties(0)
        return round(float(props.total_memory) / (1024**3), 3)
    except Exception:
        return 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="kaggle_anti086/training/configs/v2a_base_lora.yaml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--kaggle-mode", action="store_true")
    args = parser.parse_args(argv)
    report = build_model_environment_report(load_training_config(args.config), kaggle_mode=args.kaggle_mode)
    write_json_checked(args.out, report, field_name="model_environment_report")
    print(json.dumps({"status": report["status"], "out": args.out}, sort_keys=True))
    return 0 if report["status"] in {"PASS", "WARN"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
