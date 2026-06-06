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
from kaggle_anti086.training.lora_backend import inspect_target_modules, validate_target_modules_for_peft
from kaggle_anti086.training.model_loader import check_training_dependencies, load_base_model, load_tokenizer
from kaggle_anti086.training.training_config_schema import _target_modules, load_training_config


DEFAULT_CANDIDATES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def build_nemotron_lora_target_diagnostics(
    config_path: str | Path,
    *,
    load_model: bool = False,
    inspect_peft: bool = False,
    kaggle_mode: bool = False,
    candidate_targets: list[str] | None = None,
) -> dict[str, Any]:
    config = load_training_config(config_path)
    candidates = candidate_targets or DEFAULT_CANDIDATES
    deps = check_training_dependencies(str(config.get("base_model_path", "")), kaggle_mode=kaggle_mode, config=config)
    model = None
    tokenizer_loaded = False
    failures: list[str] = []
    warnings: list[str] = []
    inspected: list[dict[str, Any]] = []
    peft_validation = {
        "status": "SKIPPED",
        "target_modules": _target_modules(config),
        "failures": [],
        "warnings": [],
    }
    peft_lora_shapes: list[dict[str, Any]] = []
    if not load_model:
        warnings.append("model_not_loaded_dry_inspection")
        status = "WARN"
    else:
        try:
            tokenizer = load_tokenizer(str(config.get("base_model_path", "")))
            tokenizer_loaded = tokenizer is not None
            model = load_base_model(
                str(config.get("base_model_path", "")),
                load_in_4bit=bool(config.get("load_in_4bit", False)),
                bf16=True,
            )
            inspected = inspect_target_modules(model, candidates)
            peft_validation = validate_target_modules_for_peft(model, _target_modules(config))
            if inspect_peft and peft_validation.get("status") == "PASS":
                peft_lora_shapes = _inspect_peft_lora_shapes(model, config)
        except Exception as exc:
            failures.append(f"diagnostics_model_inspection_failed:{type(exc).__name__}:{exc}")
        finally:
            model = None
            try:
                import gc
                gc.collect()
                import torch  # type: ignore
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        status = "FAIL" if failures or peft_validation.get("status") == "FAIL" else "PASS"
    return {
        "status": status,
        "config_path": str(config_path),
        "stage": config.get("stage"),
        "candidate_targets": candidates,
        "configured_targets": _target_modules(config),
        "load_model": load_model,
        "inspect_peft": inspect_peft,
        "kaggle_mode": kaggle_mode,
        "dependency_report": deps,
        "tokenizer_loaded": tokenizer_loaded,
        "target_modules": inspected,
        "peft_validation": peft_validation,
        "peft_lora_shapes": peft_lora_shapes,
        "peft_linear4bit_shape_incompatible": "peft_linear4bit_shape_incompatible" in peft_validation.get("failures", []),
        "packaging_allowed": False,
        "submission_allowed": False,
        "no_training_performed": True,
        "no_leaderboard_evidence": True,
        "no_0_95_evidence": True,
        "failures": failures + list(peft_validation.get("failures", []) if peft_validation.get("status") == "FAIL" else []),
        "warnings": warnings + list(deps.get("failures", []) if not kaggle_mode else []),
    }


def _inspect_peft_lora_shapes(model: Any, config: dict[str, Any]) -> list[dict[str, Any]]:
    from kaggle_anti086.training.lora_backend import prepare_model_for_v2a_training

    wrapped = prepare_model_for_v2a_training(model, config)
    shapes = []
    for name, module in wrapped.named_modules():
        lora_a = getattr(module, "lora_A", None)
        lora_b = getattr(module, "lora_B", None)
        if lora_a is None and lora_b is None:
            continue
        shapes.append(
            {
                "name": name,
                "lora_A": _module_shapes(lora_a),
                "lora_B": _module_shapes(lora_b),
            }
        )
    return shapes[:200]


def _module_shapes(value: Any) -> dict[str, list[int] | None]:
    output: dict[str, list[int] | None] = {}
    if hasattr(value, "items"):
        iterator = value.items()
    else:
        iterator = [("default", value)]
    for key, module in iterator:
        weight = getattr(module, "weight", None)
        shape = getattr(weight, "shape", None)
        output[str(key)] = None if shape is None else [int(item) for item in shape]
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose Nemotron LoRA target PEFT/Linear4bit compatibility.")
    parser.add_argument("--config", default="kaggle_anti086/training/configs/v4_solver_teacher_lora_small.yaml")
    parser.add_argument("--out", default="artifacts/sprint11/day10_nemotron_lora_target_diagnostics.json")
    parser.add_argument("--load-model", action="store_true")
    parser.add_argument("--inspect-peft", action="store_true")
    parser.add_argument("--kaggle-mode", action="store_true")
    parser.add_argument("--candidate-targets", default="q_proj,k_proj,v_proj,o_proj")
    args = parser.parse_args(argv)
    candidates = [item.strip() for item in args.candidate_targets.split(",") if item.strip()]
    report = build_nemotron_lora_target_diagnostics(
        args.config,
        load_model=args.load_model,
        inspect_peft=args.inspect_peft,
        kaggle_mode=args.kaggle_mode,
        candidate_targets=candidates,
    )
    write_json_checked(args.out, report, field_name="day10_nemotron_lora_target_diagnostics")
    print(json.dumps({"status": report["status"], "out": args.out}, sort_keys=True))
    return 0 if report["status"] in {"PASS", "WARN"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
