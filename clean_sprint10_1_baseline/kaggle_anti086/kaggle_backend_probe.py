from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import platform
from typing import Any
import zipfile

from kaggle_runtime_patches import apply_runtime_patches


DEFAULT_BASE_MODEL_PATHS = (
    "/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1",
    "/kaggle/input/nemotron-3-nano-30b-a3b-bf16",
)


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def detect_cuda() -> dict[str, Any]:
    if not has_module("torch"):
        return {"torch_import": False, "cuda_available": False, "gpu_name": None}
    import torch

    available = bool(torch.cuda.is_available())
    return {
        "torch_import": True,
        "cuda_available": available,
        "gpu_name": torch.cuda.get_device_name(0) if available else None,
    }


def detect_lora_request() -> bool:
    if not has_module("vllm"):
        return False
    try:
        from vllm.lora.request import LoRARequest  # noqa: F401
    except Exception:
        return False
    return True


def detect_anti086_input_root(base: str | Path = "/kaggle/input") -> str | None:
    base_path = Path(base)
    if not base_path.exists():
        return None
    required_sets = (
        {"curriculum_manifest.json"},
        {"win_corpus_manifest.json"},
        {"solver_coverage_report.json", "win_micro.jsonl"},
    )
    for path in sorted(base_path.rglob("*")):
        if path.is_dir() and any(all((path / name).exists() for name in required) for required in required_sets):
            return str(path)
        if path.name in {"anti086_kaggle_input.zip", "win_system_kaggle_input.zip"}:
            return str(path)
    return None


def detect_parent_adapter(base: str | Path = "/kaggle/input") -> str | None:
    base_path = Path(base)
    if not base_path.exists():
        return None
    for config in sorted(base_path.rglob("adapter_config.json")):
        if (config.parent / "adapter_model.safetensors").exists():
            return str(config.parent)
    return None


def probe_backend(
    *,
    base_model_path: str | None = None,
    input_base: str | Path = "/kaggle/input",
    output_path: str | Path = "/kaggle/working/anti086_backend_probe.json",
) -> dict[str, Any]:
    runtime_patch = apply_runtime_patches()
    cuda = detect_cuda()
    vllm_available = has_module("vllm")
    cutlass_root = Path("/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/nvidia_cutlass_dsl/python_packages")
    ptxas = Path("/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/triton/backends/nvidia/bin/ptxas-blackwell")
    selected_base = base_model_path or next((p for p in DEFAULT_BASE_MODEL_PATHS if Path(p).exists()), DEFAULT_BASE_MODEL_PATHS[0])
    anti086_root = detect_anti086_input_root(input_base)
    report = {
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        **cuda,
        "transformers_import": has_module("transformers"),
        "peft_import": has_module("peft"),
        "vllm_import": vllm_available,
        "lora_request_import": detect_lora_request(),
        "cutlass_path_exists": cutlass_root.exists(),
        "ptxas_blackwell_path_exists": ptxas.exists(),
        "base_model_path": selected_base,
        "base_model_path_exists": Path(selected_base).exists(),
        "anti086_input_root": anti086_root,
        "win_input_root": anti086_root,
        "anti086_input_root_exists": anti086_root is not None,
        "parent_adapter_path": detect_parent_adapter(input_base),
        "runtime_patch": runtime_patch,
        "micro_stack_test_label": "INFRASTRUCTURE STACK TEST ONLY - NOT A 0.95 CANDIDATE - NOT MAIN TRAINING - NOT SUBMISSION READY",
    }
    if report["cuda_available"] and report["transformers_import"] and report["peft_import"] and report["base_model_path_exists"] and report["anti086_input_root_exists"]:
        report["recommended_mode"] = "BACKEND_OK_FOR_MICRO" if report["vllm_import"] and report["lora_request_import"] else "BACKEND_NO_VLLM_BUT_TRAIN_POSSIBLE"
    else:
        report["recommended_mode"] = "BACKEND_NOT_SAFE"
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, sort_keys=True, indent=2), encoding="utf-8")
    return report


def main() -> None:
    report = probe_backend()
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
