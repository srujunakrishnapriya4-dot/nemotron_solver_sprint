from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nemotron_engine.core.schemas import stable_hash


BACKENDS = {
    "kaggle_rtx_pro_6000_blackwell": {
        "can_build_corpus": True,
        "can_tokenize": True,
        "can_run_vLLM": True,
        "can_train_Nemotron": True,
        "can_use_Unsloth": True,
        "can_use_cut_cross_entropy": True,
        "can_use_Triton_CUTLASS": True,
        "can_package_adapter": True,
        "risk_level": "medium",
        "expected_runtime": "scarce_gpu; use micro->stage gates",
        "recommended_use": "final compatibility training and vLLM eval only after micro gates pass",
    },
    "amd_developer_cloud": {
        "can_build_corpus": True,
        "can_tokenize": True,
        "can_run_vLLM": False,
        "can_train_Nemotron": False,
        "can_use_Unsloth": False,
        "can_use_cut_cross_entropy": False,
        "can_use_Triton_CUTLASS": False,
        "can_package_adapter": True,
        "risk_level": "high_for_training_low_for_data",
        "expected_runtime": "$200 credits best spent on CPU-heavy generation/validation unless ROCm stack is proven",
        "recommended_use": "synthetic generation, corpus construction, tokenizer/mask validation, tests; not final training by default",
    },
    "tinker_style_backend": {
        "can_build_corpus": True,
        "can_tokenize": True,
        "can_run_vLLM": True,
        "can_train_Nemotron": True,
        "can_use_Unsloth": None,
        "can_use_cut_cross_entropy": True,
        "can_use_Triton_CUTLASS": True,
        "can_package_adapter": True,
        "risk_level": "low_if_available",
        "expected_runtime": "preferred heavy SFT backend if user has access",
        "recommended_use": "heavy anti086 SFT if available; export corpus/config only otherwise",
    },
    "local_cpu_only": {
        "can_build_corpus": True,
        "can_tokenize": False,
        "can_run_vLLM": False,
        "can_train_Nemotron": False,
        "can_use_Unsloth": False,
        "can_use_cut_cross_entropy": False,
        "can_use_Triton_CUTLASS": False,
        "can_package_adapter": True,
        "risk_level": "safe_but_no_training",
        "expected_runtime": "fast for reports, impossible for 30B training",
        "recommended_use": "planning, audits, deterministic artifact generation, unit tests",
    },
}


def build_backend_capability_matrix(output_path: str | Path = "artifacts/anti086/backend_capability_matrix.json") -> dict[str, Any]:
    payload = {
        "backends": BACKENDS,
        "verdict": {
            "preferred_final_training": "kaggle_rtx_pro_6000_blackwell_or_tinker_style_backend",
            "amd_cloud_default": "data_generation_and_validation_only",
            "do_not_assume_amd_cuda_stack": True,
            "local_cpu_default": "no_training",
        },
    }
    payload["matrix_hash"] = stable_hash(payload)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    return payload
