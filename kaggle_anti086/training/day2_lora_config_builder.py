from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


SAFE_TARGET_MODULE_SETS = {
    "qv": ["q_proj", "v_proj"],
    "qkvo": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "tinker_broad": ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj", "up_proj", "down_proj"],
}

ALLOWED_TARGET_MODULES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj",
    "out_proj",
    "up_proj",
    "down_proj",
}

FORBIDDEN_TARGET_MODULE_PATTERNS = {
    "all-linear",
    "all_linear",
    "*",
}


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def validate_target_modules(modules: List[str]) -> List[str]:
    problems: List[str] = []
    if not modules:
        problems.append("target_modules_empty")
        return problems

    normalized = [str(m).strip() for m in modules if str(m).strip()]
    if len(normalized) != len(modules):
        problems.append("target_modules_have_empty_values")

    for m in normalized:
        if m in FORBIDDEN_TARGET_MODULE_PATTERNS:
            problems.append(f"forbidden_target_module:{m}")
        if m not in ALLOWED_TARGET_MODULES:
            problems.append(f"unknown_or_unsafe_target_module:{m}")

    if len(set(normalized)) != len(normalized):
        problems.append("target_modules_duplicate_values")

    return problems


def validate_lora_config(config: Dict[str, Any]) -> List[str]:
    problems: List[str] = []

    rank = int(config.get("rank", 0) or 0)
    alpha = int(config.get("alpha", 0) or 0)
    dropout = float(config.get("dropout", -1.0))
    lr = float(config.get("learning_rate", 0.0) or 0.0)
    epochs = float(config.get("epochs", 0.0) or 0.0)
    max_seq_len = int(config.get("max_seq_len", 0) or 0)

    if rank <= 0:
        problems.append("rank_missing_or_nonpositive")
    if rank > 32:
        problems.append("rank_gt_32")
    if rank not in {8, 16, 32}:
        problems.append(f"rank_not_recommended:{rank}")

    if alpha <= 0:
        problems.append("alpha_missing_or_nonpositive")
    if alpha not in {16, 32, 64}:
        problems.append(f"alpha_not_recommended:{alpha}")

    if not (0.0 <= dropout <= 0.10):
        problems.append(f"dropout_out_of_safe_range:{dropout}")
    if not (1e-6 <= lr <= 8e-5):
        problems.append(f"learning_rate_out_of_safe_range:{lr}")
    if epochs <= 0 or epochs > 1.0:
        problems.append(f"epochs_must_be_0_to_1_initially:{epochs}")
    if max_seq_len not in {1024, 1536, 2048}:
        problems.append(f"max_seq_len_not_initial_safe_choice:{max_seq_len}")

    if config.get("bf16") is not True:
        problems.append("bf16_not_true")
    if config.get("gradient_checkpointing") is not True:
        problems.append("gradient_checkpointing_not_true")

    problems.extend(validate_target_modules(list(config.get("target_modules", []))))

    if config.get("uses_dpo") and not config.get("requires_sft_baseline_adapter"):
        problems.append("dpo_used_before_sft_baseline")

    return problems


def make_variant(
    name: str,
    dataset_variant: str,
    target_module_set: str,
    rank: int,
    alpha: int,
    lr: float,
    max_seq_len: int,
    train_rows_cap: int,
    priority: int,
    stage: str = "first_wave_sft",
    uses_dpo: bool = False,
    requires_sft_baseline_adapter: bool = False,
) -> Dict[str, Any]:
    target_modules = SAFE_TARGET_MODULE_SETS[target_module_set]
    cfg = {
        "schema_version": 1,
        "adapter_name": name,
        "stage": stage,
        "dataset_variant": dataset_variant,
        "target_module_set": target_module_set,
        "target_modules": target_modules,
        "rank": rank,
        "alpha": alpha,
        "dropout": 0.05,
        "learning_rate": lr,
        "epochs": 1.0,
        "max_seq_len": max_seq_len,
        "bf16": True,
        "gradient_checkpointing": True,
        "train_rows_cap": train_rows_cap,
        "eval_rows_cap": 5000,
        "uses_dpo": uses_dpo,
        "requires_sft_baseline_adapter": requires_sft_baseline_adapter,
        "priority": priority,
        "training_authorized": True,
        "package_authorized": False,
        "submission_authorized": False,
        "leaderboard_claim": False,
    }
    cfg["validation_problems"] = validate_lora_config(cfg)
    cfg["safe_to_train"] = len(cfg["validation_problems"]) == 0
    return cfg


def build_default_variant_matrix() -> Dict[str, Any]:
    variants = [
        make_variant(
            name="smoke_direct_qv_r16",
            dataset_variant="direct",
            target_module_set="qv",
            rank=16,
            alpha=32,
            lr=2e-5,
            max_seq_len=1024,
            train_rows_cap=2000,
            priority=1,
            stage="smoke_sft",
        ),
        make_variant(
            name="smoke_mixed_qv_r16",
            dataset_variant="mixed_curriculum",
            target_module_set="qv",
            rank=16,
            alpha=32,
            lr=2e-5,
            max_seq_len=1024,
            train_rows_cap=2000,
            priority=2,
            stage="smoke_sft",
        ),
        make_variant(
            name="A1_direct_qv_r32",
            dataset_variant="direct",
            target_module_set="qv",
            rank=32,
            alpha=64,
            lr=2e-5,
            max_seq_len=2048,
            train_rows_cap=50000,
            priority=10,
        ),
        make_variant(
            name="A2_short_trace_qv_r32",
            dataset_variant="short_trace",
            target_module_set="qv",
            rank=32,
            alpha=64,
            lr=2e-5,
            max_seq_len=2048,
            train_rows_cap=50000,
            priority=11,
        ),
        make_variant(
            name="A3_mixed_curriculum_qv_r32",
            dataset_variant="mixed_curriculum",
            target_module_set="qv",
            rank=32,
            alpha=64,
            lr=2e-5,
            max_seq_len=2048,
            train_rows_cap=100000,
            priority=12,
        ),
        make_variant(
            name="A4_format_heavy_qv_r32",
            dataset_variant="format_heavy",
            target_module_set="qv",
            rank=32,
            alpha=64,
            lr=1.5e-5,
            max_seq_len=2048,
            train_rows_cap=50000,
            priority=13,
        ),
        make_variant(
            name="A5_composed_heavy_qv_r32",
            dataset_variant="composed_heavy",
            target_module_set="qv",
            rank=32,
            alpha=64,
            lr=2e-5,
            max_seq_len=2048,
            train_rows_cap=50000,
            priority=14,
        ),
        make_variant(
            name="B1_mixed_qkvo_r32_low_lr",
            dataset_variant="mixed_curriculum",
            target_module_set="qkvo",
            rank=32,
            alpha=64,
            lr=1e-5,
            max_seq_len=2048,
            train_rows_cap=100000,
            priority=30,
            stage="second_wave_if_first_wave_improves",
        ),
        make_variant(
            name="D1_dpo_lite_from_best_sft",
            dataset_variant="dpo_train",
            target_module_set="qv",
            rank=32,
            alpha=64,
            lr=5e-6,
            max_seq_len=2048,
            train_rows_cap=5000,
            priority=40,
            stage="dpo_lite_only_after_best_sft",
            uses_dpo=True,
            requires_sft_baseline_adapter=True,
        ),
    ]

    unsafe = [v for v in variants if not v["safe_to_train"]]

    return {
        "schema_version": 1,
        "created_by": "DAY2_LORA_CONFIG_BUILDER",
        "status": "PASS" if not unsafe else "FAIL",
        "variant_count": len(variants),
        "safe_variant_count": len([v for v in variants if v["safe_to_train"]]),
        "unsafe_variant_count": len(unsafe),
        "safe_target_module_sets": SAFE_TARGET_MODULE_SETS,
        "allowed_target_modules": sorted(ALLOWED_TARGET_MODULES),
        "variants": variants,
        "recommended_first_training_order": [
            "smoke_direct_qv_r16",
            "smoke_mixed_qv_r16",
            "A1_direct_qv_r32",
            "A3_mixed_curriculum_qv_r32",
            "A2_short_trace_qv_r32",
            "A4_format_heavy_qv_r32",
            "A5_composed_heavy_qv_r32",
        ],
        "dpo_policy": "DPO-lite is blocked until at least one SFT adapter beats base/v2a_50.",
        "training_authorized": True,
        "package_authorized": False,
        "submission_authorized": False,
        "leaderboard_claim": False,
        "no_0_93_evidence": True,
        "no_0_95_evidence": True,
        "blocked_reasons": [] if not unsafe else ["unsafe_lora_variant_present"],
    }


def build_training_config_manifest(variant_matrix: Dict[str, Any]) -> Dict[str, Any]:
    blocked = []
    if variant_matrix.get("status") != "PASS":
        blocked.append("variant_matrix_not_pass")

    for variant in variant_matrix.get("variants", []):
        if variant.get("rank", 999) > 32:
            blocked.append(f"rank_gt_32:{variant.get('adapter_name')}")
        if variant.get("uses_dpo") and not variant.get("requires_sft_baseline_adapter"):
            blocked.append(f"dpo_before_sft:{variant.get('adapter_name')}")
        if variant.get("package_authorized") is True or variant.get("submission_authorized") is True:
            blocked.append(f"unsafe_package_or_submission_flag:{variant.get('adapter_name')}")

    return {
        "schema_version": 1,
        "created_by": "DAY2_TRAINING_CONFIG_MANIFEST",
        "status": "PASS" if not blocked else "FAIL",
        "base_model_family": "NVIDIA-Nemotron-3-Nano-30B",
        "adapter_rank_limit": 32,
        "initial_training_policy": {
            "start_with_smoke": True,
            "start_with_sft_only": True,
            "dpo_before_sft_baseline": False,
            "all_linear_forbidden_initially": True,
            "max_initial_epochs": 1.0,
            "max_initial_seq_len": 2048,
        },
        "variant_matrix_status": variant_matrix.get("status"),
        "recommended_first_training_order": variant_matrix.get("recommended_first_training_order", []),
        "training_authorized": not blocked,
        "package_authorized": False,
        "submission_authorized": False,
        "leaderboard_claim": False,
        "no_0_93_evidence": True,
        "no_0_95_evidence": True,
        "blocked_reasons": blocked,
    }


def write_default_configs(out_dir: Path) -> Dict[str, Any]:
    variant_matrix = build_default_variant_matrix()
    config_manifest = build_training_config_manifest(variant_matrix)

    write_json(out_dir / "day2_lora_variant_matrix.json", variant_matrix)
    write_json(out_dir / "day2_training_config_manifest.json", config_manifest)

    return {
        "variant_matrix": variant_matrix,
        "training_config_manifest": config_manifest,
    }
