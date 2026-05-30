from __future__ import annotations

FORBIDDEN = (
    "test_label_use",
    "hardcoded_test_ids",
    "kaggle_api_calls",
    "auto_submit",
    "fake_adapters",
    "placeholder_eval",
    "full_prompt_loss",
    "generic_trainer_final_path",
)

REQUIRED = (
    "verified_rule_extraction",
    "private_like_validation",
    "token_mask_contract",
    "real_eval_metrics",
    "parent_child_eval",
    "package_gate",
)


def build_implementation_contract() -> dict:
    return {
        "forbidden": FORBIDDEN,
        "required": REQUIRED,
        "gpu_training_allowed_only_after": (
            "token_mask_contract_pass",
            "real_eval_available",
            "private_like_splits_present",
            "micro_stage_only_first",
        ),
    }
