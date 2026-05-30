from __future__ import annotations


def select_training_recipe(backend: dict) -> dict:
    if backend.get("vex_style_available") and backend.get("vllm_available") and backend.get("token_corpus_available"):
        return {
            "recommended_mode": "vex_style_available",
            "reason": "Use preferred VEX/Unsloth/cut-cross-entropy/vLLM path with anti086 curriculum.",
            "required_inputs": ("anti086_kaggle_input.zip", "base_model", "vLLM", "LoRA backend"),
            "forbidden_actions": ("generic_trainer", "full_prompt_loss", "blind_grpo"),
            "expected_score_risk": "best available path but still unproven beyond 0.86",
        }
    if backend.get("tinker_cloud_available"):
        return {
            "recommended_mode": "tinker_cloud_available",
            "reason": "Export corpus/config for Tinker-style training; do not fake Kaggle heavy backend.",
            "required_inputs": ("anti086 corpus", "tinker cloud"),
            "forbidden_actions": ("local_training", "test_label_use"),
            "expected_score_risk": "medium; depends on private-like validation",
        }
    if backend.get("kaggle_gpu_available"):
        return {
            "recommended_mode": "kaggle_micro_only",
            "reason": "Only sanity checks are viable without VEX token backend and vLLM.",
            "required_inputs": ("micro corpus",),
            "forbidden_actions": ("main_training", "submission_packaging"),
            "expected_score_risk": "not a serious score path",
        }
    return {
        "recommended_mode": "no_viable_training",
        "reason": "No backend can safely train; do not waste GPU.",
        "required_inputs": (),
        "forbidden_actions": ("training", "packaging_custom_child"),
        "expected_score_risk": "baseline only",
    }
