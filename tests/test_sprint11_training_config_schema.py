from kaggle_anti086.training.training_config_schema import STATUS_FAIL, STATUS_PARENT_MISSING, STATUS_PASS_RUNNABLE, validate_training_config


def _cfg(**overrides):
    cfg = {
        "stage": "v2a_base_lora",
        "base_model_path": "/kaggle/input/nemotron-model",
        "parent_adapter_path": "none",
        "train_direct_path": "artifacts/sprint11/train_v2_verified_direct_answer.jsonl",
        "train_solver_corrected_path": "artifacts/sprint11/train_v2_solver_corrected.jsonl",
        "train_abstain_safety_path": "artifacts/sprint11/train_v2_abstain_safety.jsonl",
        "train_hard_negative_path": "artifacts/sprint11/train_v2_hard_negative.jsonl",
        "output_adapter_dir": "/kaggle/working/anti086_adapters/v2a",
        "rank": 32,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj", "o_proj"],
        "learning_rate": 1e-7,
        "max_seq_len": 1024,
        "micro_batch_size": 1,
        "gradient_accumulation": 8,
        "num_steps": 500,
        "assistant_only_loss": True,
        "full_prompt_loss": False,
        "train_on_user": False,
        "direct_answer_weight": 1.0,
        "solver_corrected_weight": 0.5,
        "abstain_safety_weight": 0.0,
        "hard_negative_weight": 0.0,
        "adapter_size_limit_mb": 1500,
        "not_submission_ready": True,
    }
    cfg.update(overrides)
    return cfg


def test_valid_v2a_passes():
    assert validate_training_config(_cfg()).status == STATUS_PASS_RUNNABLE


def test_rank_full_prompt_assistant_user_failures():
    assert validate_training_config(_cfg(rank=33)).status == STATUS_FAIL
    assert validate_training_config(_cfg(full_prompt_loss=True)).status == STATUS_FAIL
    assert validate_training_config(_cfg(assistant_only_loss=False)).status == STATUS_FAIL
    assert validate_training_config(_cfg(train_on_user=True)).status == STATUS_FAIL


def test_unsafe_paths_and_lm_head_fail():
    assert validate_training_config(_cfg(output_adapter_dir="/kaggle/input/bad")).status == STATUS_FAIL
    assert validate_training_config(_cfg(output_adapter_dir="/tmp/bad")).status == STATUS_FAIL
    assert validate_training_config(_cfg(target_modules=["q_proj", "v_proj", "o_proj", "lm_head"])).status == STATUS_FAIL


def test_missing_parent_blocks_not_fail():
    result = validate_training_config(_cfg(stage="v2b_tinker_parent_lora", parent_adapter_path="REQUIRED_TINKER_PARENT_ADAPTER_PATH", learning_rate=5e-8))
    assert result.status == STATUS_PARENT_MISSING
    assert result.runnable is False
