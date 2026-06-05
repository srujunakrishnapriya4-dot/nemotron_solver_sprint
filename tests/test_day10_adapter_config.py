from __future__ import annotations

from pathlib import Path

from kaggle_anti086.training.training_config_schema import load_training_config


def test_day10_small_and_wide_configs_are_present_and_safe():
    small = load_training_config("kaggle_anti086/training/configs/v4_solver_teacher_lora_small.yaml")
    wide = load_training_config("kaggle_anti086/training/configs/v4_solver_teacher_lora_wide.yaml")

    assert small["stage"] == "v4_solver_teacher_lora_small"
    assert wide["stage"] == "v4_solver_teacher_lora_wide"
    assert set(small["target_modules"]) == {"q_proj", "v_proj", "o_proj"}
    assert {"q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"} == set(wide["target_modules"])
    for config in (small, wide):
        assert config["rank"] <= 32
        assert config["assistant_only_loss"] is True
        assert config["full_prompt_loss"] is False
        assert config["train_on_user"] is False
        assert "lm_head" not in config["target_modules"]
        assert Path(config["train_teacher_path"]).as_posix().endswith("day10_solver_teacher_direct.jsonl")
