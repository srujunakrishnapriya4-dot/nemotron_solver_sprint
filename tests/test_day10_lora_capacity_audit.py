from __future__ import annotations

from kaggle_anti086.training.day10_lora_capacity_audit import audit_config, audit_configs


def test_capacity_audit_small_viable_and_wide_blocked_without_model_confirmation():
    report = audit_configs([
        "kaggle_anti086/training/configs/v4_solver_teacher_lora_small.yaml",
        "kaggle_anti086/training/configs/v4_solver_teacher_lora_wide.yaml",
    ])

    assert report["status"] == "PASS"
    assert report["small_config_viable"] is True
    assert report["wide_config_status"] == "BLOCKED_TARGETS_UNCONFIRMED"
    assert "target_modules_unverified_without_real_model" in report["warnings"]


def test_capacity_audit_rejects_lm_head(tmp_path):
    config = tmp_path / "bad.yaml"
    config.write_text(
        "\n".join(
            [
                "stage: v4_solver_teacher_lora_small",
                "rank: 32",
                "target_modules: q_proj,v_proj,o_proj,lm_head",
                "adapter_size_limit_mb: 1500",
                "assistant_only_loss: true",
                "full_prompt_loss: false",
                "train_on_user: false",
                "abstain_weight: 0.0",
            ]
        ),
        encoding="utf-8",
    )

    report = audit_config(config, available_modules={"q_proj", "v_proj", "o_proj", "lm_head"})

    assert report["status"] == "FAIL"
    assert "forbidden_target_module" in report["failures"]
