from kaggle_anti086.training.adapter_constraint_audit import build_adapter_constraint_audit
from kaggle_anti086.training.training_config_schema import validate_training_config
from tests.test_sprint11_training_config_schema import _cfg


def test_adapter_constraints_real_configs():
    report = build_adapter_constraint_audit("kaggle_anti086/training/configs")
    assert report["status"] == "PASS"
    assert "v2a_base_lora" in report["runnable_configs"]
    assert "v3_hard_family_repair" in report["blocked_configs"]
    assert report["v3_blocked"] is True


def test_rank_target_parent_and_path_checks():
    assert validate_training_config(_cfg(rank=64)).status == "FAIL"
    assert validate_training_config(_cfg(target_modules=["q_proj", "v_proj", "o_proj", "lm_head"])).status == "FAIL"
    assert validate_training_config(_cfg(output_adapter_dir="/kaggle/input/x")).status == "FAIL"
    assert validate_training_config(_cfg(stage="v2c_best_parent_lora", parent_adapter_path="REQUIRED_BEST_PARENT_ADAPTER_PATH", learning_rate=5e-8)).status == "PASS_BLOCKED_PARENT_MISSING"
