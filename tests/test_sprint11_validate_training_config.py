from pathlib import Path

from kaggle_anti086.training.validate_training_config import validate_config_dir


def test_validate_config_dir_reports_runnable_and_blocked():
    report = validate_config_dir(Path("kaggle_anti086/training/configs"))
    assert report["status"] == "PASS"
    assert report["runnable_count"] >= 1
    stages = {cfg["stage"]: cfg["status"] for cfg in report["configs"]}
    assert stages["v2a_base_lora"] == "PASS_RUNNABLE"
    assert stages["v2b_tinker_parent_lora"] == "PASS_BLOCKED_PARENT_MISSING"
    assert stages["v3_hard_family_repair"] == "PASS_NOT_RUNNABLE_YET"
