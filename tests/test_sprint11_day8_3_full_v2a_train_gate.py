import json
from pathlib import Path

from kaggle_anti086.training.day8_3_full_v2a_train import validate_full_train_gate
from kaggle_anti086.training.training_config_schema import load_training_config


def _config():
    return load_training_config("kaggle_anti086/training/configs/v2a_base_lora.yaml")


def _smoke(path: Path, decision="ALLOW_DAY8_3_FULL_V2A_TRAINING", allowed=True, guard="PASS"):
    path.write_text(
        json.dumps({"decision": decision, "full_training_allowed": allowed, "gates": {"package_submission_guard": {"status": guard}}}),
        encoding="utf-8",
    )


def test_missing_smoke_report_fails(monkeypatch):
    monkeypatch.setattr("kaggle_anti086.training.day8_3_full_v2a_train.build_package_submission_guard", lambda: {"status": "PASS"})
    _, failures = validate_full_train_gate(_config(), smoke_report_path=None)
    assert "smoke_report_missing" in failures


def test_smoke_report_not_allow_fails(monkeypatch, tmp_path):
    monkeypatch.setattr("kaggle_anti086.training.day8_3_full_v2a_train.build_package_submission_guard", lambda: {"status": "PASS"})
    smoke = tmp_path / "smoke.json"
    _smoke(smoke, decision="NEEDS_KAGGLE_SMOKE_TRAIN", allowed=False)
    _, failures = validate_full_train_gate(_config(), smoke_report_path=smoke)
    assert "smoke_report_decision_not_allow" in failures


def test_allowing_smoke_report_passes_gate(monkeypatch, tmp_path):
    monkeypatch.setattr("kaggle_anti086.training.day8_3_full_v2a_train.build_package_submission_guard", lambda: {"status": "PASS"})
    smoke = tmp_path / "smoke.json"
    _smoke(smoke)
    readiness = tmp_path / "readiness.json"
    readiness.write_text(json.dumps({"decision": "ALLOW_DAY8_3_FULL_V2A_TRAINING", "full_training_allowed": True}), encoding="utf-8")
    _, failures = validate_full_train_gate(_config(), smoke_report_path=smoke, smoke_readiness_path=readiness)
    assert failures == []


def test_non_v2a_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr("kaggle_anti086.training.day8_3_full_v2a_train.build_package_submission_guard", lambda: {"status": "PASS"})
    smoke = tmp_path / "smoke.json"
    _smoke(smoke)
    config = dict(_config())
    config["stage"] = "v2b_tinker_parent_lora"
    _, failures = validate_full_train_gate(config, smoke_report_path=smoke)
    assert "non_v2a_stage_rejected" in failures


def test_existing_configured_adapter_dir_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr("kaggle_anti086.training.day8_3_full_v2a_train.build_package_submission_guard", lambda: {"status": "PASS"})
    smoke = tmp_path / "smoke.json"
    _smoke(smoke)
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    config = dict(_config())
    config["output_adapter_dir"] = str(adapter_dir)
    _, failures = validate_full_train_gate(config, smoke_report_path=smoke)
    assert "configured_output_adapter_dir_already_contains_adapter" in failures


def test_day8_2c_readiness_block_fails(monkeypatch, tmp_path):
    monkeypatch.setattr("kaggle_anti086.training.day8_3_full_v2a_train.build_package_submission_guard", lambda: {"status": "PASS"})
    smoke = tmp_path / "smoke.json"
    _smoke(smoke)
    readiness = tmp_path / "readiness.json"
    readiness.write_text(json.dumps({"decision": "BLOCK_FULL_TRAINING_MEMORY_BACKEND_BUG", "full_training_allowed": False}), encoding="utf-8")
    _, failures = validate_full_train_gate(_config(), smoke_report_path=smoke, smoke_readiness_path=readiness)
    assert "day8_2c_smoke_readiness_not_allow" in failures


def test_day8_2c_readiness_allow_passes(monkeypatch, tmp_path):
    monkeypatch.setattr("kaggle_anti086.training.day8_3_full_v2a_train.build_package_submission_guard", lambda: {"status": "PASS"})
    smoke = tmp_path / "smoke.json"
    _smoke(smoke)
    readiness = tmp_path / "readiness.json"
    readiness.write_text(json.dumps({"decision": "ALLOW_DAY8_3_FULL_V2A_TRAINING", "full_training_allowed": True}), encoding="utf-8")
    _, failures = validate_full_train_gate(_config(), smoke_report_path=smoke, smoke_readiness_path=readiness)
    assert failures == []
