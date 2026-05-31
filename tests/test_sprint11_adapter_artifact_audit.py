import json

from kaggle_anti086.training.adapter_artifact_audit import audit_adapter_artifacts


def test_valid_adapter_passes(tmp_path):
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 32, "target_modules": ["q_proj", "v_proj", "o_proj"]}), encoding="utf-8")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"x")
    report = audit_adapter_artifacts(tmp_path)
    assert report["status"] == "PASS"
    assert report["packageable"] is False


def test_dirty_adapter_fails(tmp_path):
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 64, "target_modules": ["lm_head"]}), encoding="utf-8")
    (tmp_path / "optimizer.pt").write_bytes(b"x")
    report = audit_adapter_artifacts(tmp_path)
    assert report["status"] == "FAIL"
    assert "adapter_model_missing" in report["failures"]
    assert "forbidden_files_present" in report["failures"]
