from __future__ import annotations

from kaggle_anti086.training.day10_adapter_package_guard import rehearse_adapter_package


def test_package_rehearsal_accepts_root_level_adapter_only(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"safe")

    report = rehearse_adapter_package(adapter, repo_root=tmp_path)

    assert report["status"] == "PASS"
    assert report["package_created"] is False
    assert report["packaging_allowed"] is False
    assert report["submission_allowed"] is False


def test_package_rehearsal_rejects_nested_and_submission_zip(tmp_path):
    adapter = tmp_path / "adapter"
    nested = adapter / "nested"
    nested.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (nested / "adapter_model.safetensors").write_bytes(b"bad")
    (tmp_path / "submission.zip").write_bytes(b"nope")

    report = rehearse_adapter_package(adapter, repo_root=tmp_path)

    assert report["status"] == "FAIL"
    assert "adapter_model_missing_at_package_root" in report["failures"]
    assert "nested_adapter_model_found" in report["failures"]
    assert "forbidden_package_artifact_inside_repo" in report["failures"]
