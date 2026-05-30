from __future__ import annotations

from dataclasses import replace
import json

import pytest

from nemotron_engine.training_backend.adapter_validator import (
    AdapterValidationReport,
    AdapterValidationConfig,
    AdapterValidationError,
    compute_adapter_artifact_hashes,
    validate_adapter_output,
)


def write_adapter(tmp_path, rank: object = 8, *, model_count: int = 1):
    adapter = tmp_path / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(json.dumps({"r": rank}), encoding="utf-8")
    if model_count >= 1:
        (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    if model_count >= 2:
        (adapter / "adapter_model.bin").write_bytes(b"adapter-bin")
    return adapter


def test_valid_adapter_dir_with_rank_32_and_model_passes(tmp_path) -> None:
    adapter = write_adapter(tmp_path, rank=32)
    report = validate_adapter_output(adapter)
    assert report.passed
    assert report.rank == 32
    assert "adapter_config.json" in report.artifact_hashes
    assert report.adapter_hash == validate_adapter_output(adapter).adapter_hash


def test_rank_over_32_and_string_bool_rank_rejected(tmp_path) -> None:
    assert not validate_adapter_output(write_adapter(tmp_path / "a", rank=33)).passed
    assert any("rank" in error or "adapter_config" in error for error in validate_adapter_output(write_adapter(tmp_path / "b", rank="32")).errors)
    assert not validate_adapter_output(write_adapter(tmp_path / "c", rank=True)).passed


def test_missing_adapter_config_rejected(tmp_path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"x")
    report = validate_adapter_output(adapter)
    assert not report.passed
    assert any("adapter_config" in error for error in report.errors)


def test_invalid_adapter_config_rejected(tmp_path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{bad", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"x")
    assert not validate_adapter_output(adapter).passed


def test_missing_adapter_model_rejected(tmp_path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 8}), encoding="utf-8")
    report = validate_adapter_output(adapter)
    assert not report.passed
    assert any("model" in error for error in report.errors)


def test_more_than_one_adapter_model_rejected_unless_allowed(tmp_path) -> None:
    adapter = write_adapter(tmp_path, model_count=2)
    report = validate_adapter_output(adapter)
    assert not report.passed
    assert any("more than one" in error for error in report.errors)
    allowed = validate_adapter_output(AdapterValidationConfig(adapter, allow_multiple_adapter_model_files=True))
    assert allowed.passed


@pytest.mark.parametrize("name", ["pytorch_model.bin", "model.safetensors", "optimizer.pt", "scheduler.pt", "trainer_state.json"])
def test_forbidden_checkpoint_and_training_state_rejected(tmp_path, name: str) -> None:
    adapter = write_adapter(tmp_path)
    (adapter / name).write_bytes(b"x")
    assert not validate_adapter_output(adapter).passed


@pytest.mark.parametrize("name", ["kaggle.json", ".env", "token.txt", "credentials.json", "sft_rows.jsonl", "dpo_pairs.jsonl", "train.jsonl"])
def test_secret_and_dataset_files_rejected(tmp_path, name: str) -> None:
    adapter = write_adapter(tmp_path)
    (adapter / name).write_text("x", encoding="utf-8")
    assert not validate_adapter_output(adapter).passed


def test_symlink_reparse_rejected_if_supported(tmp_path) -> None:
    adapter = write_adapter(tmp_path)
    link = adapter / "linked_tokenizer.json"
    try:
        link.symlink_to(adapter / "adapter_config.json")
    except OSError:
        pytest.skip("symlink creation unavailable")
    report = validate_adapter_output(adapter)
    assert not report.passed
    assert any("symlink" in error for error in report.errors)


def test_forged_report_hash_rejected(tmp_path) -> None:
    report = validate_adapter_output(write_adapter(tmp_path))
    with pytest.raises(AdapterValidationError, match="report_hash"):
        replace(report, report_hash="forged")


def test_direct_forged_adapter_hash_rejected(tmp_path) -> None:
    report = validate_adapter_output(write_adapter(tmp_path))
    with pytest.raises(AdapterValidationError, match="adapter_hash"):
        replace(report, adapter_hash="f" * 64)


def test_passed_report_requires_artifacts_model_files_and_no_errors(tmp_path) -> None:
    report = validate_adapter_output(write_adapter(tmp_path))
    with pytest.raises(AdapterValidationError, match="artifact_hashes"):
        replace(report, artifact_hashes={})
    with pytest.raises(AdapterValidationError, match="adapter_model_files"):
        replace(report, adapter_model_files=())
    with pytest.raises(AdapterValidationError, match="errors"):
        replace(report, errors=("bad",))


def test_symlink_reparse_rejected_with_monkeypatch_fallback(tmp_path, monkeypatch) -> None:
    adapter = write_adapter(tmp_path)
    target = adapter / "adapter_model.safetensors"
    monkeypatch.setattr(type(target), "is_symlink", lambda self: self.name == "adapter_model.safetensors")
    report = validate_adapter_output(adapter)
    assert not report.passed
    assert any("symlink" in error for error in report.errors)


def test_compute_adapter_artifact_hashes_deterministic(tmp_path) -> None:
    adapter = write_adapter(tmp_path)
    assert compute_adapter_artifact_hashes(adapter) == compute_adapter_artifact_hashes(adapter)
