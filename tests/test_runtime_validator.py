from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.packaging.runtime_validator import (
    RuntimeValidationConfig,
    RuntimeValidationError,
    RuntimeValidationReport,
    validate_offline_runtime,
)
from nemotron_engine.runtime.serving_config import ServingConfig


@contextmanager
def temp_root(name: str):
    root = Path(__file__).resolve().parent / ".tmp_pass9_runtime" / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def serving(**updates: object) -> ServingConfig:
    data = {
        "temperature": 0.0,
        "top_p": 1.0,
        "num_samples": 1,
        "majority_vote": False,
        "max_tokens": 512,
        "prompt_template_hash": "prompt-hash",
        "tokenizer_hash": "tokenizer-hash",
        "model_hash": "model-hash",
        "adapter_hash": "adapter-hash",
    }
    data.update(updates)
    return ServingConfig(**data)


def test_no_backend_strict_mode_fails_without_loaded_claims() -> None:
    report = validate_offline_runtime(RuntimeValidationConfig(serving_config=serving(), dry_run=True))
    assert not report.passed
    assert not report.runtime_loaded
    assert not report.model_loaded
    assert not report.adapter_loaded


def test_dry_run_allow_unloaded_can_pass_with_warning() -> None:
    report = validate_offline_runtime(
        RuntimeValidationConfig(serving_config=serving(), dry_run=True, dry_run_allow_unloaded=True)
    )
    assert report.passed
    assert "runtime_unloaded_dry_run" in report.warnings
    assert not report.runtime_loaded


def test_fake_runtime_loaded_without_backend_rejected() -> None:
    with pytest.raises(RuntimeValidationError):
        RuntimeValidationReport(
            passed=True,
            dry_run=False,
            runtime_loaded=True,
            model_loaded=True,
            adapter_loaded=False,
            backend_validated=False,
            backend_name=None,
            manifest_hash=None,
            serving_config_hash="serving",
            adapter_checked=False,
        )


def test_malformed_backend_success_rejected() -> None:
    with pytest.raises(RuntimeValidationError):
        validate_offline_runtime(
            RuntimeValidationConfig(
                serving_config=serving(),
                dry_run=False,
                backend_result={"backend_name": "mock", "runtime_loaded": True, "model_loaded": True},
            )
        )


def test_backend_failure_rejected() -> None:
    with pytest.raises(RuntimeValidationError):
        validate_offline_runtime(
            RuntimeValidationConfig(
                serving_config=serving(),
                dry_run=False,
                backend_result={
                    "backend_name": "mock",
                    "runtime_loaded": True,
                    "model_loaded": False,
                    "adapter_loaded": False,
                },
            )
        )


def test_backend_explicit_success_accepted() -> None:
    report = validate_offline_runtime(
        RuntimeValidationConfig(
            serving_config=serving(),
            dry_run=False,
            backend_result={
                "backend_name": "mock",
                "runtime_loaded": True,
                "model_loaded": True,
                "adapter_loaded": False,
            },
        )
    )
    assert report.passed
    assert report.backend_validated
    assert report.runtime_loaded and report.model_loaded


def test_runtime_validator_rejects_kaggle_backend_name() -> None:
    with pytest.raises(RuntimeValidationError):
        validate_offline_runtime(
            RuntimeValidationConfig(
                serving_config=serving(),
                dry_run=False,
                backend_result={
                    "backend_name": "kaggle-runtime",
                    "runtime_loaded": True,
                    "model_loaded": True,
                    "adapter_loaded": False,
                },
            )
        )


@pytest.mark.parametrize("claim", ["kaggle_runtime_success", "submission_success", "leaderboard_success"])
def test_runtime_validator_rejects_forbidden_success_claims(claim: str) -> None:
    with pytest.raises(RuntimeValidationError):
        validate_offline_runtime(
            RuntimeValidationConfig(
                serving_config=serving(),
                dry_run=False,
                backend_result={
                    "backend_name": "mock",
                    "runtime_loaded": True,
                    "model_loaded": True,
                    "adapter_loaded": False,
                    claim: True,
                },
            )
        )


def test_backend_metadata_adapter_config_rank_above_32_rejected() -> None:
    report = validate_offline_runtime(
        RuntimeValidationConfig(
            serving_config=serving(),
            adapter_config={"r": 64},
            dry_run=True,
            dry_run_allow_unloaded=True,
        )
    )
    assert not report.passed
    assert any("<= 32" in error for error in report.errors)


def test_adapter_dir_rank_above_32_rejected() -> None:
    with temp_root("adapter_rank") as tmp_path:
        (tmp_path / "adapter_config.json").write_text('{"r": 64}\n', encoding="utf-8")
        report = validate_offline_runtime(
            RuntimeValidationConfig(
                serving_config=serving(),
                adapter_dir=tmp_path,
                dry_run=True,
                dry_run_allow_unloaded=True,
            )
        )
        assert not report.passed
        assert any("<= 32" in error for error in report.errors)


@pytest.mark.parametrize("payload", ['{"r": "64"}\n', '{"r": true}\n'])
def test_adapter_dir_malformed_rank_rejected(payload: str) -> None:
    with temp_root("adapter_malformed") as tmp_path:
        (tmp_path / "adapter_config.json").write_text(payload, encoding="utf-8")
        report = validate_offline_runtime(
            RuntimeValidationConfig(
                serving_config=serving(),
                adapter_dir=tmp_path,
                dry_run=True,
                dry_run_allow_unloaded=True,
            )
        )
        assert not report.passed
        assert any("integer" in error for error in report.errors)


def test_forged_runtime_report_hash_rejected() -> None:
    report = validate_offline_runtime(
        RuntimeValidationConfig(serving_config=serving(), dry_run=True, dry_run_allow_unloaded=True)
    )
    with pytest.raises(RuntimeValidationError):
        replace(report, report_hash="forged")
