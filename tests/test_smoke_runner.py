from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from nemotron_engine.release.import_safety import ImportSafetyConfig, audit_import_safety
from nemotron_engine.smoke_training import (
    SmokeRunConfig,
    SmokeRunReport,
    SmokeRunnerError,
    SmokeTrainingConfig,
    build_tiny_sft_smoke_dataset,
    run_smoke_training,
)
from nemotron_engine.smoke_training.smoke_backend import SmokeBackendResult
from nemotron_engine.training_backend import AdapterValidationConfig, BackendInvocation


SHA = "e" * 64


def invocation(*, dry_run: bool = False) -> BackendInvocation:
    return BackendInvocation(
        invocation_id="smoke-inv",
        backend_kind="external_sft",
        stage="sft",
        training_plan_hash="plan",
        lora_config_hash="lora",
        dataset_manifest_hash="dataset",
        input_row_count=1,
        input_pair_count=0,
        seed=1,
        dry_run=dry_run,
        output_dir=None if dry_run else "out/smoke",
    )


def run_config(tmp_path, *, dry_run: bool = False, require_log: bool = True) -> SmokeRunConfig:
    adapter = write_adapter(tmp_path)
    return SmokeRunConfig(
        smoke_config=SmokeTrainingConfig(
            stage="sft",
            max_examples=1,
            max_steps=2,
            max_runtime_seconds=30,
            dry_run=dry_run,
            output_dir=None if dry_run else "out/smoke",
            require_log_validation=require_log,
        ),
        smoke_dataset=build_tiny_sft_smoke_dataset(),
        invocation=invocation(dry_run=dry_run),
        adapter_validation_config=AdapterValidationConfig(adapter),
    )


def write_adapter(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 8}), encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"x")
    return adapter


def backend_payload(inv: BackendInvocation, *, adapter: bool = True):
    return {
        "result_id": "res",
        "invocation_hash": inv.invocation_hash,
        "backend_kind": inv.backend_kind,
        "stage": inv.stage,
        "status": "success",
        "trained": True,
        "adapter_path": "adapter" if adapter else None,
        "checkpoint_path": None,
        "artifacts": [
            {"path": "adapter/adapter_model.safetensors", "role": "adapter_model", "sha256": SHA, "size_bytes": 1, "required": True}
        ]
        if adapter
        else [],
        "metrics": [{"name": "loss", "value": 1.0, "step": 0, "split": None, "metadata": {}}],
        "started": True,
        "finished": True,
        "error": None,
        "metadata": {},
    }


def test_dry_run_does_not_call_backend_and_trained_false(tmp_path) -> None:
    called = {"count": 0}

    def backend(_inv):
        called["count"] += 1
        raise AssertionError("backend should not be called")

    report = run_smoke_training(run_config(tmp_path, dry_run=True), backend)
    assert called["count"] == 0
    assert report.trained is False
    assert report.called_backend is False
    assert report.passed is True


def test_non_dry_run_without_backend_raises(tmp_path) -> None:
    with pytest.raises(SmokeRunnerError):
        run_smoke_training(run_config(tmp_path, dry_run=False), None)


def test_backend_exception_becomes_failed_report(tmp_path) -> None:
    def backend(_inv):
        raise RuntimeError("boom")

    report = run_smoke_training(run_config(tmp_path), backend)
    assert report.passed is False
    assert report.trained is False
    assert any("backend exception" in error for error in report.errors)


def test_fake_trained_backend_without_adapter_rejected(tmp_path) -> None:
    def backend(inv):
        return {"backend_result": backend_payload(inv, adapter=False), "training_log": [{"step": 0, "loss": 1.0}]}

    report = run_smoke_training(run_config(tmp_path), backend)
    assert report.passed is False
    assert report.trained is False
    assert report.training_run_manifest_hash is None


def test_trained_backend_with_valid_temp_adapter_and_log_passes(tmp_path) -> None:
    cfg = run_config(tmp_path)
    adapter = cfg.adapter_validation_config.adapter_dir

    def backend(inv):
        return SmokeBackendResult(
            backend_payload(inv),
            adapter_dir=adapter,
            training_log=[{"step": 0, "loss": 1.0, "grad_norm": 1.0}],
        )

    report = run_smoke_training(cfg, backend)
    assert report.passed
    assert report.trained
    assert report.backend_result_hash
    assert report.adapter_validation_hash
    assert report.training_log_hash
    assert report.training_run_manifest_hash


def test_runner_rejects_invalid_training_log_when_required(tmp_path) -> None:
    cfg = run_config(tmp_path)
    adapter = cfg.adapter_validation_config.adapter_dir

    def backend(inv):
        return SmokeBackendResult(
            backend_payload(inv),
            adapter_dir=adapter,
            training_log=[{"step": 0, "unsupported_metric": 1.0}],
        )

    report = run_smoke_training(cfg, backend)
    assert not report.passed
    assert not report.trained
    assert any("unsupported log key" in error or "training log validation" in error for error in report.errors)


def test_runner_rejects_adapter_dir_mismatch(tmp_path) -> None:
    cfg = run_config(tmp_path)
    other = tmp_path / "other_adapter"

    def backend(inv):
        return SmokeBackendResult(
            backend_payload(inv),
            adapter_dir=other,
            training_log=[{"step": 0, "loss": 1.0}],
        )

    report = run_smoke_training(cfg, backend)
    assert not report.passed
    assert not report.trained
    assert report.training_run_manifest_hash is None


def test_runner_rejects_failed_adapter_validation_report(tmp_path) -> None:
    cfg = run_config(tmp_path)
    adapter = cfg.adapter_validation_config.adapter_dir
    (Path(adapter) / "optimizer.pt").write_bytes(b"x")

    def backend(inv):
        return SmokeBackendResult(
            backend_payload(inv),
            adapter_dir=adapter,
            training_log=[{"step": 0, "loss": 1.0}],
        )

    report = run_smoke_training(cfg, backend)
    assert not report.passed
    assert not report.trained
    assert report.adapter_validation_hash
    assert report.training_run_manifest_hash is None


def test_smoke_run_report_rejects_forged_hash_and_passed_with_errors(tmp_path) -> None:
    report = run_smoke_training(run_config(tmp_path, dry_run=True), backend=None)
    with pytest.raises(SmokeRunnerError, match="report_hash"):
        replace(report, report_hash="forged")
    with pytest.raises(SmokeRunnerError, match="passed=True"):
        SmokeRunReport(
            smoke_config_hash="c",
            smoke_dataset_hash="d",
            backend_invocation_hash="i",
            backend_run_report_hash="b",
            backend_result_hash="r",
            adapter_validation_hash=None,
            training_log_hash=None,
            training_run_manifest_hash=None,
            called_backend=False,
            trained=False,
            passed=True,
            errors=("bad",),
            metadata={"dry_run": True},
        )


def test_smoke_run_report_strict_trained_evidence_requirements() -> None:
    base = dict(
        smoke_config_hash="c",
        smoke_dataset_hash="d",
        backend_invocation_hash="i",
        backend_run_report_hash="br",
        backend_result_hash="r",
        adapter_validation_hash="a",
        training_log_hash="l",
        training_run_manifest_hash="m",
        called_backend=True,
        trained=True,
        passed=True,
    )
    with pytest.raises(SmokeRunnerError):
        SmokeRunReport(**{**base, "training_log_hash": None, "metadata": {}})
    with pytest.raises(SmokeRunnerError):
        SmokeRunReport(**{**base, "backend_run_report_hash": None})


def test_smoke_run_report_dry_run_pass_requires_explicit_metadata() -> None:
    with pytest.raises(SmokeRunnerError):
        SmokeRunReport(
            "c",
            "d",
            "i",
            "br",
            None,
            None,
            None,
            None,
            False,
            False,
            True,
            metadata={},
        )
    report = SmokeRunReport(
        "c",
        "d",
        "i",
        "br",
        None,
        None,
        None,
        None,
        False,
        False,
        True,
        metadata={"dry_run": True},
    )
    assert report.passed


def test_pass12_source_has_no_shell_model_or_kaggle_calls() -> None:
    report = audit_import_safety(
        ImportSafetyConfig(repository_root=".", scan_roots=("src/nemotron_engine/smoke_training",), allow_test_files=False)
    )
    assert report.passed, report.errors
