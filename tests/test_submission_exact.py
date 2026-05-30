from __future__ import annotations

from dataclasses import replace
import io
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.evaluation.submission_exact import (
    SubmissionExactConfig,
    SubmissionExactError,
    SubmissionExactReport,
    compare_serving_configs,
    validate_submission_artifact_refs,
    validate_submission_exact_config,
)
from nemotron_engine.runtime.serving_config import ServingConfig


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


def test_accepts_strict_default_serving_config_with_required_hashes() -> None:
    report = validate_submission_exact_config(SubmissionExactConfig(serving_config=serving(), adapter_config={"r": 8}))

    assert report.passed is True
    assert report.required_hashes_present is True
    assert report.adapter_checked is True
    assert report.adapter_rank == 8


def test_rejects_non_exact_decoding_settings() -> None:
    for config in (
        serving(temperature=0.1, strict_submission_mode=False),
        serving(top_p=0.9),
        serving(num_samples=2, strict_submission_mode=False),
        serving(majority_vote=True, strict_submission_mode=False),
        serving(max_tokens=0, strict_submission_mode=False),
    ):
        report = validate_submission_exact_config(SubmissionExactConfig(serving_config=config, adapter_config={"r": 8}))
        assert report.passed is False


def test_rejects_missing_hashes_in_strict_mode() -> None:
    report = validate_submission_exact_config(SubmissionExactConfig(serving_config=serving(model_hash=None), adapter_config={"r": 8}))

    assert report.passed is False
    assert any("missing required hashes" in error for error in report.errors)


def test_validate_submission_artifact_refs_rejects_zero_and_multiple_sources() -> None:
    assert validate_submission_artifact_refs().valid is False
    assert validate_submission_artifact_refs(adapter_config={"r": 8}, adapter_dir="x").valid is False


def test_validate_submission_artifact_refs_rejects_bad_adapter_config() -> None:
    result = validate_submission_artifact_refs(adapter_config={"r": 33})

    assert result.valid is False
    assert result.errors


def test_rejects_adapter_rank_above_32() -> None:
    report = validate_submission_exact_config(SubmissionExactConfig(serving_config=serving(), adapter_config={"r": 33}))

    assert report.passed is False
    assert any("rank" in error or "must be <=" in error for error in report.errors)


def test_compare_serving_configs_rejects_adapter_hash_mismatch() -> None:
    report = compare_serving_configs(serving(adapter_hash="a"), serving(adapter_hash="b"))

    assert report.passed is False
    assert any("adapter_hash" in error for error in report.errors)


def test_report_hash_deterministic_and_forgery_rejected() -> None:
    report = validate_submission_exact_config(SubmissionExactConfig(serving_config=serving(), adapter_config={"r": 8}))
    again = validate_submission_exact_config(SubmissionExactConfig(serving_config=serving(), adapter_config={"r": 8}))

    assert report.report_hash == again.report_hash
    with pytest.raises(SubmissionExactError):
        replace(report, report_hash="forged")


def test_report_rejects_passed_true_with_bad_critical_fields() -> None:
    report = validate_submission_exact_config(SubmissionExactConfig(serving_config=serving(), adapter_config={"r": 8}))
    cases = (
        {"temperature": 0.1},
        {"num_samples": 2},
        {"majority_vote": True},
        {"prompt_template_hash": ""},
        {"tokenizer_hash": ""},
        {"model_hash": ""},
    )
    for update in cases:
        critical = dict(report.critical_serving_fields)
        critical.update(update)
        with pytest.raises(SubmissionExactError):
            replace(report, critical_serving_fields=critical)


def test_compare_serving_configs_rejects_all_critical_mismatches_without_artifacts() -> None:
    cases = (
        {"stop": ("END",)},
        {"batch_size": 2},
        {"max_tokens": 256},
        {"prompt_template_hash": "other-prompt"},
        {"tokenizer_hash": "other-tokenizer"},
        {"model_hash": "other-model"},
        {"adapter_hash": "other-adapter"},
    )
    for update in cases:
        report = compare_serving_configs(serving(), serving(**update))
        assert report.passed is False
        assert any(next(iter(update)) in error for error in report.errors)

    assert compare_serving_configs(serving(), serving()).passed is True


def test_validate_submission_artifact_refs_rejects_adapter_dir_and_zip_rank_above_32() -> None:
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("adapter_config.json", json.dumps({"r": 33}))
    assert validate_submission_artifact_refs(submission_zip=buffer.getvalue()).valid is False

    adapter_dir = Path(__file__).parent / ".pass8_adapter_rank_probe"
    try:
        adapter_dir.mkdir()
    except PermissionError:
        return
    try:
        (adapter_dir / "adapter_config.json").write_text(json.dumps({"r": 33}), encoding="utf-8")
        assert validate_submission_artifact_refs(adapter_dir=adapter_dir).valid is False
    finally:
        try:
            (adapter_dir / "adapter_config.json").unlink()
            adapter_dir.rmdir()
        except OSError:
            pass


def test_dry_run_report_cannot_claim_runtime_success() -> None:
    report = validate_submission_exact_config(SubmissionExactConfig(serving_config=serving(), adapter_config={"r": 8}, dry_run=True))

    assert report.runtime_success is False
    with pytest.raises(SubmissionExactError):
        SubmissionExactReport(
            config_hash=report.config_hash,
            serving_config_hash=report.serving_config_hash,
            passed=True,
            dry_run=True,
            runtime_success=True,
            required_hashes_present=True,
            adapter_checked=False,
        )
