from __future__ import annotations

from dataclasses import replace
import json
import math

import pytest

from nemotron_engine.training_backend.training_log_validator import (
    TrainingLogValidationError,
    validate_training_log,
)


def test_valid_metric_log_passes() -> None:
    report = validate_training_log([{"step": 0, "loss": 2.0, "grad_norm": 1.0}, {"step": 1, "eval_loss": 1.5, "accuracy": 0.5}])
    assert report.passed
    assert report.entry_count == 2
    assert report.last_step == 1


def test_nan_inf_metric_rejected() -> None:
    assert not validate_training_log([{"step": 1, "loss": math.nan}]).passed
    assert not validate_training_log([{"step": 1, "loss": math.inf}]).passed


def test_loss_explosion_and_grad_norm_too_high_rejected() -> None:
    assert not validate_training_log([{"step": 1, "loss": 101.0}]).passed
    assert not validate_training_log([{"step": 1, "grad_norm": 1001.0}]).passed


def test_bad_error_rates_rejected() -> None:
    assert not validate_training_log([{"step": 1, "format_error_rate": 1.5}]).passed
    assert not validate_training_log([{"step": 1, "extraction_error_rate": -0.1}]).passed


def test_fake_kaggle_leaderboard_score_claims_rejected() -> None:
    assert not validate_training_log([{"step": 1, "loss": 1.0, "kaggle_success": True}]).passed
    assert not validate_training_log([{"step": 1, "loss": 1.0, "claim": "leaderboard success"}]).passed
    assert not validate_training_log([{"step": 1, "loss": 1.0, "claim": "95+ guaranteed"}]).passed
    assert not validate_training_log([{"step": 1, "loss": 1.0, "trained": True}]).passed


def test_unsupported_log_keys_rejected_but_metadata_allowed() -> None:
    assert not validate_training_log([{"step": 1, "loss": 1.0, "note": "hello"}]).passed
    assert not validate_training_log([{"step": 1, "loss": 1.0, "custom_metric": 0.1}]).passed
    assert validate_training_log([{"step": 1, "loss": 1.0, "metadata": {"backend": "local"}}]).passed
    assert not validate_training_log([{"step": 1, "loss": 1.0, "metadata": {"leaderboard_success": True}}]).passed


def test_jsonl_parse_deterministic(tmp_path) -> None:
    lines = [json.dumps({"step": 0, "loss": 1.0}), json.dumps({"step": 1, "reward": 0.2})]
    path = tmp_path / "train.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert validate_training_log(lines).series_hash == validate_training_log(path).series_hash


def test_malformed_jsonl_line_rejected() -> None:
    with pytest.raises(TrainingLogValidationError, match="JSONL"):
        validate_training_log(['{"step": 1, "loss":'])


def test_forged_report_hash_rejected() -> None:
    report = validate_training_log([{"step": 0, "loss": 1.0}])
    with pytest.raises(TrainingLogValidationError, match="report_hash"):
        replace(report, report_hash="forged")
