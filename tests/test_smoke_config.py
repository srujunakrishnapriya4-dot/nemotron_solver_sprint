from __future__ import annotations

from dataclasses import replace

import pytest

from nemotron_engine.smoke_training.smoke_config import (
    SmokeTrainingConfig,
    SmokeTrainingConfigError,
    compute_smoke_config_hash,
    validate_smoke_training_config,
)


def test_valid_config_passes_and_hash_is_deterministic() -> None:
    config = SmokeTrainingConfig(stage="sft", max_examples=2, max_steps=3, max_runtime_seconds=30)
    assert validate_smoke_training_config(config) is config
    assert config.config_hash == compute_smoke_config_hash(config)
    assert config.config_hash == SmokeTrainingConfig(stage="sft", max_examples=2, max_steps=3, max_runtime_seconds=30).config_hash


def test_bounds_and_grpo_rejected() -> None:
    with pytest.raises(SmokeTrainingConfigError):
        SmokeTrainingConfig(stage="sft", max_examples=17, max_steps=3, max_runtime_seconds=30)
    with pytest.raises(SmokeTrainingConfigError):
        SmokeTrainingConfig(stage="sft", max_examples=2, max_steps=21, max_runtime_seconds=30)
    with pytest.raises(SmokeTrainingConfigError):
        SmokeTrainingConfig(stage="sft", max_examples=2, max_steps=3, max_runtime_seconds=901)
    with pytest.raises(SmokeTrainingConfigError):
        SmokeTrainingConfig(stage="grpo", max_examples=2, max_steps=3, max_runtime_seconds=30)


def test_bool_seed_and_string_dry_run_rejected() -> None:
    with pytest.raises(SmokeTrainingConfigError):
        SmokeTrainingConfig(stage="sft", max_examples=2, max_steps=3, max_runtime_seconds=30, seed=True)
    with pytest.raises(SmokeTrainingConfigError):
        SmokeTrainingConfig(stage="sft", max_examples=2, max_steps=3, max_runtime_seconds=30, dry_run="false")  # type: ignore[arg-type]


def test_output_dir_required_when_not_dry_run() -> None:
    with pytest.raises(SmokeTrainingConfigError):
        SmokeTrainingConfig(stage="sft", max_examples=2, max_steps=3, max_runtime_seconds=30, dry_run=False)
    assert SmokeTrainingConfig(stage="sft", max_examples=2, max_steps=3, max_runtime_seconds=30, dry_run=False, output_dir="out")


def test_forged_config_hash_rejected() -> None:
    config = SmokeTrainingConfig(stage="sft", max_examples=2, max_steps=3, max_runtime_seconds=30)
    with pytest.raises(SmokeTrainingConfigError, match="config_hash"):
        replace(config, config_hash="forged")


@pytest.mark.parametrize(
    "metadata",
    [
        {"kaggle_success": True},
        {"leaderboard_ready": True},
        {"submission": "passed"},
        {"note": "95+ guaranteed"},
        {"score": "guaranteed"},
    ],
)
def test_forbidden_success_metadata_rejected(metadata) -> None:
    with pytest.raises(SmokeTrainingConfigError):
        SmokeTrainingConfig(stage="sft", max_examples=2, max_steps=3, max_runtime_seconds=30, metadata=metadata)
