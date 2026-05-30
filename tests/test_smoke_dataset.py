from __future__ import annotations

from dataclasses import replace

import pytest

from nemotron_engine.smoke_training.smoke_config import SmokeTrainingConfig
from nemotron_engine.smoke_training.smoke_dataset import (
    SmokeDataset,
    SmokeDatasetError,
    SmokeDatasetExample,
    build_tiny_dpo_smoke_dataset,
    build_tiny_sft_smoke_dataset,
    compute_smoke_dataset_hash,
    validate_smoke_dataset,
)


def config(stage: str = "sft", max_examples: int = 4) -> SmokeTrainingConfig:
    return SmokeTrainingConfig(stage=stage, max_examples=max_examples, max_steps=2, max_runtime_seconds=30)


def test_tiny_sft_and_dpo_datasets_are_deterministic() -> None:
    assert build_tiny_sft_smoke_dataset().dataset_hash == build_tiny_sft_smoke_dataset().dataset_hash
    assert build_tiny_dpo_smoke_dataset().dataset_hash == build_tiny_dpo_smoke_dataset().dataset_hash
    validate_smoke_dataset(build_tiny_sft_smoke_dataset(), config())
    validate_smoke_dataset(build_tiny_dpo_smoke_dataset(), config("dpo"))


def test_dataset_size_limit_and_empty_examples_rejected() -> None:
    example = build_tiny_sft_smoke_dataset().examples[0]
    other = SmokeDatasetExample("smoke-sft-0002", "sft", "3 -> 4\n5 -> ?", target="6")
    dataset = SmokeDataset("too-large", "sft", (example, other))
    with pytest.raises(SmokeDatasetError):
        validate_smoke_dataset(dataset, config(max_examples=1))
    with pytest.raises(SmokeDatasetError):
        SmokeDataset("empty", "sft", ())


def test_required_sft_and_dpo_fields_enforced() -> None:
    with pytest.raises(SmokeDatasetError):
        SmokeDatasetExample("e", "sft", "", target="target")
    with pytest.raises(SmokeDatasetError):
        SmokeDatasetExample("e", "sft", "prompt", target=None)
    with pytest.raises(SmokeDatasetError):
        SmokeDatasetExample("e", "dpo", "", chosen="chosen", rejected="rejected")
    with pytest.raises(SmokeDatasetError):
        SmokeDatasetExample("e", "dpo", "prompt", chosen=None, rejected="rejected")
    with pytest.raises(SmokeDatasetError):
        SmokeDatasetExample("e", "dpo", "prompt", chosen="same", rejected="same")
    with pytest.raises(SmokeDatasetError):
        SmokeDatasetExample("e", "dpo", "prompt", chosen="ok", rejected=None)


def test_forged_example_and_dataset_hashes_rejected() -> None:
    example = build_tiny_sft_smoke_dataset().examples[0]
    with pytest.raises(SmokeDatasetError, match="example_hash"):
        replace(example, example_hash="forged")
    dataset = build_tiny_sft_smoke_dataset()
    with pytest.raises(SmokeDatasetError, match="dataset_hash"):
        replace(dataset, dataset_hash="forged")
    assert dataset.dataset_hash == compute_smoke_dataset_hash(dataset)


def test_duplicate_example_id_rejected() -> None:
    example = build_tiny_sft_smoke_dataset().examples[0]
    other = SmokeDatasetExample(example.example_id, "sft", "x", target="y")
    with pytest.raises(SmokeDatasetError):
        SmokeDataset("dup", "sft", (example, other))


@pytest.mark.parametrize(
    "metadata",
    [
        {"split": "eval"},
        {"split": "private_like"},
        {"split": "holdout"},
        {"split": "stress_only"},
        {"contaminated": True},
        {"contamination_flags": ("x",)},
        {"leaderboard_ready": True},
    ],
)
def test_contaminated_eval_private_holdout_stress_metadata_rejected(metadata) -> None:
    with pytest.raises(SmokeDatasetError):
        SmokeDatasetExample("bad", "sft", "prompt", target="target", metadata=metadata)
    with pytest.raises(SmokeDatasetError):
        build_tiny_sft_smoke_dataset(metadata=metadata)
