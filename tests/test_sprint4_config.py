from __future__ import annotations

from dataclasses import replace
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nemotron_engine.adapter_training_data.kaggle_export import Sprint4ExportError, load_sprint4_config  # noqa: E402
from kaggle_sprint4.kaggle_train_curated_adapter import parse_dataset_mixture, validate_training_config  # noqa: E402


def write_config(path: Path, dataset_path: str, *, rank: int = 32, variant: str = "variant_c_hard_oversampled") -> None:
    path.write_text(
        f"""parent_adapter_path: null
curated_dataset_path: {dataset_path}
output_adapter_dir: /kaggle/working/custom_adapter
submission_zip_path: /kaggle/working/submission.zip
variant_name: {variant}
max_train_rows: 12000
dataset_mixture:
  - train_family_tagged.jsonl
  - train_solver_distilled.jsonl
lora_rank: {rank}
lora_alpha: 64
lora_dropout: 0.05
learning_rate: 0.00002
num_epochs: 1
max_seq_len: 2048
batch_size: 1
grad_accum: 4
seed: 1337
""",
        encoding="utf-8",
    )


def test_config_validates_known_variant_and_dataset_path(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    config_path = tmp_path / "config.yaml"
    write_config(config_path, str(dataset))

    config = load_sprint4_config(config_path)

    assert config.variant_name == "variant_c_hard_oversampled"
    assert config.lora_rank == 32
    assert config.dataset_mixture == ("train_family_tagged.jsonl", "train_solver_distilled.jsonl")


def test_config_rejects_lora_rank_over_32(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    config_path = tmp_path / "config.yaml"
    write_config(config_path, str(dataset), rank=64)

    with pytest.raises(Sprint4ExportError):
        load_sprint4_config(config_path)


def test_config_rejects_missing_dataset_path(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    write_config(config_path, str(tmp_path / "missing"))

    with pytest.raises(Sprint4ExportError):
        load_sprint4_config(config_path)


def test_config_rejects_unknown_variant(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    config_path = tmp_path / "config.yaml"
    write_config(config_path, str(dataset), variant="surprise")

    with pytest.raises(Sprint4ExportError):
        load_sprint4_config(config_path)


def test_config_hash_forgery_rejected(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    config_path = tmp_path / "config.yaml"
    write_config(config_path, str(dataset))
    config = load_sprint4_config(config_path)

    with pytest.raises(Sprint4ExportError):
        replace(config, config_hash="forged")


def test_training_config_rejects_high_lr_unless_explicitly_allowed() -> None:
    config = {
        "variant_name": "retention_raw_family_tagged",
        "lora_rank": 32,
        "learning_rate": 0.00001,
        "dataset_mixture": {"train_family_tagged.jsonl": 1.0},
    }

    with pytest.raises(ValueError):
        validate_training_config(dict(config))

    config["allow_high_lr"] = True
    assert validate_training_config(dict(config))["learning_rate"] == 0.00001


def test_training_config_rejects_full_prompt_loss_without_override() -> None:
    config = {
        "variant_name": "retention_raw_family_tagged",
        "lora_rank": 32,
        "learning_rate": 0.000001,
        "dataset_mixture": {"train_family_tagged.jsonl": 1.0},
        "assistant_only_loss": False,
    }

    with pytest.raises(ValueError):
        validate_training_config(config)


def test_dataset_mixture_mapping_weights_parsed_correctly() -> None:
    mixture = parse_dataset_mixture({"train_solver_distilled.jsonl": 0.7, "train_hard_oversampled.jsonl": 0.3})

    assert mixture == [("train_solver_distilled.jsonl", 0.7), ("train_hard_oversampled.jsonl", 0.3)]


def test_dataset_mixture_already_normalized_pairs_round_trip() -> None:
    mixture = parse_dataset_mixture([("train_solver_distilled.jsonl", 0.7)])

    assert mixture == [("train_solver_distilled.jsonl", 0.7)]
