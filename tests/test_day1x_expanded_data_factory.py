from __future__ import annotations

from pathlib import Path

from kaggle_anti086.training.day1_teacher_trainability_gate import verify_trainable_row
from kaggle_anti086.training.day1_verified_data_schema import read_jsonl
from kaggle_anti086.training.day1x_composed_teachers import REQUIRED_COMPOSED_FAMILIES
from kaggle_anti086.training.day1x_expanded_data_factory import (
    ORIGINAL_FAMILIES,
    build_expanded_dataset,
    generate_composed_family_rows,
    generate_original_family_rows,
    validate_expanded_split,
)


def test_small_smoke_dataset_generates_requested_splits_and_valid_rows(tmp_path: Path) -> None:
    result = build_expanded_dataset(220, 40, 40, 123, tmp_path, stage="smoke")
    assert result["status"] == "PASS"
    assert result["counts"] == {"train": 220, "eval": 40, "probe": 40}
    rows = read_jsonl(Path(result["files"]["train"])) + read_jsonl(Path(result["files"]["eval"])) + read_jsonl(Path(result["files"]["probe"]))
    assert all(verify_trainable_row(row).trainable for row in rows)
    assert all(row["verification_status"] == "PASS" for row in rows)
    assert all(row["ambiguity_count"] == 0 for row in rows)
    assert len({row["id"] for row in rows}) == len(rows)
    assert len({row["prompt"] for row in rows}) == len(rows)
    assert {"original_fraction", "composed_fraction", "public_style_fraction", "format_stress_fraction"} <= set(result["mixture_report"])
    rejected = read_jsonl(tmp_path / "day1x_rejected_rows.jsonl")
    assert rejected and rejected[0]["rejection_reasons"]


def test_original_and_composed_generators_cover_all_families_when_enough_rows() -> None:
    original = generate_original_family_rows(len(ORIGINAL_FAMILIES), 1)
    composed = generate_composed_family_rows(len(REQUIRED_COMPOSED_FAMILIES), 1)
    assert set(ORIGINAL_FAMILIES).issubset({row["family"] for row in original})
    assert set(REQUIRED_COMPOSED_FAMILIES).issubset({row["family"] for row in composed})


def test_stage_writing_and_validation_reports_expected_files(tmp_path: Path) -> None:
    result = build_expanded_dataset(20, 5, 5, 44, tmp_path, stage="25k")
    assert Path(result["files"]["train"]).name == "day1x_verified_sft_train_25k.jsonl"
    assert Path(result["files"]["eval"]).name == "day1x_verified_sft_eval_2k.jsonl"
    assert Path(result["files"]["probe"]).name == "day1x_verified_probe_2k.jsonl"
    assert validate_expanded_split(read_jsonl(Path(result["files"]["train"])), "train")["status"] == "PASS"


def test_generation_is_deterministic_and_seed_sensitive(tmp_path: Path) -> None:
    first = build_expanded_dataset(20, 5, 5, 5, tmp_path / "a", stage="dryrun")
    second = build_expanded_dataset(20, 5, 5, 5, tmp_path / "b", stage="dryrun")
    third = build_expanded_dataset(20, 5, 5, 6, tmp_path / "c", stage="dryrun")
    first_ids = [row["id"] for row in read_jsonl(Path(first["files"]["train"]))[:5]]
    second_ids = [row["id"] for row in read_jsonl(Path(second["files"]["train"]))[:5]]
    third_ids = [row["id"] for row in read_jsonl(Path(third["files"]["train"]))[:5]]
    assert first_ids == second_ids
    assert first_ids != third_ids
