from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from kaggle_anti086.training.day1_verified_data_schema import read_jsonl, write_jsonl
from kaggle_anti086.training.day1x_expanded_data_factory import build_expanded_dataset
from kaggle_anti086.training.day1x_expanded_manifest import build_day1x_100k_decision_report, build_expanded_manifest
from kaggle_anti086.training.day1x_holdout_builder import write_holdout_files


def _expected(*stages: str) -> dict[str, dict[str, int]]:
    sizes = {"25k": {"train": 30, "eval": 10, "probe": 10}, "50k": {"train": 40, "eval": 10, "probe": 10}, "100k": {"train": 120, "eval": 20, "probe": 20}}
    return {stage: sizes[stage] for stage in stages}


def test_manifest_pass_on_clean_small_100k_with_threshold_override(tmp_path: Path) -> None:
    write_holdout_files(tmp_path, 2, 1)
    build_expanded_dataset(120, 20, 20, 1, tmp_path, stage="100k")
    manifest = build_expanded_manifest(tmp_path, expected_stage_sizes=_expected("100k"))
    assert manifest["status"] == "PASS"
    assert manifest["selected_stage_for_day2"] == "100k"


def test_manifest_warn_when_only_50k_equivalent_exists(tmp_path: Path) -> None:
    write_holdout_files(tmp_path, 2, 2)
    build_expanded_dataset(40, 10, 10, 2, tmp_path, stage="50k")
    manifest = build_expanded_manifest(tmp_path, expected_stage_sizes=_expected("50k", "100k"))
    assert manifest["status"] == "WARN"
    assert manifest["selected_stage_for_day2"] == "50k"


def test_manifest_fails_on_safety_counter_and_split_leakage(tmp_path: Path) -> None:
    write_holdout_files(tmp_path, 2, 3)
    result = build_expanded_dataset(120, 20, 20, 3, tmp_path, stage="100k")
    train_path = Path(result["files"]["train"])
    rows = read_jsonl(train_path)
    rows[0]["target_text"] = "\\boxed{1}\\boxed{2}"
    write_jsonl(train_path, rows)
    manifest = build_expanded_manifest(tmp_path, expected_stage_sizes=_expected("100k"))
    assert manifest["status"] == "FAIL"
    assert manifest["format_error_count"] > 0

    rows = read_jsonl(train_path)
    rows[0] = deepcopy(rows[1])
    write_jsonl(train_path, rows)
    manifest = build_expanded_manifest(tmp_path, expected_stage_sizes=_expected("100k"))
    assert manifest["split_leakage_count"] > 0 or manifest["duplicate_prompt_count"] > 0


def test_manifest_falls_back_only_when_no_clean_expanded_stage(tmp_path: Path) -> None:
    write_holdout_files(tmp_path, 2, 4)
    manifest = build_expanded_manifest(tmp_path, expected_stage_sizes=_expected("25k"))
    assert manifest["status"] == "FAIL"
    assert manifest["selected_stage_for_day2"] == "basic_day1"
    assert manifest["fallback_to_basic_day1"] is True


def test_decision_report_flags_and_next_phase(tmp_path: Path) -> None:
    write_holdout_files(tmp_path, 2, 5)
    build_expanded_dataset(30, 10, 10, 5, tmp_path, stage="25k")
    manifest = build_expanded_manifest(tmp_path, expected_stage_sizes=_expected("25k"))
    decision = build_day1x_100k_decision_report(manifest)
    assert decision["training_authorized"] is True
    assert decision["package_authorized"] is False
    assert decision["submission_authorized"] is False
    assert decision["leaderboard_claim"] is False
    assert decision["no_0_93_evidence"] is True
    assert decision["next_phase"] == "DAY1X_DPO_HARD_NEGATIVE_EXPANSION_AND_VARIANTS"
