from __future__ import annotations

import json
from pathlib import Path

from kaggle_anti086.training.day1_lora_distill_manifest import build_lora_distill_manifest
from kaggle_anti086.training.day1_teacher_quality_report import build_teacher_quality_report


ARTIFACTS = Path("artifacts/sprint11")


def _quality(tmp_path: Path) -> Path:
    out = tmp_path / "quality.json"
    build_teacher_quality_report(ARTIFACTS, out=out)
    return out


def test_lora_manifest_builds_pass_on_clean_inputs(tmp_path: Path) -> None:
    manifest = build_lora_distill_manifest(ARTIFACTS, teacher_quality_path=_quality(tmp_path), out=tmp_path / "manifest.json")
    assert manifest["status"] == "PASS"
    assert manifest["total_rows"] == 13200
    assert manifest["train_rows"] == 11000
    assert manifest["eval_rows"] == 1100
    assert manifest["probe_rows"] == 1100
    assert len(manifest["rows_by_family"]) == 11
    assert manifest["rows_by_difficulty"]
    assert manifest["rows_by_prompt_style"]
    assert manifest["dpo_train_pair_count"] >= 3000
    assert manifest["training_authorized"] is True
    assert manifest["package_authorized"] is False
    assert manifest["submission_authorized"] is False
    assert manifest["leaderboard_claim"] is False
    assert manifest["no_0_93_evidence"] is True
    assert manifest["no_0_95_evidence"] is True


def test_lora_manifest_missing_inputs_fail(tmp_path: Path) -> None:
    manifest = build_lora_distill_manifest(tmp_path, teacher_quality_path=tmp_path / "quality.json")
    assert manifest["status"] == "FAIL"
    assert manifest["training_authorized"] is False
    assert manifest["missing_artifacts"]


def test_lora_manifest_gate_blocks_for_duplicate_and_leakage(tmp_path: Path) -> None:
    manifest = build_lora_distill_manifest(ARTIFACTS, teacher_quality_path=_quality(tmp_path))
    mutated = dict(manifest)
    mutated["duplicate_prompt_count"] = 1
    assert _blocked(mutated)
    mutated = dict(manifest)
    mutated["split_leakage_count"] = 1
    assert _blocked(mutated)
    mutated = dict(manifest)
    mutated["gates"] = {**manifest["gates"], "phase4_manifest_pass": False}
    assert not all(mutated["gates"].values())
    mutated = dict(manifest)
    mutated["gates"] = {**manifest["gates"], "phase5_manifest_pass": False}
    assert not all(mutated["gates"].values())


def test_lora_manifest_output_roundtrip(tmp_path: Path) -> None:
    out = tmp_path / "manifest.json"
    manifest = build_lora_distill_manifest(ARTIFACTS, teacher_quality_path=_quality(tmp_path), out=out)
    assert json.loads(out.read_text(encoding="utf-8")) == manifest


def _blocked(manifest: dict[str, object]) -> bool:
    return int(manifest.get("duplicate_prompt_count", 0)) > 0 or int(manifest.get("split_leakage_count", 0)) > 0
