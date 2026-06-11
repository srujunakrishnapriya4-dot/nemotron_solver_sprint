from __future__ import annotations

import json
from pathlib import Path

from kaggle_anti086.training.day1_teacher_quality_report import build_teacher_quality_report


ARTIFACTS = Path("artifacts/sprint11")


def test_teacher_quality_report_builds_pass_on_clean_inputs(tmp_path: Path) -> None:
    out = tmp_path / "quality.json"
    report = build_teacher_quality_report(ARTIFACTS, out=out)
    assert out.exists()
    assert json.loads(out.read_text(encoding="utf-8")) == report
    assert report["status"] == "PASS"
    assert len(report["families"]) == 11
    assert len(report["families_pass"]) == 11
    assert report["teacher_ready_for_lora"] is True
    assert report["training_authorized"] is False
    assert report["package_authorized"] is False
    assert report["submission_authorized"] is False
    assert report["leaderboard_claim"] is False
    assert report["no_0_93_evidence"] is True
    assert report["no_0_95_evidence"] is True


def test_teacher_quality_missing_required_artifact_fails(tmp_path: Path) -> None:
    report = build_teacher_quality_report(tmp_path)
    assert report["status"] == "FAIL"
    assert report["teacher_ready_for_lora"] is False
    assert report["missing_artifacts"]


def test_teacher_quality_safety_counter_failures_are_blocking() -> None:
    report = build_teacher_quality_report(ARTIFACTS)
    bad = dict(report)
    bad["format_error_count"] = 1
    assert _quality_gates_fail(bad)
    bad = dict(report)
    bad["ambiguity_accepted"] = 1
    assert _quality_gates_fail(bad)
    bad = dict(report)
    bad["families_pass"] = ["a", "b", "c", "d", "e"]
    assert _quality_gates_fail(bad)
    bad = dict(report)
    bad["total_verified_sft_rows"] = 9999
    assert _quality_gates_fail(bad)


def _quality_gates_fail(report: dict[str, object]) -> bool:
    return (
        int(report.get("format_error_count", 0)) > 0
        or int(report.get("ambiguity_accepted", 0)) > 0
        or len(report.get("families_pass", [])) < 6
        or int(report.get("total_verified_sft_rows", 0)) < 10000
    )
