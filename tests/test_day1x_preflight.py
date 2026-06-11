from __future__ import annotations

import json
from pathlib import Path

from kaggle_anti086.training.day1x_preflight import build_day1x_preflight_report


def _json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _jsonl(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join('{"x":1}\n' for _ in range(count)), encoding="utf-8")


def _root(tmp_path: Path) -> Path:
    art = tmp_path / "artifacts/sprint11"
    _json(
        art / "day1_closeout_and_artifact_alias_report.json",
        {
            "status": "PASS",
            "day1_complete": True,
            "safety": {"training_authorized": True, "safe_to_train_lora": True, "package_authorized": False, "submission_authorized": False, "leaderboard_claim": False, "no_0_93_evidence": True, "no_0_95_evidence": True},
        },
    )
    _json(art / "day1x_composed_teacher_report.json", {"status": "PASS"})
    _json(art / "day1x_public_style_prompt_report.json", {"status": "PASS"})
    _json(art / "day1x_decision_report_prompt_stage.json", {"status": "PASS", "ready_for_day1x_100k_factory": True})
    _jsonl(art / "phase4_verified_sft_train.jsonl", 10000)
    _jsonl(art / "phase4_verified_sft_eval.jsonl", 1000)
    _jsonl(art / "phase4_verified_eval_probe.jsonl", 1000)
    _jsonl(art / "phase5_dpo_train_pairs.jsonl", 3000)
    return tmp_path


def test_preflight_passes_with_real_existing_artifacts_if_available() -> None:
    report = build_day1x_preflight_report(".")
    assert report["status"] == "PASS"
    assert report["ready_for_expanded_factory"] is True


def test_preflight_fails_closed_when_day1_closeout_missing(tmp_path: Path) -> None:
    report = build_day1x_preflight_report(tmp_path)
    assert report["status"] == "FAIL"
    assert "missing_day1_closeout" in report["blocked_reasons"]


def test_preflight_fails_on_unsafe_flags_and_prompt_stage(tmp_path: Path) -> None:
    root = _root(tmp_path)
    art = root / "artifacts/sprint11"
    closeout = json.loads((art / "day1_closeout_and_artifact_alias_report.json").read_text())
    closeout["safety"]["training_authorized"] = False
    closeout["safety"]["package_authorized"] = True
    _json(art / "day1_closeout_and_artifact_alias_report.json", closeout)
    _json(art / "day1x_decision_report_prompt_stage.json", {"status": "FAIL", "ready_for_day1x_100k_factory": False})
    report = build_day1x_preflight_report(root)
    assert "training_authorized_bad" in report["blocked_reasons"]
    assert "package_authorized_bad" in report["blocked_reasons"]
    assert "prompt_stage_decision_not_pass" in report["blocked_reasons"]


def test_preflight_checks_base_row_counts(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _jsonl(root / "artifacts/sprint11/phase4_verified_sft_train.jsonl", 9999)
    report = build_day1x_preflight_report(root)
    assert report["status"] == "FAIL"
    assert "phase4_train_rows_below_minimum" in report["blocked_reasons"]
