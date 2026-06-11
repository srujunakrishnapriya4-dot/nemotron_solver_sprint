from __future__ import annotations

import json
from pathlib import Path

from kaggle_anti086.training.day1_lora_distill_manifest import build_lora_distill_manifest
from kaggle_anti086.training.day1_phase6_gate import build_phase6_decision_report, evaluate_training_authorization
from kaggle_anti086.training.day1_teacher_quality_report import build_teacher_quality_report


ARTIFACTS = Path("artifacts/sprint11")


def _teacher_quality() -> dict[str, object]:
    return {
        "status": "PASS",
        "teacher_ready_for_lora": True,
        "total_train_rows": 11000,
        "total_eval_rows": 1100,
        "total_dpo_train_pairs": 5500,
        "families_pass": ["a", "b", "c", "d", "e", "f"],
        "format_error_count": 0,
        "verification_fail_count": 0,
        "ambiguity_accepted": 0,
        "abstain_placeholders_accepted": 0,
        "duplicate_prompt_count": 0,
        "split_leakage_count": 0,
        "package_authorized": False,
        "submission_authorized": False,
        "leaderboard_claim": False,
        "no_0_93_evidence": True,
        "no_0_95_evidence": True,
    }


def _manifest() -> dict[str, object]:
    return {
        "status": "PASS",
        "training_authorized": True,
        "dpo_train_pair_count": 5500,
        "package_authorized": False,
        "submission_authorized": False,
        "leaderboard_claim": False,
        "no_0_93_evidence": True,
        "no_0_95_evidence": True,
    }


def test_evaluate_training_authorization_authorizes_clean_manifest() -> None:
    result = evaluate_training_authorization(_teacher_quality(), _manifest())
    assert result["training_authorized"] is True
    assert result["safe_to_train_lora"] is True
    assert result["blocked_reasons"] == []


def test_evaluate_training_authorization_blocks_required_failures() -> None:
    cases = [
        ("teacher_quality_fail", {**_teacher_quality(), "status": "FAIL"}, _manifest()),
        ("lora_manifest_fail", _teacher_quality(), {**_manifest(), "status": "FAIL"}),
        ("total_train_rows_below_10000", {**_teacher_quality(), "total_train_rows": 9999}, _manifest()),
        ("total_eval_rows_below_1000", {**_teacher_quality(), "total_eval_rows": 999}, _manifest()),
        ("dpo_train_pairs_below_3000", _teacher_quality(), {**_manifest(), "dpo_train_pair_count": 2999}),
        ("teacher_quality_package_authorized_true", {**_teacher_quality(), "package_authorized": True}, _manifest()),
        ("teacher_quality_submission_authorized_true", {**_teacher_quality(), "submission_authorized": True}, _manifest()),
        ("teacher_quality_leaderboard_claim_true", {**_teacher_quality(), "leaderboard_claim": True}, _manifest()),
        ("teacher_quality_no_0_93_evidence_false", {**_teacher_quality(), "no_0_93_evidence": False}, _manifest()),
        ("teacher_quality_no_0_95_evidence_false", {**_teacher_quality(), "no_0_95_evidence": False}, _manifest()),
    ]
    for reason, quality, manifest in cases:
        result = evaluate_training_authorization(quality, manifest)
        assert result["training_authorized"] is False
        assert reason in result["blocked_reasons"]


def test_phase6_decision_report_safe_to_train_only_when_authorized(tmp_path: Path) -> None:
    quality_path = tmp_path / "quality.json"
    manifest_path = tmp_path / "manifest.json"
    out = tmp_path / "decision.json"
    build_teacher_quality_report(ARTIFACTS, out=quality_path)
    build_lora_distill_manifest(ARTIFACTS, teacher_quality_path=quality_path, out=manifest_path)
    report = build_phase6_decision_report(
        artifacts_dir=ARTIFACTS,
        teacher_quality_path=quality_path,
        lora_manifest_path=manifest_path,
        out=out,
    )
    assert json.loads(out.read_text(encoding="utf-8")) == report
    assert report["status"] == "PASS"
    assert report["training_authorized"] is True
    assert report["safe_to_train_lora"] is True
    assert report["safe_to_package"] is False
    assert report["safe_to_submit"] is False
    assert report["package_authorized"] is False
    assert report["submission_authorized"] is False
