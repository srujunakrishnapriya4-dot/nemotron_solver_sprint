from __future__ import annotations

import json
from pathlib import Path

from kaggle_anti086.training.day1_teacher_verifier_report import build_teacher_verifier_report


def test_verifier_report_script_creates_json_and_keeps_training_blocked(tmp_path: Path) -> None:
    out = tmp_path / "day1_teacher_verifier_report.json"
    report = build_teacher_verifier_report(tmp_path, per_family=3, seed=123, out=out)

    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded == report
    assert report["status"] == "PASS"
    assert report["generated_rows_checked"] == 33
    assert report["generated_rows_failed"] == 0
    assert report["bad_rows_injected"] == 9
    assert report["bad_rows_accidentally_passed"] == 0
    assert report["trainability_gate_ready_for_phase4"] is True
    assert report["safe_to_train_lora"] is False
    assert report["safe_to_package"] is False
    assert report["safe_to_submit"] is False
    assert report["no_0_93_evidence"] is True
    assert report["no_0_95_evidence"] is True


def test_verifier_report_records_required_rejection_counts(tmp_path: Path) -> None:
    report = build_teacher_verifier_report(tmp_path, per_family=2, seed=321)

    assert report["generated_rows_passed"] == 22
    assert report["bad_rows_rejected"] == report["bad_rows_injected"]
    assert report["duplicate_prompt_count"] >= 1
    assert report["abstain_placeholder_count"] >= 1
    assert report["multiple_box_count"] >= 1
    assert report["text_after_box_count"] >= 1
    assert report["answer_mismatch_count"] >= 1
    assert report["ambiguity_nonzero_count"] >= 1
    assert report["unknown_family_count"] >= 1
    assert report["unsafe_metadata_count"] >= 1
