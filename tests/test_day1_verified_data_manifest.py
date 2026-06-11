from __future__ import annotations

from kaggle_anti086.training.day1_verified_data_manifest import build_manifest, manifest_status


def _audits(status: str = "PASS") -> dict[str, dict[str, object]]:
    return {
        "format": {"status": status, "counts": {}, "failures": [], "warnings": []},
        "dedup_leakage": {"status": status, "counts": {}, "failures": [], "warnings": []},
        "family_balance": {"status": status, "counts": {}, "failures": [], "warnings": []},
        "prompt_diversity": {"status": status, "counts": {}, "failures": [], "warnings": []},
        "teacher_verification": {"status": status, "counts": {}, "failures": [], "warnings": []},
    }


def _row() -> dict[str, object]:
    return {
        "family": "unit_conversion",
        "split": "train",
        "prompt_style": "direct",
        "difficulty": "easy",
    }


def test_manifest_status_pass_when_all_audits_pass() -> None:
    manifest, decision = build_manifest(
        seed=123,
        input_reports={},
        output_files={},
        counts={"train_rows": 1, "eval_rows": 1, "probe_rows": 1, "hard_negative_pairs": 1, "rejected_rows": 0},
        train_rows=[_row()],
        eval_rows=[{**_row(), "split": "eval"}],
        probe_rows=[{**_row(), "split": "probe"}],
        rejected_rows=[],
        audits=_audits("PASS"),
    )
    assert manifest_status(manifest, _audits("PASS")) == "PASS"
    assert manifest.gates["phase4_ready_for_phase5"] is True
    assert decision["phase4_ready_for_phase5"] is True
    assert manifest.gates["phase4_ready_for_lora_training"] is False


def test_manifest_status_fail_when_format_audit_fails() -> None:
    audits = _audits("PASS")
    audits["format"] = {"status": "FAIL", "counts": {}, "failures": ["bad"], "warnings": []}
    manifest, decision = build_manifest(
        seed=123,
        input_reports={},
        output_files={},
        counts={"train_rows": 1, "eval_rows": 1, "probe_rows": 1, "hard_negative_pairs": 1, "rejected_rows": 0},
        train_rows=[_row()],
        eval_rows=[{**_row(), "split": "eval"}],
        probe_rows=[{**_row(), "split": "probe"}],
        rejected_rows=[],
        audits=audits,
    )
    assert manifest_status(manifest, audits) == "FAIL"
    assert manifest.gates["phase4_ready_for_phase5"] is False
    assert decision["safe_to_train_lora"] is False
    assert decision["safe_to_package"] is False
    assert decision["safe_to_submit"] is False


def test_decision_report_keeps_lora_training_blocked_even_when_phase5_ready() -> None:
    manifest, decision = build_manifest(
        seed=123,
        input_reports={},
        output_files={},
        counts={"train_rows": 1, "eval_rows": 1, "probe_rows": 1, "hard_negative_pairs": 1, "rejected_rows": 0},
        train_rows=[_row()],
        eval_rows=[{**_row(), "split": "eval"}],
        probe_rows=[{**_row(), "split": "probe"}],
        rejected_rows=[],
        audits=_audits("PASS"),
    )
    assert manifest.gates["phase4_ready_for_phase5"] is True
    assert manifest.gates["phase4_ready_for_lora_training"] is False
    assert manifest.safe_to_train_lora is False
    assert decision["phase4_ready_for_lora_training"] is False
