from __future__ import annotations

from copy import deepcopy

from kaggle_anti086.training.day1x_composed_teacher_report import (
    build_composed_teacher_report,
    build_decision_report,
    build_public_style_prompt_report,
)
from kaggle_anti086.training.day1x_composed_teachers import generate_composed_rows
from kaggle_anti086.training.day1x_prompt_augmentation import augment_rows


def _reports() -> tuple[dict[str, object], dict[str, object], dict[str, object], list[dict[str, object]]]:
    rows = generate_composed_rows(per_family=2, seed=123)
    augmented = augment_rows(rows, ["public_style_minimal"], seed=456, max_per_row=1)
    composed = build_composed_teacher_report(rows, per_family=2)
    public = build_public_style_prompt_report(rows, augmented)
    decision = build_decision_report(composed, public)
    return composed, public, decision, rows


def test_clean_small_reports_pass_and_are_ready() -> None:
    composed, public, decision, _rows = _reports()
    assert composed["status"] == "PASS"
    assert public["status"] == "PASS"
    assert decision["status"] == "PASS"
    assert composed["ready_for_day1x_100k_factory"] is True
    assert decision["ready_for_next_phase"] is True
    assert decision["next_phase"] == "DAY1X_HOLDOUTS_AND_EXPANDED_DATA_FACTORY_100K"


def test_injected_format_error_ambiguity_and_abstain_fail_report() -> None:
    _composed, _public, _decision, rows = _reports()
    bad_format = deepcopy(rows)
    bad_format[0]["target_text"] = "\\boxed{1}\\boxed{2}"
    bad_format[0]["trace"] = bad_format[0]["target_text"]
    assert build_composed_teacher_report(bad_format, per_family=2)["status"] == "FAIL"

    ambiguous = deepcopy(rows)
    ambiguous[0]["ambiguity_count"] = 1
    assert build_composed_teacher_report(ambiguous, per_family=2)["status"] == "FAIL"

    abstain = deepcopy(rows)
    abstain[0]["answer"] = "ABSTAIN"
    abstain[0]["target_text"] = "\\boxed{ABSTAIN}"
    abstain[0]["trace"] = abstain[0]["target_text"]
    assert build_composed_teacher_report(abstain, per_family=2)["status"] == "FAIL"


def test_missing_required_family_does_not_pass() -> None:
    _composed, _public, _decision, rows = _reports()
    missing = [row for row in rows if row["family"] != "composed_gravity_unit"]
    report = build_composed_teacher_report(missing, per_family=2)
    assert report["status"] in {"WARN", "FAIL"}
    assert report["status"] != "PASS"


def test_decision_report_preserves_safety_authorization_flags() -> None:
    composed, public, decision, _rows = _reports()
    assert decision["training_authorized"] is True
    assert decision["package_authorized"] is False
    assert decision["submission_authorized"] is False
    assert decision["leaderboard_claim"] is False
    assert decision["no_0_93_evidence"] is True
    assert decision["no_0_95_evidence"] is True

    failed = dict(composed)
    failed["format_error_count"] = 1
    blocked = build_decision_report(failed, public)
    assert blocked["ready_for_day1x_100k_factory"] is False
