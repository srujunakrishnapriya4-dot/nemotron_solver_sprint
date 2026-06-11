from __future__ import annotations

import json
from pathlib import Path

import pytest

from kaggle_anti086.training.day3_sequence_pattern_post_audit import build_sequence_pattern_post_audit


def _write_repair(path: Path, *, recovered: int = 0, unsafe: int = 0) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_by": "DAY3_SEQUENCE_PATTERN_REPAIR_REPORT",
                "status": "FAIL" if unsafe else ("WARN" if recovered == 0 else "PASS"),
                "family": "sequence_pattern",
                "rows_seen": 60,
                "previous_correct": 0,
                "new_correct": recovered,
                "newly_recovered": recovered,
                "remaining_wrong_or_abstain": 60 - recovered,
                "unsafe_answer_count": unsafe,
                "projected_total_correct_if_applied": 1383 + recovered,
                "projected_raw_exact_match_if_applied": (1383 + recovered) / 1792,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def test_zero_recovery_reclassifies_sequence_pattern(tmp_path):
    report = build_sequence_pattern_post_audit(repair_report_path=_write_repair(tmp_path / "repair.json"))

    assert report["status"] == "WARN"
    assert report["measurement"]["newly_recovered"] == 0
    assert report["reclassification"]["new_label"] == "BLOCKED_NO_LOCAL_RECOVERY"


def test_zero_recovery_revises_projected_max_to_custom_numeral_only(tmp_path):
    report = build_sequence_pattern_post_audit(repair_report_path=_write_repair(tmp_path / "repair.json"))
    projection = report["revised_recovery_projection"]

    assert projection["projected_correct_if_remaining_plausible_solved"] == 1495
    assert projection["projected_accuracy_if_remaining_plausible_solved"] == 1495 / 1792
    assert projection["remaining_authorized_families"] == {"custom_numeral": 112}


def test_remaining_gap_to_0_93_becomes_172(tmp_path):
    report = build_sequence_pattern_post_audit(repair_report_path=_write_repair(tmp_path / "repair.json"))

    assert report["revised_recovery_projection"]["remaining_gap_to_0_93"] == 172
    assert report["revised_recovery_projection"]["can_reach_0_93_from_current_revised_plan"] is False


def test_unsafe_answers_cause_fail(tmp_path):
    report = build_sequence_pattern_post_audit(repair_report_path=_write_repair(tmp_path / "repair.json", unsafe=1))

    assert report["status"] == "FAIL"
    assert report["decision"]["keep_sequence_pattern_solver"] == "no"


def test_router_ensemble_or_verifier_changed_causes_warn(tmp_path):
    report = build_sequence_pattern_post_audit(
        repair_report_path=_write_repair(tmp_path / "repair.json"),
        router_changed=True,
    )

    assert report["status"] == "WARN"
    assert report["implementation_summary"]["router_changed"] is True
    assert report["decision"]["keep_sequence_pattern_solver"] == "quarantine"


def test_numeric_formula_failure_is_recorded(tmp_path):
    report = build_sequence_pattern_post_audit(
        repair_report_path=_write_repair(tmp_path / "repair.json"),
        numeric_formula_failure_status="failed",
    )

    failure = report["known_unresolved_failures"][0]
    assert failure["test"] == "tests/test_sprint11_numeric_formula_solver.py::test_numeric_formula_ambiguous_not_low_risk_verified"
    assert failure["classification"] == "pre_existing_unresolved_with_evidence"
    assert failure["status"] == "failed"


def test_v2a_package_submission_and_leaderboard_remain_blocked(tmp_path):
    report = build_sequence_pattern_post_audit(repair_report_path=_write_repair(tmp_path / "repair.json"))

    assert report["blocked_actions"] == {
        "v2a_150": "blocked",
        "package": "blocked",
        "submission": "blocked",
        "leaderboard_claim": "blocked",
    }
    assert report["leaderboard_claim"] is False


def test_no_0_93_evidence_remains_true(tmp_path):
    report = build_sequence_pattern_post_audit(repair_report_path=_write_repair(tmp_path / "repair.json"))

    assert report["no_0_93_evidence"] is True
    assert report["no_0_95_evidence"] is True


def test_output_is_deterministic(tmp_path):
    path = _write_repair(tmp_path / "repair.json")
    first = build_sequence_pattern_post_audit(repair_report_path=path)
    second = build_sequence_pattern_post_audit(repair_report_path=path)

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_commit_authorized_is_false(tmp_path):
    report = build_sequence_pattern_post_audit(repair_report_path=_write_repair(tmp_path / "repair.json"))

    assert report["decision"]["commit_authorized"] is False


def test_missing_repair_report_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_sequence_pattern_post_audit(repair_report_path=tmp_path / "missing.json")
