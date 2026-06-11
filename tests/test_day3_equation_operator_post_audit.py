from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from kaggle_anti086.training.day3_equation_operator_post_audit import build_equation_operator_post_audit


TMP = Path("artifacts/test_tmp/day3_equation_operator_post_audit")


def _clean() -> Path:
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True, exist_ok=True)
    return TMP


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _repair(root: Path, *, recovered=0, unsafe=0, rows=90, projected=0.7717633928571429) -> Path:
    return _write_json(
        root / "repair.json",
        {
            "schema_version": 1,
            "rows_seen": rows,
            "newly_recovered": recovered,
            "unsafe_answer_count": unsafe,
            "projected_raw_exact_match_if_applied": projected,
        },
    )


def _report(root: Path, *, recovered=0, unsafe=0, status="pre_existing_unresolved_with_evidence") -> dict:
    return build_equation_operator_post_audit(
        repair_report_path=_repair(root, recovered=recovered, unsafe=unsafe),
        numeric_formula_failure_status=status,
    )


def test_zero_recovery_reclassifies_equation_operator_blocked():
    report = _report(_clean())

    assert report["reclassification"]["new_label"] == "BLOCKED_NO_LOCAL_RECOVERY"
    assert report["reclassification"]["equation_operator_actual_recovered_rows"] == 0


def test_zero_recovery_revises_projection_to_1555_over_1792():
    report = _report(_clean())

    projection = report["revised_recovery_projection"]
    assert projection["projected_correct_if_remaining_plausible_solved"] == 1555
    assert projection["projected_accuracy_if_remaining_plausible_solved"] == pytest.approx(1555 / 1792)
    assert projection["projected_accuracy_if_remaining_plausible_solved"] == pytest.approx(0.8677455357142857)


def test_remaining_gap_to_0_93_becomes_112_rows():
    report = _report(_clean())

    assert report["revised_recovery_projection"]["remaining_gap_to_0_93"] == 112
    assert report["revised_recovery_projection"]["can_reach_0_93_from_current_revised_plan"] is False


def test_unsafe_answers_cause_fail():
    report = _report(_clean(), unsafe=1)

    assert report["status"] == "FAIL"
    assert report["decision"]["keep_equation_operator_solver"] == "no"


def test_regression_from_equation_operator_causes_fail():
    report = _report(_clean(), status="regression_from_equation_operator")

    assert report["status"] == "FAIL"
    assert report["decision"]["keep_equation_operator_solver"] == "no"


def test_unknown_numeric_formula_failure_keeps_commit_unauthorized():
    report = _report(_clean(), status="unknown_blocker")

    assert report["status"] == "WARN"
    assert report["decision"]["commit_authorized"] is False


def test_pre_existing_failure_still_keeps_commit_unauthorized():
    report = _report(_clean(), status="pre_existing_unresolved_with_evidence")

    assert report["regression_audit"]["numeric_formula_failure_status"] == "pre_existing_unresolved_with_evidence"
    assert report["decision"]["commit_authorized"] is False


def test_router_ensemble_verifier_decisions_quarantine_when_zero_recovery():
    report = _report(_clean())

    assert report["file_decisions"]["kaggle_anti086/solvers/equation_operator_solver.py"] == "KEEP"
    assert report["file_decisions"]["kaggle_anti086/solvers/router.py"] == "QUARANTINE"
    assert report["file_decisions"]["kaggle_anti086/solvers/solver_ensemble.py"] == "QUARANTINE"
    assert report["file_decisions"]["kaggle_anti086/solvers/verifier.py"] == "QUARANTINE"


def test_scale_package_submission_and_leaderboard_remain_blocked():
    report = _report(_clean())

    assert report["blocked_actions"]["v2a_150"] == "blocked"
    assert report["blocked_actions"]["package"] == "blocked"
    assert report["blocked_actions"]["submission"] == "blocked"
    assert report["blocked_actions"]["leaderboard_claim"] == "blocked"
    assert report["leaderboard_claim"] is False


def test_no_0_93_evidence_remains_true():
    report = _report(_clean())

    assert report["no_0_93_evidence"] is True
    assert report["no_0_95_evidence"] is True


def test_output_deterministic():
    root = _clean()
    report_a = _report(root)
    report_b = _report(root)

    assert json.dumps(report_a, sort_keys=True) == json.dumps(report_b, sort_keys=True)
