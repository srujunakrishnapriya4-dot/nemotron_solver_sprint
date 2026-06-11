from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from kaggle_anti086.training.day3_equation_operator_quarantine_report import (
    KNOWN_NUMERIC_FAILURE,
    build_equation_operator_quarantine_report,
)


TMP = Path("artifacts/test_tmp/day3_equation_operator_quarantine_report")


def _clean() -> Path:
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True, exist_ok=True)
    return TMP


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _repair(root: Path, *, recovered=0, unsafe=0) -> Path:
    return _write_json(
        root / "repair.json",
        {
            "schema_version": 1,
            "rows_seen": 90,
            "newly_recovered": recovered,
            "unsafe_answer_count": unsafe,
            "projected_raw_exact_match_if_applied": 0.7717633928571429,
        },
    )


def _post(root: Path) -> Path:
    return _write_json(root / "post.json", {"schema_version": 1, "decision": {"keep_equation_operator_solver": "quarantine"}})


def _report(root: Path, **kwargs) -> dict:
    return build_equation_operator_quarantine_report(
        repair_report_path=_repair(root, recovered=kwargs.pop("recovered", 0), unsafe=kwargs.pop("unsafe", 0)),
        post_audit_path=_post(root),
        **kwargs,
    )


def test_quarantine_actions_all_true_by_default():
    report = _report(_clean())

    assert all(report["quarantine_actions"].values())
    assert report["quarantine_actions"]["standalone_solver_kept"] is True
    assert report["quarantine_actions"]["router_reverted"] is True


def test_zero_recovery_is_warn_and_blocked_no_local_recovery():
    report = _report(_clean())

    assert report["status"] == "WARN"
    assert report["reclassification"]["new_label"] == "BLOCKED_NO_LOCAL_RECOVERY"
    assert report["reclassification"]["day3_score_gain"] == 0


def test_unsafe_answers_fail_report():
    report = _report(_clean(), unsafe=1)

    assert report["status"] == "FAIL"
    assert "equation_operator_unsafe_answers_nonzero" in report["failures"]


def test_incomplete_quarantine_cleanup_fails():
    report = _report(_clean(), quarantine_actions={"router_reverted": False})

    assert report["status"] == "FAIL"
    assert "quarantine_cleanup_incomplete" in report["failures"]


def test_known_numeric_formula_failure_recorded():
    report = _report(_clean())

    failure = report["known_unresolved_failures"][0]
    assert failure["test"] == KNOWN_NUMERIC_FAILURE
    assert failure["classification"] == "pre_existing_unresolved_with_evidence"
    assert failure["blocks_full_clean_commit"] is True


def test_integration_commit_blocked_but_day3_can_continue():
    report = _report(_clean())

    assert report["decision"]["safe_to_continue_day3"] is True
    assert report["decision"]["safe_to_commit_solver_integration"] is False
    assert report["decision"]["next_family"] == "sequence_pattern"


def test_blocked_actions_and_no_score_claims():
    report = _report(_clean())

    assert report["blocked_actions"]["v2a_150"] == "blocked"
    assert report["blocked_actions"]["package"] == "blocked"
    assert report["blocked_actions"]["submission"] == "blocked"
    assert report["leaderboard_claim"] is False
    assert report["no_0_93_evidence"] is True
    assert report["no_0_95_evidence"] is True


def test_missing_repair_report_raises_loudly():
    root = _clean()
    with pytest.raises(FileNotFoundError):
        build_equation_operator_quarantine_report(repair_report_path=root / "missing.json", post_audit_path=_post(root))


def test_output_deterministic():
    root = _clean()
    report_a = _report(root)
    report_b = _report(root)

    assert json.dumps(report_a, sort_keys=True) == json.dumps(report_b, sort_keys=True)
