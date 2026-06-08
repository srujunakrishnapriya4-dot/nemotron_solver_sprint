from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from kaggle_anti086.training.day2_abstain_policy_gate import build_abstain_policy_gate


TMP = Path("artifacts/test_tmp/day2_abstain_policy_gate")


def _clean() -> Path:
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True, exist_ok=True)
    return TMP


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _final_report(root: Path, *, raw=0.77, answerable=1.0, abstain_rows=409, abstain_rate=1.0, unsafe=0.0) -> Path:
    return _write_json(
        root / "final.json",
        {
            "schema_version": 1,
            "overall": {
                "raw_exact_match": raw,
                "answerable_accuracy": answerable,
                "abstain_rows": abstain_rows,
                "correct_abstain_rate": abstain_rate,
                "unsafe_abstain_answer_rate": unsafe,
            },
        },
    )


def _gate(root: Path, **kwargs) -> dict:
    final = _final_report(root, **kwargs)
    v2 = _write_json(root / "v2.json", {"schema_version": 2})
    return build_abstain_policy_gate(final_solver_report_path=final, failure_report_v2_path=v2)


def test_raw_exact_match_below_0_93_keeps_no_0_93_evidence_true():
    report = _gate(_clean(), raw=0.77)

    assert report["no_0_93_evidence"] is True
    assert "raw_exact_match_below_0_93" in report["warnings"]


def test_abstain_rows_with_unknown_policy_warn_or_fail():
    report = _gate(_clean(), abstain_rows=10)

    assert report["status"] in {"WARN", "FAIL"}
    assert "final_submission_abstain_policy_unknown" in report["warnings"]
    assert report["decision"]["final_submission_policy_known"] is False


def test_unsafe_abstain_answer_rate_fails():
    report = _gate(_clean(), unsafe=0.25)

    assert report["status"] == "FAIL"
    assert "unsafe_abstain_answer_rate_nonzero" in report["failures"]


def test_answerable_accuracy_one_does_not_authorize_submission():
    report = _gate(_clean(), answerable=1.0, abstain_rate=1.0)

    assert report["metrics"]["answerable_accuracy"] == 1.0
    assert report["decision"]["safe_to_package"] is False
    assert report["decision"]["safe_to_submit"] is False


def test_safe_to_package_and_submit_are_always_false_even_with_high_raw_score():
    report = _gate(_clean(), raw=0.99, answerable=1.0, abstain_rows=0, abstain_rate=None)

    assert report["decision"]["safe_to_package"] is False
    assert report["decision"]["safe_to_submit"] is False
    assert report["leaderboard_claim"] is False


def test_missing_final_solver_report_raises_loudly():
    root = _clean()

    with pytest.raises(FileNotFoundError):
        build_abstain_policy_gate(final_solver_report_path=root / "missing.json")


def test_policy_modes_are_both_present():
    report = _gate(_clean())

    assert set(report["policy_modes"]) == {"abstain_allowed", "abstain_not_allowed"}


def test_output_deterministic():
    root = _clean()
    report_a = _gate(root)
    report_b = _gate(root)

    assert json.dumps(report_a, sort_keys=True) == json.dumps(report_b, sort_keys=True)
