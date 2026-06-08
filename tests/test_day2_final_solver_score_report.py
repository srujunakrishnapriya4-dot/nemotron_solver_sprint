from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from kaggle_anti086.training.day2_final_solver_score_report import build_final_solver_score_report


TMP = Path("artifacts/test_tmp/day2_final_solver_score_report")


def _clean() -> Path:
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True, exist_ok=True)
    return TMP


def _row(**kwargs) -> dict:
    row = {
        "row_id": "r0",
        "dataset": "core_eval",
        "family": "numeric_formula",
        "expected": "17",
        "gold_answer": "17",
        "raw_output": "17",
        "extracted_answer": "17",
        "normalized_answer": "17",
        "correct": True,
        "valid": True,
        "failure_reason": None,
        "route_reason": "verified_solver",
    }
    row.update(kwargs)
    return row


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def _inputs(root: Path, rows: list[dict]) -> dict[str, Path]:
    return {
        "core_eval": _write_jsonl(root / "core.jsonl", rows),
        "family_eval": _write_jsonl(root / "family.jsonl", []),
        "rule_holdout": _write_jsonl(root / "rule.jsonl", []),
        "anti_leak": _write_jsonl(root / "anti.jsonl", []),
    }


def _report(root: Path, rows: list[dict], *, v2: dict | None = None) -> dict:
    return build_final_solver_score_report(
        input_files=_inputs(root, rows),
        failure_report_v2_path=_write_json(root / "v2.json", v2 or {"schema_version": 2, "by_family": {}}),
        eval_ladder_report_path=_write_json(root / "ladder.json", {"decision": {"reason_codes": []}}),
        decision_report_path=_write_json(root / "decision.json", {}),
        hardening_audit_path=_write_json(root / "hardening.json", {"summary": {"blocked_count": 0}}),
    )


def test_answerable_and_abstain_rows_are_counted_separately():
    root = _clean()
    rows = [
        _row(row_id="answerable_correct"),
        _row(row_id="answerable_wrong", raw_output="18", extracted_answer="18", normalized_answer="18", correct=False),
        _row(row_id="abstain_correct", expected="ABSTAIN", gold_answer="ABSTAIN", raw_output="", extracted_answer="", normalized_answer="", correct=False, failure_reason="all_solvers_abstained"),
        _row(row_id="abstain_unsafe", expected="ABSTAIN", gold_answer="ABSTAIN", raw_output="17", extracted_answer="17", normalized_answer="17", correct=False),
    ]

    report = _report(root, rows)

    assert report["overall"]["rows"] == 4
    assert report["overall"]["answerable_rows"] == 2
    assert report["overall"]["abstain_rows"] == 2


def test_raw_exact_match_includes_all_rows():
    root = _clean()
    report = _report(
        root,
        [
            _row(row_id="a"),
            _row(row_id="b", correct=False, raw_output="18", extracted_answer="18", normalized_answer="18"),
            _row(row_id="c", expected="ABSTAIN", gold_answer="ABSTAIN", raw_output="", extracted_answer="", normalized_answer="", correct=False, failure_reason="all_solvers_abstained"),
            _row(row_id="d", expected="ABSTAIN", gold_answer="ABSTAIN", raw_output="x", extracted_answer="x", normalized_answer="x", correct=False),
        ],
    )

    assert report["overall"]["raw_exact_match"] == 0.25


def test_answerable_accuracy_correct_abstain_rate_and_unsafe_rate():
    root = _clean()
    report = _report(
        root,
        [
            _row(row_id="a"),
            _row(row_id="b", correct=False, raw_output="18", extracted_answer="18", normalized_answer="18"),
            _row(row_id="c", expected="ABSTAIN", gold_answer="ABSTAIN", raw_output="", extracted_answer="", normalized_answer="", correct=False, failure_reason="all_solvers_abstained"),
            _row(row_id="d", expected="ABSTAIN", gold_answer="ABSTAIN", raw_output="x", extracted_answer="x", normalized_answer="x", correct=False),
        ],
    )

    assert report["overall"]["answerable_accuracy"] == 0.5
    assert report["overall"]["correct_abstain_rate"] == 0.5
    assert report["overall"]["unsafe_abstain_answer_rate"] == 0.5


def test_leaderboard_risk_includes_unresolved_abstain_policy():
    root = _clean()
    report = _report(root, [_row(expected="ABSTAIN", gold_answer="ABSTAIN", raw_output="", extracted_answer="", normalized_answer="", correct=False, failure_reason="all_solvers_abstained")])

    assert "abstain_policy_unresolved_for_final_leaderboard" in report["overall"]["leaderboard_risk_reasons"]
    assert report["overall"]["leaderboard_risk_score"] >= 0.8


def test_old_v1_repair_families_resolved_if_v2_has_zero_answerable_wrong():
    root = _clean()
    report = _report(
        root,
        [_row()],
        v2={
            "schema_version": 2,
            "by_family": {
                "custom_numeral": {"answerable_wrong": 0},
                "equation_operator": {"answerable_wrong": 0},
                "sequence_pattern": {"answerable_wrong": 0},
                "permutation_sorting": {"answerable_wrong": 0},
            },
        },
    )

    assert set(report["old_v1_family_repair_status"].values()) == {"resolved_by_abstain_policy"}


def test_decisions_remain_blocked():
    root = _clean()
    report = _report(root, [_row()])

    assert report["decision"]["v2a_150_authorized"] is False
    assert report["decision"]["package_authorized"] is False
    assert report["decision"]["submission_authorized"] is False
    assert report["leaderboard_claim"] is False


def test_missing_prediction_file_raises_loudly():
    root = _clean()
    with pytest.raises(FileNotFoundError):
        build_final_solver_score_report(
            input_files={
                "core_eval": root / "missing.jsonl",
                "family_eval": root / "family.jsonl",
                "rule_holdout": root / "rule.jsonl",
                "anti_leak": root / "anti.jsonl",
            }
        )


def test_output_is_deterministic():
    root = _clean()
    rows = [_row(), _row(row_id="b", expected="ABSTAIN", gold_answer="ABSTAIN", raw_output="", extracted_answer="", normalized_answer="", correct=False, failure_reason="all_solvers_abstained")]
    report_a = _report(root, rows)
    report_b = _report(root, rows)

    assert json.dumps(report_a, sort_keys=True) == json.dumps(report_b, sort_keys=True)
