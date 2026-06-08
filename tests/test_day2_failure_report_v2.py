from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from kaggle_anti086.training.day2_failure_report_v2 import build_failure_report_v2, classify_row


TMP = Path("artifacts/test_tmp/day2_failure_report_v2")


def _clean() -> Path:
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True, exist_ok=True)
    return TMP


def _row(**kwargs):
    row = {
        "row_id": "r0",
        "dataset": "core_eval",
        "family": "numeric_formula",
        "subfamily": "linear",
        "expected": "17",
        "gold_answer": "17",
        "raw_output": "17",
        "normalized_answer": "17",
        "extracted_answer": "17",
        "selected_source": "solver",
        "solver_candidate": "17",
        "solver_confidence": 0.9,
        "solver_verification_status": "PASS",
        "failure_reason": None,
        "route_reason": "verified_solver",
        "valid": True,
        "correct": True,
    }
    row.update(kwargs)
    return row


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


def test_expected_abstain_empty_solver_abstain_is_abstain_correct():
    row = _row(
        expected="ABSTAIN",
        gold_answer="ABSTAIN",
        raw_output="",
        normalized_answer="",
        extracted_answer="",
        solver_candidate="",
        correct=False,
        valid=True,
        failure_reason="all_solvers_abstained",
        route_reason="solver_abstained",
    )

    assert classify_row(row) == "abstain_correct"


def test_expected_abstain_non_empty_output_is_abstain_unsafe_answer():
    row = _row(expected="ABSTAIN", gold_answer="ABSTAIN", raw_output="17", normalized_answer="17", extracted_answer="17", correct=False)

    assert classify_row(row) == "abstain_unsafe_answer"


def test_answerable_correct_classifies_answerable_correct():
    assert classify_row(_row(correct=True, valid=True)) == "answerable_correct"


def test_answerable_empty_abstain_classifies_unsupported_answerable():
    row = _row(raw_output="", normalized_answer="", extracted_answer="", solver_candidate="", correct=False, failure_reason="all_solvers_abstained")

    assert classify_row(row) == "unsupported_answerable_or_solver_abstained"


def test_answerable_non_empty_wrong_classifies_wrong_answer():
    row = _row(raw_output="18", normalized_answer="18", extracted_answer="18", solver_candidate="18", correct=False, valid=True)

    assert classify_row(row) == "answerable_wrong_answer"


def test_parse_failure_on_answerable_row_classifies_parse_error():
    row = _row(raw_output="", normalized_answer="", extracted_answer="", solver_candidate="", correct=False, failure_reason="parser_failed")

    assert classify_row(row) == "answerable_parse_error"


def test_ambiguity_failure_on_answerable_row_classifies_ambiguity_bug():
    row = _row(raw_output="", normalized_answer="", extracted_answer="", solver_candidate="", correct=False, failure_reason="multiple fits found")

    assert classify_row(row) == "answerable_ambiguity_bug"


def test_invalid_answerable_row_classifies_invalid_output():
    row = _row(raw_output="???", normalized_answer="???", extracted_answer="???", solver_candidate="???", correct=False, valid=False)

    assert classify_row(row) == "answerable_invalid_output"


def test_priority_actions_use_answerable_wrong_not_abstain_rows():
    root = _clean()
    rows = []
    for idx in range(100):
        rows.append(
            _row(
                row_id=f"abstain-{idx}",
                family="symbol_mapping",
                expected="ABSTAIN",
                gold_answer="ABSTAIN",
                raw_output="",
                normalized_answer="",
                extracted_answer="",
                solver_candidate="",
                correct=False,
                failure_reason="all_solvers_abstained",
                route_reason="solver_abstained",
            )
        )
    for idx in range(5):
        rows.append(_row(row_id=f"wrong-{idx}", family="custom_numeral", raw_output="X", normalized_answer="X", extracted_answer="X", solver_candidate="X", correct=False))

    report = build_failure_report_v2(_inputs(root, rows))

    assert report["overall"]["abstain_correct"] == 100
    assert report["overall"]["answerable_wrong"] == 5
    assert [action["family"] for action in report["priority_actions"]] == ["custom_numeral"]
    assert report["priority_actions"][0]["priority"] == "P2"


def test_correct_abstentions_do_not_create_repair_backlog_items():
    root = _clean()
    rows = [
        _row(
            row_id="a",
            expected="ABSTAIN",
            gold_answer="ABSTAIN",
            raw_output="",
            normalized_answer="",
            extracted_answer="",
            solver_candidate="",
            correct=False,
            failure_reason="all_solvers_abstained",
            route_reason="solver_abstained",
        )
    ]

    report = build_failure_report_v2(_inputs(root, rows))

    assert report["priority_actions"] == []
    assert report["repair_backlog"] == {"p0": [], "p1": [], "p2": [], "p3": []}


def test_report_never_authorizes_v2a_150_or_packaging():
    root = _clean()
    report = build_failure_report_v2(_inputs(root, [_row()]))

    assert report["decision"]["train_v2a_150_authorized"] is False
    assert report["packaging_allowed"] is False
    assert report["submission_allowed"] is False
    assert report["leaderboard_claim"] is False


def test_missing_prediction_file_raises_loudly():
    root = _clean()
    existing = _write_jsonl(root / "core.jsonl", [])
    with pytest.raises(FileNotFoundError, match="missing required solver-only prediction file"):
        build_failure_report_v2(
            {
                "core_eval": existing,
                "family_eval": root / "missing_family.jsonl",
                "rule_holdout": root / "missing_rule.jsonl",
                "anti_leak": root / "missing_anti.jsonl",
            }
        )


def test_class_counts_sum_to_total_rows_and_every_row_classified_once():
    root = _clean()
    rows = [
        _row(row_id="correct"),
        _row(row_id="wrong", raw_output="18", normalized_answer="18", extracted_answer="18", solver_candidate="18", correct=False),
        _row(row_id="invalid", raw_output="???", normalized_answer="???", extracted_answer="???", solver_candidate="???", correct=False, valid=False),
        _row(row_id="abstain", expected="ABSTAIN", gold_answer="ABSTAIN", raw_output="", normalized_answer="", extracted_answer="", solver_candidate="", correct=False, failure_reason="all_solvers_abstained"),
    ]

    report = build_failure_report_v2(_inputs(root, rows))

    assert report["overall"]["rows"] == 4
    assert sum(report["overall"]["class_counts"].values()) == 4
    for dataset_bucket in report["by_dataset"].values():
        assert sum(dataset_bucket["class_counts"].values()) == dataset_bucket["rows"]
    for family_bucket in report["by_family"].values():
        assert sum(family_bucket["class_counts"].values()) == family_bucket["rows"]
