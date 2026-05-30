from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle_anti086"))

from kaggle_build_parent_calibrated_eval import classify_parent_records, select_candidate_rows  # noqa: E402


def test_parent_zero_calibration_is_invalid() -> None:
    report = classify_parent_records(
        [
            {"id": "a", "family": "bit_manipulation", "exact_match": False, "prompt_hash": "h1"},
            {"id": "b", "family": "unit_conversion", "exact_match": False, "prompt_hash": "h2"},
        ]
    )
    assert report["valid"] is False
    assert report["invalid_reason"] == "parent_zero"
    assert report["parent_exact_match"] == 0.0


def test_parent_calibration_requires_mixed_correct_and_wrong_rows() -> None:
    report = classify_parent_records(
        [
            {"id": "a", "family": "bit_manipulation", "exact_match": True, "prompt_hash": "h1"},
            {"id": "b", "family": "unit_conversion", "exact_match": False, "prompt_hash": "h2"},
        ]
    )
    assert report["valid"] is True
    assert report["parent_correct_ids"] == ["a"]
    assert report["parent_wrong_ids"] == ["b"]


def test_candidate_selection_is_family_balanced_and_direct_answer() -> None:
    rows = [
        {"id": "b1", "prompt": "bits?", "answer": "1010", "family": "bit_manipulation"},
        {"id": "r1", "prompt": "roman?", "answer": "X", "family": "roman_numeral"},
        {"id": "e1", "prompt": "eq?", "answer": "7", "family": "equation_symbolic", "corpus_type": "unsafe_trace"},
    ]
    selected = select_candidate_rows(rows, limit=3)
    assert [row["id"] for row in selected] == ["b1", "r1"]
    assert all(row["corpus_type"] == "direct_answer_only" for row in selected)
