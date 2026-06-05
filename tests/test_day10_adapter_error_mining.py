from __future__ import annotations

from kaggle_anti086.training.day10_adapter_error_mining import classify_failure, mine_errors


def test_error_mining_classifies_expected_failure_modes():
    assert classify_failure(family="numeric_formula", pred="17.01", expected="17", expected_behavior="answer") == "wrong_numeric_precision"
    assert classify_failure(family="bit_manipulation", pred="101", expected="0101", expected_behavior="answer") == "wrong_binary_width"
    assert classify_failure(family="word_cipher", pred="BLUE_?", expected="BLUE CAT", expected_behavior="answer") == "cipher_partial"
    assert classify_failure(family="symbol_mapping", pred="unknown", expected="@&", expected_behavior="answer") == "symbol_unknown"
    assert classify_failure(family="unknown", pred="ABSTAIN", expected="17", expected_behavior="answer") == "abstain_overuse"


def test_error_mining_report_counts_failures():
    rows = [
        {"family": "numeric_formula", "expected": "17", "day10_pred": "17", "correct": True},
        {"family": "numeric_formula", "expected": "17", "day10_pred": "17.01", "correct": False},
        {"family": "word_cipher", "expected": "BLUE", "day10_pred": "", "correct": False},
    ]

    report, mined = mine_errors(rows)

    assert report["status"] == "PASS"
    assert report["failure_count"] == 2
    assert report["failure_class_counts"]["wrong_numeric_precision"] == 1
    assert report["failure_class_counts"]["empty_output"] == 1
    assert len(mined) == 2
