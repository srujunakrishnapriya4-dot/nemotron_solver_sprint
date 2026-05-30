from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.programs.leakage import detect_target_leakage  # noqa: E402


def test_no_leakage_for_normal_prompt() -> None:
    report = detect_target_leakage("1 -> 2\n3 -> ?", "4", "derive rule\nAPPLY\n3 -> 4")
    assert report.has_leakage is False


def test_leakage_if_target_answer_in_candidate_before_apply() -> None:
    report = detect_target_leakage("1 -> 2\n3 -> ?", "42", "The answer is 42 before APPLY")
    assert report.has_leakage is True


def test_empty_target_answer_no_false_positive() -> None:
    assert detect_target_leakage("answer is 7", "", "answer is 7").has_leakage is False


def test_one_digit_target_answer_requires_suspicious_context() -> None:
    assert detect_target_leakage("1 -> 2\n3 -> ?", "4", "saw 4 tokens").has_leakage is False
    assert detect_target_leakage("1 -> 2\n3 -> ?", "4", "final answer is 4 before APPLY").has_leakage is True
