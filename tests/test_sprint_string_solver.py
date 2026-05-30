from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.sprint_solvers import parse_problem, solve_string_problem  # noqa: E402


def test_length_solved() -> None:
    result = solve_string_problem(parse_problem("abc -> 3\nde -> 2\nTarget: fghi -> ?"))

    assert result.status == "solved"
    assert result.prediction == "4"


def test_count_vowels_solved() -> None:
    result = solve_string_problem(parse_problem("aei -> 3\nbcdf -> 0\nTarget: audio -> ?"))

    assert result.status == "solved"
    assert result.prediction == "4"


def test_first_alpha_index_solved() -> None:
    result = solve_string_problem(parse_problem("abc -> 1\nzebra -> 26\nTarget: Dog -> ?"))

    assert result.status == "solved"
    assert result.prediction == "4"


def test_reverse_numeric_string_converts_to_integer() -> None:
    result = solve_string_problem(parse_problem("123 -> 321\n450 -> 054\nTarget: 789 -> ?"))

    assert result.status == "solved"
    assert result.prediction == "987"


def test_reverse_non_numeric_string_rejected_as_invalid_format() -> None:
    result = solve_string_problem(parse_problem("abc -> cba\nTarget: def -> ?"))

    assert result.status == "abstain"
    assert any(candidate.metadata.get("rejection_reason") == "invalid_output_format" for candidate in result.rejected_candidates)


def test_remove_vowels_solved_when_output_numeric_valid() -> None:
    result = solve_string_problem(parse_problem("a1e2 -> 12\ni3o4 -> 34\nTarget: u5a6 -> ?"))

    assert result.status == "solved"
    assert result.prediction == "56"


def test_caesar_shift_requires_sufficient_evidence() -> None:
    result = solve_string_problem(parse_problem("a -> b\nTarget: z -> ?"))

    assert result.status == "abstain"
    assert any(candidate.metadata.get("rejection_reason") == "insufficient_shift_evidence" for candidate in result.rejected_candidates)


def test_caesar_target_unsupported_chars_rejected() -> None:
    result = solve_string_problem(parse_problem("ab -> bc\nTarget: z9 -> ?"))

    assert result.status == "abstain"
    assert any(candidate.metadata.get("rejection_reason") == "unsupported_chars" for candidate in result.rejected_candidates)


def test_string_output_outside_allowed_range_rejected() -> None:
    target = "z" * 1000
    result = solve_string_problem(parse_problem(f"zzzz -> 488\nTarget: {target} -> ?"))

    assert result.status == "abstain"
    assert any(candidate.metadata.get("rejection_reason") == "invalid_output_format" for candidate in result.rejected_candidates)


def test_disagreement_returns_disagreement() -> None:
    result = solve_string_problem(parse_problem("12 -> 2\n34 -> 2\nTarget: abc -> ?"))

    assert result.status == "disagreement"
    assert {candidate.target_prediction for candidate in result.verified_candidates} >= {"0", "3"}
