from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.parsing import canonicalize_prompt  # noqa: E402
from nemotron_engine.solvers.universal_synthesizer import synthesize  # noqa: E402


def test_synthesizer_finds_simple_templates() -> None:
    add = synthesize(canonicalize_prompt("add", "1 -> 2\n2 -> 3\n4 -> ?"))
    bit_not = synthesize(canonicalize_prompt("bit", "00 -> 11\n10 -> 01\n01 -> ?"))
    reverse = synthesize(canonicalize_prompt("rev", "12 -> 21\n34 -> 43\n56 -> ?"))
    digit_sum = synthesize(canonicalize_prompt("sum", "123 -> 6\n111 -> 3\n222 -> ?"))

    assert add.best_attempt.proof.target_execution.output_value == "5"
    assert bit_not.best_attempt.proof.target_execution.output_value == "10"
    assert reverse.best_attempt.proof.target_execution.output_value == "65"
    assert digit_sum.best_attempt.proof.target_execution.output_value == "6"


def test_synthesizer_finds_mul_affine_and_symbolic_binary_op() -> None:
    mul = synthesize(canonicalize_prompt("mul", "2 -> 6\n4 -> 12\n5 -> ?"))
    affine = synthesize(canonicalize_prompt("aff", "1 -> 5\n2 -> 7\n4 -> ?"))
    symbolic = synthesize(canonicalize_prompt("sym", "1 @ 2 -> 3\n3 @ 4 -> 7\n5 @ 6 -> ?"))

    assert mul.best_attempt.proof.target_execution.output_value == "15"
    assert affine.best_attempt.proof.target_execution.output_value == "11"
    assert symbolic.best_attempt.proof.target_execution.output_value == "11"


def test_synthesizer_detects_ambiguity_and_conflicts() -> None:
    result = synthesize(canonicalize_prompt("amb", "1 -> 2\n3 -> ?"))

    assert result.best_attempt is None
    assert result.ambiguity_report.ambiguous is True


def test_max_candidates_budget_enforced_and_recorded() -> None:
    result = synthesize(canonicalize_prompt("p", "1 -> 2\n3 -> ?"), max_candidates=1)

    assert any(attempt.reason == "candidate_budget_exhausted" for attempt in result.rejected_attempts)
    assert any(attempt.metadata.get("skipped_due_budget") is True for attempt in result.rejected_attempts)
    budget = next(attempt for attempt in result.rejected_attempts if attempt.reason == "candidate_budget_exhausted")
    assert budget.metadata["max_candidates"] == 1
    assert budget.metadata["enumerated_candidates"] == 1


def test_max_candidates_rejects_invalid_types() -> None:
    problem = canonicalize_prompt("p", "1 -> 2\n3 -> ?")

    for value in ("10", True, 1.5, None):
        with pytest.raises(ValueError):
            synthesize(problem, max_candidates=value)  # type: ignore[arg-type]


def test_max_candidates_above_hard_cap_clamps_deterministically() -> None:
    result = synthesize(canonicalize_prompt("p", "1 -> 2\n3 -> ?"), max_candidates=999)
    budget = next(attempt for attempt in result.rejected_attempts if attempt.reason == "candidate_budget_exhausted")

    assert budget.metadata["max_candidates"] == 200


def test_synthesizer_does_not_use_expected_target_answer_for_normal_selection() -> None:
    problem = canonicalize_prompt("p", "1 -> 2\n2 -> 3\n4 -> ?")
    result = synthesize(problem, expected_target_output="999", evaluation_mode=False)

    assert result.best_attempt is not None
    assert result.best_attempt.proof.target_execution.output_value == "5"
