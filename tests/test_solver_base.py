from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.core.schemas import AmbiguityReport, Program, ProgramStep, ProgramStepKind, VerificationStatus  # noqa: E402
from nemotron_engine.parsing import canonicalize_prompt  # noqa: E402
from nemotron_engine.solvers.base import SolverResult, build_solver_result, rejected_attempt, verify_candidate_program  # noqa: E402


def _add_one() -> Program:
    return Program(
        "add_one",
        (
            ProgramStep(ProgramStepKind.PARSE, "parse_int", output_key="x"),
            ProgramStep(ProgramStepKind.TRANSFORM, "add_const", {"const": 1}, "y"),
            ProgramStep(ProgramStepKind.FORMAT, "raw", output_key="out"),
        ),
        "out",
    )


def _identity() -> Program:
    return Program(
        "identity",
        (
            ProgramStep(ProgramStepKind.PARSE, "parse_int", output_key="x"),
            ProgramStep(ProgramStepKind.FORMAT, "raw", output_key="out"),
        ),
        "out",
    )


def test_verify_candidate_program_accepts_valid_program() -> None:
    problem = canonicalize_prompt("p", "1 -> 2\n3 -> ?")
    attempt = verify_candidate_program(problem, _add_one(), solver_name="test")

    assert attempt.status is VerificationStatus.PASS
    assert attempt.program is not None
    assert attempt.proof is not None
    assert attempt.proof.is_training_safe is True


def test_verify_candidate_program_rejects_failing_program() -> None:
    problem = canonicalize_prompt("p", "1 -> 2\n3 -> ?")
    attempt = verify_candidate_program(problem, _identity(), solver_name="test")

    assert attempt.status is not VerificationStatus.PASS
    assert attempt.reason


def test_build_solver_result_sets_best_only_for_unique_target_output() -> None:
    problem = canonicalize_prompt("p", "1 -> 2\n3 -> ?")
    accepted = verify_candidate_program(problem, _add_one(), solver_name="test")

    result = build_solver_result(problem, "test", [accepted])
    assert result.best_attempt == accepted


def test_empty_attempts_produce_no_solution_not_silent_success() -> None:
    problem = canonicalize_prompt("p", "1 -> 2\n3 -> ?")
    result = build_solver_result(problem, "test", [])

    assert result.best_attempt is None
    assert result.ambiguity_report.ambiguous is True
    assert result.rejected_attempts[0].reason == "no_solution"


def test_rejected_attempts_always_have_reason() -> None:
    attempt = rejected_attempt("test", "unsupported")

    assert attempt.reason == "unsupported"
    assert attempt.status is VerificationStatus.REJECTED


def test_rejected_attempt_with_empty_reason_rejected() -> None:
    with pytest.raises(ValueError):
        rejected_attempt("test", "")


def test_manual_inconsistent_solver_result_rejected() -> None:
    problem = canonicalize_prompt("p", "1 -> 2\n3 -> ?")
    accepted = verify_candidate_program(problem, _add_one(), solver_name="test")
    rejected = rejected_attempt("test", "bad")

    with pytest.raises(ValueError):
        SolverResult(problem.problem_id, "test", (accepted,), (accepted,), (rejected,), AmbiguityReport(1, ("4",), True, False), accepted)


def test_best_attempt_not_in_accepted_rejected() -> None:
    problem = canonicalize_prompt("p", "1 -> 2\n3 -> ?")
    accepted = verify_candidate_program(problem, _add_one(), solver_name="test")

    with pytest.raises(ValueError):
        SolverResult(problem.problem_id, "test", (accepted,), (), (accepted,), AmbiguityReport(1, ("4",), True, False), accepted)


def test_ambiguous_solver_result_with_best_rejected() -> None:
    problem = canonicalize_prompt("p", "1 -> 2\n3 -> ?")
    accepted = verify_candidate_program(problem, _add_one(), solver_name="test")

    with pytest.raises(ValueError):
        SolverResult(problem.problem_id, "test", (accepted,), (accepted,), (), AmbiguityReport(2, ("4", "5"), False, True), accepted)


def test_accepted_attempt_with_non_pass_status_rejected() -> None:
    problem = canonicalize_prompt("p", "1 -> 2\n3 -> ?")
    rejected = rejected_attempt("test", "bad")

    with pytest.raises(ValueError):
        SolverResult(problem.problem_id, "test", (rejected,), (rejected,), (), AmbiguityReport(0), None)
