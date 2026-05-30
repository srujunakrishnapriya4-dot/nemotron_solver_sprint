from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.core.schemas import AmbiguityReport, FormatReport, LeakageReport, Program, ProgramStep, ProgramStepKind, VerificationStatus  # noqa: E402
from nemotron_engine.parsing import canonicalize_prompt  # noqa: E402
from nemotron_engine.programs.ranker import choose_unique_best, rank_candidate_programs  # noqa: E402
from nemotron_engine.solvers.base import SolverAttempt, verify_candidate_program  # noqa: E402


def _program(name: str, extra: bool = False) -> Program:
    steps = [
        ProgramStep(ProgramStepKind.PARSE, "parse_int", output_key="x"),
        ProgramStep(ProgramStepKind.TRANSFORM, "add_const", {"const": 1}, "y"),
    ]
    if extra:
        steps.append(ProgramStep(ProgramStepKind.TRANSFORM, "add_const", {"const": 0}, "z"))
    steps.append(ProgramStep(ProgramStepKind.FORMAT, "raw", output_key="out"))
    return Program(name, tuple(steps), "out")


def test_verified_simple_program_outranks_more_complex_program() -> None:
    problem = canonicalize_prompt("p", "1 -> 2\n3 -> ?")
    simple = verify_candidate_program(problem, _program("simple"), solver_name="test")
    complex_attempt = verify_candidate_program(problem, _program("complex", extra=True), solver_name="test")

    ranks = rank_candidate_programs([complex_attempt, simple])
    assert ranks[0].attempt_index == 1


def test_candidate_with_leakage_loses() -> None:
    problem = canonicalize_prompt("p", "1 -> 2\n3 -> ?")
    good = verify_candidate_program(problem, _program("good"), solver_name="test")
    assert good.proof is not None
    leaky_proof = replace(good.proof, leakage_report=LeakageReport(True, ("leak",), ()), shadow_verifier_pass=False)
    leaky = SolverAttempt("test", good.program, leaky_proof, VerificationStatus.FAIL, "leakage_detected")

    assert rank_candidate_programs([leaky, good])[0].attempt_index == 1


def test_fail_attempt_with_training_safe_proof_is_not_chosen() -> None:
    problem = canonicalize_prompt("p", "1 -> 2\n3 -> ?")
    good = verify_candidate_program(problem, _program("good"), solver_name="test")
    forged = SolverAttempt("test", good.program, good.proof, VerificationStatus.FAIL, "forged_fail")

    assert choose_unique_best([forged]) is None


def test_ambiguous_and_invalid_candidates_lose() -> None:
    problem = canonicalize_prompt("p", "1 -> 2\n3 -> ?")
    good = verify_candidate_program(problem, _program("good"), solver_name="test")
    assert good.proof is not None
    ambiguous = SolverAttempt(
        "test",
        good.program,
        replace(good.proof, ambiguity_report=AmbiguityReport(2, ("4", "5"), False, True), shadow_verifier_pass=False),
        VerificationStatus.FAIL,
        "ambiguous",
    )
    invalid = SolverAttempt(
        "test",
        good.program,
        replace(good.proof, format_report=FormatReport(False, errors=("bad",)), shadow_verifier_pass=False),
        VerificationStatus.FAIL,
        "invalid",
    )

    assert rank_candidate_programs([ambiguous, invalid, good])[0].attempt_index == 2


def test_choose_unique_best_none_when_candidates_disagree() -> None:
    problem = canonicalize_prompt("p", "1 -> 2\n3 -> ?")
    add_one = verify_candidate_program(problem, _program("add_one"), solver_name="test")
    add_two_program = Program(
        "add_two",
        (
            ProgramStep(ProgramStepKind.PARSE, "parse_int", output_key="x"),
            ProgramStep(ProgramStepKind.TRANSFORM, "add_const", {"const": 2}, "y"),
            ProgramStep(ProgramStepKind.FORMAT, "raw", output_key="out"),
        ),
        "out",
    )
    other_problem = canonicalize_prompt("q", "1 -> 3\n3 -> ?")
    add_two = verify_candidate_program(other_problem, add_two_program, solver_name="test")

    assert choose_unique_best([add_one, add_two]) is None


def test_choose_unique_best_succeeds_when_candidates_agree() -> None:
    problem = canonicalize_prompt("p", "1 -> 2\n3 -> ?")
    first = verify_candidate_program(problem, _program("first"), solver_name="test")
    second = verify_candidate_program(problem, _program("second", extra=True), solver_name="test")

    assert choose_unique_best([first, second]) == first


def test_ranking_does_not_use_expected_target_output() -> None:
    problem = canonicalize_prompt("p", "1 -> 2\n3 -> ?")
    attempt = verify_candidate_program(problem, _program("add_one"), solver_name="test")
    attempt.metadata["expected_target_output"] = "999"

    assert choose_unique_best([attempt]) == attempt
