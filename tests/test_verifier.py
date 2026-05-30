from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.core.schemas import Program, ProgramStep, ProgramStepKind  # noqa: E402
from nemotron_engine.parsing import canonicalize_prompt  # noqa: E402
from nemotron_engine.programs.verifier import verify_program_on_problem  # noqa: E402


def _add_one_program() -> Program:
    return Program(
        "add_one",
        (
            ProgramStep(ProgramStepKind.PARSE, "parse_int", output_key="x"),
            ProgramStep(ProgramStepKind.TRANSFORM, "add_const", {"const": 1}, "y"),
            ProgramStep(ProgramStepKind.FORMAT, "raw", output_key="out"),
        ),
        "out",
    )


def _wrong_program() -> Program:
    return Program(
        "same",
        (
            ProgramStep(ProgramStepKind.PARSE, "parse_int", output_key="x"),
            ProgramStep(ProgramStepKind.FORMAT, "raw", output_key="out"),
        ),
        "out",
    )


def test_valid_program_passes_examples_and_target_exists() -> None:
    problem = canonicalize_prompt("p", "1 -> 2\n3 -> 4\n5 -> ?")
    proof = verify_program_on_problem(problem, _add_one_program())

    assert proof.primary_verifier_pass is True
    assert proof.target_execution is not None
    assert proof.target_execution.output_value == "6"


def test_wrong_program_fails_examples() -> None:
    problem = canonicalize_prompt("p", "1 -> 2\n3 -> ?")
    proof = verify_program_on_problem(problem, _wrong_program())

    assert proof.primary_verifier_pass is False
    assert proof.example_executions[0].passed is False


def test_wrong_program_with_one_failed_example_does_not_primary_pass() -> None:
    problem = canonicalize_prompt("p", "1 -> 1\n2 -> 3\n4 -> ?")
    proof = verify_program_on_problem(problem, _add_one_program())

    assert [trace.passed for trace in proof.example_executions] == [False, True]
    assert proof.primary_verifier_pass is False


def test_verifier_uses_target_expected_output_for_leakage() -> None:
    problem = canonicalize_prompt("p", "1 -> 2\nTarget: 41 -> 42")
    proof = verify_program_on_problem(problem, _add_one_program())

    assert proof.leakage_report.has_leakage is True


def test_primary_verifier_does_not_set_shadow_or_training_safe() -> None:
    problem = canonicalize_prompt("p", "1 -> 2\n3 -> ?")
    proof = verify_program_on_problem(problem, _add_one_program())

    assert proof.shadow_verifier_pass is False
    assert proof.is_training_safe is False
