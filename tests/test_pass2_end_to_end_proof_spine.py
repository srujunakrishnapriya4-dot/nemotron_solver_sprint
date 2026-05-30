from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.core.schemas import AmbiguityReport, ExecutionTrace, LeakageReport, Program, ProgramStep, ProgramStepKind  # noqa: E402
from nemotron_engine.parsing import canonicalize_prompt, verify_round_trip  # noqa: E402
from nemotron_engine.programs.shadow_verifier import shadow_verify_proof  # noqa: E402
from nemotron_engine.programs.verifier import verify_program_on_problem  # noqa: E402


def _program() -> Program:
    return Program(
        "add_one",
        (
            ProgramStep(ProgramStepKind.PARSE, "parse_int", output_key="x"),
            ProgramStep(ProgramStepKind.TRANSFORM, "add_const", {"const": 1}, "y"),
            ProgramStep(ProgramStepKind.FORMAT, "raw", output_key="out"),
        ),
        "out",
    )


def test_tiny_full_proof_spine_training_safe() -> None:
    problem = canonicalize_prompt("p", "Examples:\n1 -> 2\n3 -> 4\nTarget:\n5 -> ?")

    assert verify_round_trip(problem) is True
    proof = verify_program_on_problem(problem, _program())
    checked = shadow_verify_proof(proof)

    assert checked.is_training_safe is True


def test_bad_row_ambiguous_or_leaky_not_training_safe() -> None:
    problem = canonicalize_prompt("p", "1 -> 2\n3 -> ?")
    proof = shadow_verify_proof(verify_program_on_problem(problem, _program()))
    ambiguous = replace(proof, ambiguity_report=AmbiguityReport(2, ("4", "5"), False, True), shadow_verifier_pass=True)
    leaky = replace(proof, leakage_report=LeakageReport(True, ("leak",), ()), shadow_verifier_pass=True)
    empty_target = replace(proof, target_execution=ExecutionTrace("3", "", None, None), shadow_verifier_pass=True)
    target_error = replace(proof, target_execution=ExecutionTrace("3", None, None, None, error="bad"), shadow_verifier_pass=True)

    assert ambiguous.is_training_safe is False
    assert leaky.is_training_safe is False
    assert empty_target.is_training_safe is False
    assert target_error.is_training_safe is False
