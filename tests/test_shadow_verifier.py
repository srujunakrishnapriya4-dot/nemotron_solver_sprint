from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.core.schemas import AmbiguityReport, ExecutionTrace, FormatReport, LeakageReport, Program, ProgramStep, ProgramStepKind, SchemaValidationError  # noqa: E402
from nemotron_engine.parsing import canonicalize_prompt  # noqa: E402
from nemotron_engine.programs.shadow_verifier import shadow_verify_proof  # noqa: E402
from nemotron_engine.programs.verifier import verify_program_on_problem  # noqa: E402


def _proof():
    problem = canonicalize_prompt("p", "1 -> 2\n3 -> ?")
    program = Program(
        "add_one",
        (
            ProgramStep(ProgramStepKind.PARSE, "parse_int", output_key="x"),
            ProgramStep(ProgramStepKind.TRANSFORM, "add_const", {"const": 1}, "y"),
            ProgramStep(ProgramStepKind.FORMAT, "raw", output_key="out"),
        ),
        "out",
    )
    return verify_program_on_problem(problem, program)


def _unsafe_ambiguity(**updates: object) -> AmbiguityReport:
    report = object.__new__(AmbiguityReport)
    data = {
        "candidate_count": 0,
        "target_outputs": (),
        "unique_target_output": True,
        "ambiguous": False,
        "reason": "forged",
    }
    data.update(updates)
    for key, value in data.items():
        object.__setattr__(report, key, value)
    return report


def test_shadow_verifier_sets_pass_on_valid_proof_and_returns_new_object() -> None:
    proof = _proof()
    checked = shadow_verify_proof(proof)

    assert checked is not proof
    assert checked.shadow_verifier_pass is True
    assert checked.is_training_safe is True
    assert proof.shadow_verifier_pass is False


def test_shadow_verifier_fails_bad_evidence() -> None:
    proof = _proof()

    assert shadow_verify_proof(replace(proof, example_executions=(ExecutionTrace("1", "9", "2", False),))).shadow_verifier_pass is False
    assert shadow_verify_proof(replace(proof, leakage_report=LeakageReport(True, ("leak",), ()))).shadow_verifier_pass is False
    assert shadow_verify_proof(replace(proof, ambiguity_report=AmbiguityReport(2, ("2", "3"), False, True))).shadow_verifier_pass is False
    assert shadow_verify_proof(replace(proof, format_report=FormatReport(False, errors=("bad",)))).shadow_verifier_pass is False


def test_shadow_verifier_fails_forged_zero_candidate_empty_examples_and_mismatched_hash() -> None:
    proof = _proof()

    assert shadow_verify_proof(replace(proof, ambiguity_report=_unsafe_ambiguity())).shadow_verifier_pass is False
    assert shadow_verify_proof(replace(proof, example_executions=())).shadow_verifier_pass is False

    forged = replace(proof)
    object.__setattr__(forged, "program_hash", "forged")
    with pytest.raises(SchemaValidationError):
        shadow_verify_proof(forged)
