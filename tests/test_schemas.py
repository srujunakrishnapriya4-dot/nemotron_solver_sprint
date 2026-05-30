from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.core.schemas import (  # noqa: E402
    AmbiguityReport,
    CanonicalProblem,
    DomainKind,
    DomainSignature,
    ExamplePair,
    ExecutableProof,
    ExecutionTrace,
    FormatReport,
    ImmutableProblemRecord,
    LeakageReport,
    ProblemSource,
    Program,
    ProgramStep,
    ProgramStepKind,
    SchemaValidationError,
    SplitName,
    SymbolTable,
    TargetQuery,
    ensure_probability,
)


def _record(**updates: object) -> ImmutableProblemRecord:
    data = {
        "problem_id": "p1",
        "source": ProblemSource.MANUAL_FIXTURE,
        "source_hash": "sha",
        "family_id": "fam",
        "primitive_family_id": "prim",
        "format_family_id": "fmt",
        "prompt_wrapper_id": "wrap",
        "split": SplitName.TRAIN,
        "allowed_for_sft": True,
        "allowed_for_dpo": True,
        "allowed_for_eval": True,
    }
    data.update(updates)
    return ImmutableProblemRecord(**data)


def _problem(**updates: object) -> CanonicalProblem:
    data = {
        "problem_id": "p1",
        "raw_prompt": "1 -> 2\n3 -> ?",
        "examples": (ExamplePair("1", "2"),),
        "target": TargetQuery("3"),
        "input_domain": DomainSignature(DomainKind.INTEGER, "1", "1"),
        "output_domain": DomainSignature(DomainKind.INTEGER, "2", "2"),
        "parse_confidence": 0.96,
        "round_trip_score": 0.98,
    }
    data.update(updates)
    return CanonicalProblem(**data)


def _proof(*, shadow: bool, leak: bool = False, ambiguous: bool = False, example_pass: bool = True) -> ExecutableProof:
    program = Program(
        program_id="prog",
        steps=(ProgramStep(ProgramStepKind.PARSE, "parse_int", output_key="x"),),
        output_key="x",
        program_hash="user-supplied",
    )
    return ExecutableProof(
        problem_id="p1",
        parse_hash="parse",
        program_hash=program.program_hash,
        locked_program=program,
        rejected_programs=(),
        example_executions=(ExecutionTrace("1", "2", "2", example_pass),),
        target_execution=ExecutionTrace("3", "4", None, None),
        ambiguity_report=AmbiguityReport(1, ("4",), unique_target_output=not ambiguous, ambiguous=ambiguous),
        leakage_report=LeakageReport(has_leakage=leak),
        format_report=FormatReport(valid=True, normalized_output="4"),
        primary_verifier_pass=True,
        shadow_verifier_pass=shadow,
    )


def test_valid_immutable_problem_record() -> None:
    assert _record().problem_id == "p1"


def test_record_rejects_empty_problem_id_and_missing_source_hash() -> None:
    with pytest.raises(SchemaValidationError):
        _record(problem_id="")
    with pytest.raises(SchemaValidationError):
        _record(source_hash="")


def test_forbidden_holdout_cannot_allow_sft() -> None:
    with pytest.raises(SchemaValidationError):
        _record(split=SplitName.FORBIDDEN_HOLDOUT, allowed_for_sft=True)


def test_canonical_problem_rejects_no_examples_and_bad_probabilities() -> None:
    with pytest.raises(SchemaValidationError):
        _problem(examples=())
    with pytest.raises(SchemaValidationError):
        _problem(parse_confidence=1.2)
    with pytest.raises(SchemaValidationError):
        _problem(round_trip_score=-0.1)


def test_canonical_problem_rejects_malformed_or_weak_high_confidence_target() -> None:
    with pytest.raises(SchemaValidationError):
        _problem(target=None)
    with pytest.raises(SchemaValidationError):
        _problem(target=TargetQuery("3", certain=False))
    with pytest.raises(SchemaValidationError):
        _problem(target=TargetQuery("UNKNOWN", certain=True))


def test_probability_rejects_string_coercion() -> None:
    with pytest.raises(SchemaValidationError):
        ensure_probability("0.95", "probability")


def test_symbol_table_collision_rejected() -> None:
    with pytest.raises(SchemaValidationError):
        SymbolTable(symbols={"a": "a", "b": "b"}, normalized_symbols={"a": "x", "b": "x"})


def test_program_hash_is_computed_not_trusted() -> None:
    program = Program(
        program_id="prog",
        steps=(ProgramStep(ProgramStepKind.PARSE, "parse_int", output_key="x"),),
        output_key="x",
        program_hash="fake",
    )

    assert program.program_hash != "fake"


def test_ambiguity_report_rejects_forged_zero_candidate_unique_state() -> None:
    with pytest.raises(SchemaValidationError):
        AmbiguityReport(candidate_count=0, unique_target_output=True, ambiguous=False)


def test_executable_proof_training_safe_gate() -> None:
    assert _proof(shadow=False).is_training_safe is False
    assert _proof(shadow=True).is_training_safe is True
    assert _proof(shadow=True, leak=True).is_training_safe is False
    assert _proof(shadow=True, ambiguous=True).is_training_safe is False
    assert _proof(shadow=True, example_pass=False).is_training_safe is False
    assert replace(_proof(shadow=True), example_executions=()).is_training_safe is False


def test_executable_proof_rejects_mismatched_program_hash() -> None:
    with pytest.raises(SchemaValidationError):
        replace(_proof(shadow=False), program_hash="forged")
