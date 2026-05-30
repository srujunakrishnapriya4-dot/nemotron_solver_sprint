from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.core.schemas import AmbiguityReport, CandidateProgram, ExecutionTrace, Program, ProgramStep, ProgramStepKind, SchemaValidationError  # noqa: E402
from nemotron_engine.programs.ambiguity import build_ambiguity_report  # noqa: E402


def _candidate(output: str | None) -> CandidateProgram:
    program = Program("p", (ProgramStep(ProgramStepKind.PARSE, "parse_int", output_key="x"),), "x")
    trace = ExecutionTrace("1", output) if output is not None else None
    return CandidateProgram(program=program, target_execution=trace)


def test_zero_candidates_ambiguous() -> None:
    report = build_ambiguity_report([])
    assert report.ambiguous is True
    assert report.unique_target_output is False


def test_one_candidate_unique() -> None:
    report = build_ambiguity_report([_candidate("2")])
    assert report.unique_target_output is True
    assert report.ambiguous is False


def test_multiple_candidates_same_target_unique() -> None:
    report = build_ambiguity_report([_candidate("2"), _candidate("2")])
    assert report.unique_target_output is True


def test_multiple_candidates_different_target_outputs_ambiguous() -> None:
    report = build_ambiguity_report([_candidate("2"), _candidate("3")])
    assert report.ambiguous is True


def test_zero_candidate_forged_unique_state_rejected() -> None:
    with pytest.raises(SchemaValidationError):
        AmbiguityReport(candidate_count=0, unique_target_output=True, ambiguous=False)
