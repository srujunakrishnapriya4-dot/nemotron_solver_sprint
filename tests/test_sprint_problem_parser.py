from __future__ import annotations

from dataclasses import replace
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.sprint_solvers import SolverCandidate, SolverResult, SprintSolverError, forbidden_metadata_keys, parse_problem  # noqa: E402


def test_parses_arrow_examples_and_target() -> None:
    problem = parse_problem("1 -> 2\n3 -> 4\nTarget: 5 -> ?", problem_id="p1")

    assert problem.problem_id == "p1"
    assert [(item.input_value, item.output_value) for item in problem.examples] == [("1", "2"), ("3", "4")]
    assert problem.target_input == "5"


def test_parses_input_output_examples() -> None:
    problem = parse_problem("Input: aa -> Output: bb\nInput: cc -> Output: ?", problem_id="p2")

    assert problem.examples[0].input_value == "aa"
    assert problem.examples[0].output_value == "bb"
    assert problem.target_input == "cc"


def test_rejects_missing_examples() -> None:
    with pytest.raises(SprintSolverError):
        parse_problem("Target: 5 -> ?")


def test_rejects_missing_target() -> None:
    with pytest.raises(SprintSolverError):
        parse_problem("1 -> 2\n3 -> 4")


def test_rejects_target_with_supplied_output() -> None:
    with pytest.raises(SprintSolverError):
        parse_problem("1 -> 2\nTarget: 3 -> 4")


def test_rejects_prompt_containing_expected_answer_or_correct_answer() -> None:
    with pytest.raises(SprintSolverError):
        parse_problem("1 -> 2\nTarget: 3 -> ?\nexpected_answer=4")
    with pytest.raises(SprintSolverError):
        parse_problem("1 -> 2\nTarget: 3 -> ?\ncorrect answer: 4")
    with pytest.raises(SprintSolverError):
        parse_problem("1 -> 2\nTarget: 3 -> ?\ncorrect answer = 4")
    with pytest.raises(SprintSolverError):
        parse_problem("1 -> 2\nTarget: 3 -> ?\ntarget answer: 4")
    with pytest.raises(SprintSolverError):
        parse_problem("1 -> 2\nTarget: 3 -> ?\ntarget_answer = 4")
    with pytest.raises(SprintSolverError):
        parse_problem("1 -> 2\nTarget: 3 -> ?\nexpected answer: 4")
    with pytest.raises(SprintSolverError):
        parse_problem("1 -> 2\nTarget: 3 -> ?\nanswer is 4")
    with pytest.raises(SprintSolverError):
        parse_problem("1 -> 2\nTarget: 3 -> ?\ngold_answer = 4")
    with pytest.raises(SprintSolverError):
        parse_problem("1 -> 2\nTarget: 3 -> ?\ngold answer: 4")
    with pytest.raises(SprintSolverError):
        parse_problem("1 -> 2\nTarget: 3 -> ?\nlabel: 4")
    with pytest.raises(SprintSolverError):
        parse_problem("1 -> 2\nTarget: 3 -> ?\nis_correct true")
    with pytest.raises(SprintSolverError):
        parse_problem("1 -> 2\nTarget: 3 -> ?\ncorrectness verified")


def test_ignores_row_expected_answer_gold_fields_for_inference() -> None:
    problem = parse_problem(
        {
            "id": "row-1",
            "prompt": "1 -> 2\nTarget: 3 -> ?",
            "expected_answer": "999",
            "gold": "888",
        }
    )

    assert problem.target_input == "3"
    assert problem.examples == (problem.examples[0],)
    assert problem.problem_id == "row-1"
    assert forbidden_metadata_keys({"expected_answer": "999", "gold": "888"}) == ("expected_answer", "gold")
    assert "999" not in problem.parse_hash
    assert "888" not in problem.parse_hash


def test_parsed_hash_forgery_rejected() -> None:
    problem = parse_problem("1 -> 2\nTarget: 3 -> ?")

    with pytest.raises(SprintSolverError):
        replace(problem, parse_hash="forged")


def test_solver_result_solved_with_no_verified_candidates_rejected() -> None:
    with pytest.raises(SprintSolverError):
        SolverResult(status="solved", prediction="4")


def test_solver_result_solved_with_disagreeing_verified_candidates_rejected() -> None:
    first = SolverCandidate("unit", ("2",), "4")
    second = SolverCandidate("unit", ("2",), "5")

    with pytest.raises(SprintSolverError):
        SolverResult(status="solved", prediction="4", verified_candidates=(first, second))


def test_solver_result_abstain_with_verified_candidates_rejected() -> None:
    candidate = SolverCandidate("unit", ("2",), "4")

    with pytest.raises(SprintSolverError):
        SolverResult(status="abstain", prediction=None, verified_candidates=(candidate,))


def test_solver_result_disagreement_with_prediction_rejected() -> None:
    first = SolverCandidate("unit", ("2",), "4")
    second = SolverCandidate("unit", ("2",), "5")

    with pytest.raises(SprintSolverError):
        SolverResult(status="disagreement", prediction="4", verified_candidates=(first, second))
