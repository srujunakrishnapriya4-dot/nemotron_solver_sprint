from __future__ import annotations

import json
from pathlib import Path

from kaggle_anti086.eval.run_solver_eval import run_solver_eval
from kaggle_anti086.solvers.solver_ensemble import SolverEnsemble


def _base_row(**overrides):
    row = {
        "id": "day10_lock_probe",
        "family": "unknown",
        "subfamily": "expected_abstain",
        "prompt": "",
        "answer": "ABSTAIN",
        "rule_id": "day10_lock_rule",
        "leakage_group": "day10_lock_lg",
        "source": "day10_lock_test",
        "solver_name": "locked_solver_stack",
        "verification_status": "verified",
        "difficulty": 0.1,
        "split": "private_like_eval",
        "metadata": {"expected_solver_behavior": "abstain"},
    }
    row.update(overrides)
    return row


def test_blank_prompt_routes_to_explicit_verified_abstain_candidate():
    result = SolverEnsemble().run_all(_base_row())

    assert result.abstained is False
    assert result.candidates
    assert result.candidates[0].answer == "ABSTAIN"
    assert result.candidates[0].source == "blank_prompt_router"
    assert result.candidates[0].verified is True


def test_explicit_abstain_scores_correctly_on_expected_abstain_rows(tmp_path: Path):
    input_path = tmp_path / "abstain_eval.jsonl"
    report_path = tmp_path / "abstain_report.json"
    predictions_path = tmp_path / "abstain_predictions.jsonl"
    input_path.write_text(json.dumps(_base_row()) + "\n", encoding="utf-8")

    report = run_solver_eval(
        input_path,
        report_path,
        predictions_path,
        allow_invalid_rows=True,
        min_behavior_accuracy=1.0,
        max_unsafe_answer_rate=0.0,
    )

    assert report["behavior_accuracy"] == 1.0
    assert report["correct_abstain_rate"] == 1.0
    assert report["unsafe_answer_rate"] == 0.0


def test_private_like_297_numeric_prompt_returns_17():
    row = _base_row(
        id="private_like_297_lock",
        family="numeric_formula",
        subfamily="linear_offset",
        prompt="[private_like #297] 1 -> 5; 2 -> 8; 3 -> 11. Now solve: 5",
        answer="17",
        rule_id="day10_numeric_lock_rule",
        leakage_group="day10_numeric_lock_lg",
        metadata={"expected_solver_behavior": "answer"},
    )

    result = SolverEnsemble().run_all(row)

    assert result.abstained is False
    assert result.candidates[0].answer == "17"
    assert result.candidates[0].source == "numeric_formula_solver"
    assert result.candidates[0].verified is True


def test_exact_fit_numeric_ranking_keeps_linear_candidate_first():
    row = _base_row(
        id="exact_fit_numeric_lock",
        family="numeric_formula",
        subfamily="linear_offset",
        prompt="1 -> 5; 2 -> 8; 3 -> 11. Now solve: 5",
        answer="17",
        rule_id="day10_exact_fit_rule",
        leakage_group="day10_exact_fit_lg",
        metadata={"expected_solver_behavior": "answer"},
    )

    result = SolverEnsemble().run_all(row)

    assert result.abstained is False
    assert result.candidates[0].answer == "17"
    assert result.candidates[0].example_consistency == 1.0
    assert result.candidates[0].risk == "low"


def test_day9_locked_reports_remain_perfect_if_present():
    report_paths = sorted(Path("artifacts/sprint11").glob("day9*report*.json"))
    locked_reports = []
    for path in report_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if any(key in path.name for key in ("private_like", "family_hard", "rule_holdout", "locked", "real_stack")):
            locked_reports.append((path, data))

    for path, report in locked_reports:
        if "exact_match" in report:
            assert report["exact_match"] == 1.0, path
        if "unsafe_answer_rate" in report:
            assert report["unsafe_answer_rate"] == 0.0, path
