from __future__ import annotations

from dataclasses import replace
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.sprint_solvers import (  # noqa: E402
    PredictionRunnerError,
    run_predictions,
    validate_prediction_run_report,
)


ARTIFACTS = {
    "solver_attempts.jsonl",
    "predictions.jsonl",
    "disagreements.jsonl",
    "abstentions.jsonl",
    "family_metrics.json",
}


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_runner_writes_all_five_artifacts(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "out"
    write_jsonl(input_path, [{"id": "p1", "prompt": "add\n1 -> 2\n2 -> 3\nTarget: 3 -> ?"}])

    run_predictions(input_path, output_dir)

    assert {path.name for path in output_dir.iterdir()} == ARTIFACTS


def test_symbolic_agreed_answer_produces_prediction(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "out"
    write_jsonl(input_path, [{"id": "p1", "prompt": "add\n1 -> 2\n2 -> 3\nTarget: 3 -> ?"}])

    report = run_predictions(input_path, output_dir)
    predictions = read_jsonl(output_dir / "predictions.jsonl")

    assert report.prediction_count == 1
    assert predictions[0]["answer"] == 4
    assert predictions[0]["source"] == "symbolic"


def test_symbolic_disagreement_writes_disagreement_and_no_prediction(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "out"
    write_jsonl(input_path, [{"id": "p1", "prompt": "bit\n00 -> 00\n11 -> 11\nTarget: 01 -> ?"}])

    report = run_predictions(input_path, output_dir)

    assert report.disagreement_count == 1
    assert read_jsonl(output_dir / "disagreements.jsonl")[0]["problem_id"] == "p1"
    assert read_jsonl(output_dir / "predictions.jsonl") == []


def test_parser_failure_writes_abstention(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "out"
    write_jsonl(input_path, [{"id": "p1", "prompt": "1 -> 2"}])

    report = run_predictions(input_path, output_dir)
    abstentions = read_jsonl(output_dir / "abstentions.jsonl")

    assert report.abstention_count == 1
    assert abstentions[0]["parser_failed"] is True
    assert abstentions[0]["reason"] == "parser_failed"


def test_forbidden_prompt_leakage_abstains_without_fallback(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "out"
    calls: list[dict[str, object]] = []
    write_jsonl(input_path, [{"id": "p1", "prompt": "1 -> 2\nTarget: 3 -> ?\nexpected_answer=4"}])

    def fallback(**kwargs):
        calls.append(kwargs)
        return 4

    run_predictions(input_path, output_dir, allow_model_fallback=True, model_fallback=fallback)
    abstentions = read_jsonl(output_dir / "abstentions.jsonl")

    assert calls == []
    assert abstentions[0]["reason"] == "parser_forbidden_leakage"
    assert abstentions[0]["fallback_allowed"] is False


def test_normal_parser_failure_may_still_call_fallback(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "out"
    calls: list[dict[str, object]] = []
    write_jsonl(input_path, [{"id": "p1", "prompt": "1 -> 2"}])

    def fallback(**kwargs):
        calls.append(kwargs)
        return 9

    run_predictions(input_path, output_dir, allow_model_fallback=True, model_fallback=fallback)
    predictions = read_jsonl(output_dir / "predictions.jsonl")

    assert len(calls) == 1
    assert calls[0]["raw_prompt"] == "1 -> 2"
    assert predictions[0]["answer"] == 9


def test_fallback_called_only_after_abstention_no_symbolic_solution(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "out"
    calls: list[dict[str, object]] = []
    write_jsonl(input_path, [{"id": "p1", "prompt": "abc -> 1\nTarget: def -> ?"}])

    def fallback(**kwargs):
        calls.append(kwargs)
        return 7

    report = run_predictions(input_path, output_dir, allow_model_fallback=True, model_fallback=fallback)
    predictions = read_jsonl(output_dir / "predictions.jsonl")

    assert len(calls) == 1
    assert report.fallback_prediction_count == 1
    assert predictions[0]["answer"] == 7
    assert predictions[0]["source"] == "model_fallback"


def test_fallback_not_called_after_symbolic_disagreement(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "out"
    called = False
    write_jsonl(input_path, [{"id": "p1", "prompt": "bit\n00 -> 00\n11 -> 11\nTarget: 01 -> ?"}])

    def fallback(**kwargs):
        nonlocal called
        called = True
        return 7

    run_predictions(input_path, output_dir, allow_model_fallback=True, model_fallback=fallback)

    assert called is False
    assert read_jsonl(output_dir / "predictions.jsonl") == []


def test_fallback_does_not_receive_eval_only_fields(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "out"
    seen: dict[str, object] = {}
    write_jsonl(
        input_path,
        [
            {
                "id": "p1",
                "prompt": "abc -> 1\nTarget: def -> ?",
                "expected_answer": 99,
                "gold": 98,
                "target_answer": 97,
                "correct_answer": 96,
            }
        ],
    )

    def fallback(**kwargs):
        seen.update(kwargs)
        return 7

    run_predictions(input_path, output_dir, allow_model_fallback=True, model_fallback=fallback)

    assert set(seen) == {"problem_id", "raw_prompt", "parsed_problem"}
    assert "expected_answer" not in seen
    assert "gold" not in seen
    assert "target_answer" not in seen
    assert "correct_answer" not in seen


def test_row_level_expected_gold_ignored_for_symbolic_inference(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "out"
    write_jsonl(input_path, [{"id": "p1", "prompt": "1 -> 2\n2 -> 3\nTarget: 3 -> ?", "expected_answer": 999, "gold": 888}])

    run_predictions(input_path, output_dir)
    predictions = read_jsonl(output_dir / "predictions.jsonl")

    assert predictions[0]["answer"] == 4


def test_invalid_fallback_output_rejected_and_abstention_recorded(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "out"
    write_jsonl(input_path, [{"id": "p1", "prompt": "abc -> 1\nTarget: def -> ?"}])

    report = run_predictions(input_path, output_dir, allow_model_fallback=True, model_fallback=lambda **_: "nope")
    abstentions = read_jsonl(output_dir / "abstentions.jsonl")

    assert report.invalid_prediction_count == 1
    assert report.prediction_count == 0
    assert abstentions[0]["reason"] == "invalid_fallback_prediction"


def test_predictions_pass_existing_answer_validation(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "out"
    write_jsonl(input_path, [{"id": "p1", "prompt": "abc -> 1\nTarget: def -> ?"}])

    report = run_predictions(input_path, output_dir, allow_model_fallback=True, model_fallback=lambda **_: 100000)

    assert report.prediction_count == 0
    assert report.invalid_prediction_count == 1


def test_family_metrics_counts_match_artifacts(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "out"
    write_jsonl(
        input_path,
        [
            {"id": "p1", "prompt": "add\n1 -> 2\n2 -> 3\nTarget: 3 -> ?"},
            {"id": "p2", "prompt": "1 -> 2"},
            {"id": "p3", "prompt": "bit\n00 -> 00\n11 -> 11\nTarget: 01 -> ?"},
        ],
    )

    report = run_predictions(input_path, output_dir)
    metrics = json.loads((output_dir / "family_metrics.json").read_text(encoding="utf-8"))

    assert metrics["prediction_count"] == len(read_jsonl(output_dir / "predictions.jsonl")) == report.prediction_count
    assert metrics["abstention_count"] == len(read_jsonl(output_dir / "abstentions.jsonl")) == report.abstention_count
    assert metrics["disagreement_count"] == len(read_jsonl(output_dir / "disagreements.jsonl")) == report.disagreement_count
    validate_prediction_run_report(report, output_dir)


def test_prediction_run_report_forged_hash_rejected(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "out"
    write_jsonl(input_path, [{"id": "p1", "prompt": "1 -> 1\n2 -> 2\nTarget: 3 -> ?"}])
    report = run_predictions(input_path, output_dir)

    with pytest.raises(PredictionRunnerError):
        replace(report, report_hash="forged")


def test_prediction_run_report_inconsistent_total_rows_rejected(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "out"
    write_jsonl(input_path, [{"id": "p1", "prompt": "1 -> 1\n2 -> 2\nTarget: 3 -> ?"}])
    report = run_predictions(input_path, output_dir)

    with pytest.raises(PredictionRunnerError):
        replace(report, total_rows=99)


def test_tampered_family_metrics_count_rejected(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "out"
    write_jsonl(input_path, [{"id": "p1", "prompt": "1 -> 1\n2 -> 2\nTarget: 3 -> ?"}])
    report = run_predictions(input_path, output_dir)
    metrics_path = output_dir / "family_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["prediction_count"] = 99
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    with pytest.raises(PredictionRunnerError):
        validate_prediction_run_report(report, output_dir)


def test_malformed_prediction_row_rejected(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "out"
    write_jsonl(input_path, [{"id": "p1", "prompt": "1 -> 1\n2 -> 2\nTarget: 3 -> ?"}])
    report = run_predictions(input_path, output_dir)
    (output_dir / "predictions.jsonl").write_text("{bad\n", encoding="utf-8")

    with pytest.raises(PredictionRunnerError):
        validate_prediction_run_report(report, output_dir)


def test_non_object_jsonl_row_rejected(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "out"
    write_jsonl(input_path, [{"id": "p1", "prompt": "1 -> 1\n2 -> 2\nTarget: 3 -> ?"}])
    report = run_predictions(input_path, output_dir)
    (output_dir / "predictions.jsonl").write_text("[]\n", encoding="utf-8")

    with pytest.raises(PredictionRunnerError):
        validate_prediction_run_report(report, output_dir)


def test_prediction_for_disagreement_problem_rejected(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "out"
    write_jsonl(input_path, [{"id": "p1", "prompt": "bit\n00 -> 00\n11 -> 11\nTarget: 01 -> ?"}])
    report = run_predictions(input_path, output_dir)
    write_jsonl(output_dir / "predictions.jsonl", [{"id": "p1", "answer": 1, "source": "symbolic", "solver": "bit_solver", "family": "bit", "prediction_hash": "x"}])

    with pytest.raises(PredictionRunnerError):
        validate_prediction_run_report(report, output_dir)


def test_output_files_stay_inside_output_dir(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "out"
    write_jsonl(input_path, [{"id": "p1", "prompt": "1 -> 1\n2 -> 2\nTarget: 3 -> ?"}])

    run_predictions(input_path, output_dir)

    assert all(path.resolve().parent == output_dir.resolve() for path in output_dir.iterdir())


def test_dsl_synthesizer_runs_last(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "out"
    write_jsonl(input_path, [{"id": "p1", "prompt": "bit\n0101 -> 1010\n1111 -> 0000\nTarget: 0011 -> ?"}])

    run_predictions(input_path, output_dir)
    attempts = read_jsonl(output_dir / "solver_attempts.jsonl")

    assert attempts[-1]["solver_name"] == "dsl_synthesizer"


def test_no_model_api_or_kaggle_behavior(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "out"
    write_jsonl(input_path, [{"id": "p1", "prompt": "1 -> 1\n2 -> 2\nTarget: 3 -> ?"}])
    original_import = __import__

    def blocked_import(name: str, *args, **kwargs):
        if name in {"openai", "kaggle", "torch", "transformers"}:
            raise AssertionError(f"forbidden import attempted: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked_import)

    run_predictions(input_path, output_dir)
