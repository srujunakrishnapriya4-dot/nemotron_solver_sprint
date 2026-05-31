import json

import pytest

from kaggle_anti086.eval.day5_eval_factory import build_eval_rows
from kaggle_anti086.eval.run_stack_benchmark import run_stack_benchmark


def _write(tmp_path, rows):
    path = tmp_path / "eval.jsonl"
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def test_stack_benchmark_solver_mode_writes_behavior_metrics(tmp_path) -> None:
    eval_path = _write(tmp_path, build_eval_rows("private_like", 128, 1105))
    report = run_stack_benchmark(eval_path, "solver", tmp_path / "report.json", tmp_path / "predictions.jsonl")
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "predictions.jsonl").exists()
    assert report["execution_status"] == "PASS"
    assert "behavior_accuracy" in report
    assert "by_family" in report
    assert "by_subfamily" in report


def test_stack_benchmark_model_plan_does_not_fake_results(tmp_path) -> None:
    eval_path = _write(tmp_path, build_eval_rows("private_like", 32, 1105))
    plan = run_stack_benchmark(eval_path, "model_plan", tmp_path / "plan.json", tmp_path / "placeholder.jsonl")
    assert plan["model_results_faked"] is False
    assert plan["mode"] == "model_plan"


def test_stack_benchmark_unsupported_mode_fails_cleanly(tmp_path) -> None:
    eval_path = _write(tmp_path, build_eval_rows("private_like", 32, 1105))
    with pytest.raises(SystemExit):
        run_stack_benchmark(eval_path, "base", tmp_path / "report.json", tmp_path / "predictions.jsonl")
