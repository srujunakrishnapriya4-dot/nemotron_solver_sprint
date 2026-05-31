import json

from kaggle_anti086.eval.day5_eval_factory import build_eval_rows
from kaggle_anti086.eval.run_stack_benchmark import run_stack_benchmark


def _write(tmp_path, rows):
    path = tmp_path / "eval.jsonl"
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def test_model_plan_writes_exact_base_parent_command_files(tmp_path) -> None:
    eval_path = _write(tmp_path, build_eval_rows("private_like", 32, 1105))
    plan = run_stack_benchmark(eval_path, "model_plan", tmp_path / "plan.json", tmp_path / "placeholder.jsonl")
    base_path = tmp_path / "day5_kaggle_base_eval_commands.txt"
    parent_path = tmp_path / "day5_kaggle_parent_eval_commands.txt"
    assert base_path.exists()
    assert parent_path.exists()
    assert "kaggle_eval_anti086_vllm.py" in base_path.read_text(encoding="utf-8")
    assert "kaggle_eval_anti086_vllm.py" in parent_path.read_text(encoding="utf-8")
    assert "--eval-file" not in base_path.read_text(encoding="utf-8")
    assert "--eval-file" not in parent_path.read_text(encoding="utf-8")
    assert plan["model_results_faked"] is False
    assert plan["command_files"]["base"].endswith("day5_kaggle_base_eval_commands.txt")
