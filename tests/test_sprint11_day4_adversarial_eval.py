import json

import pytest

from kaggle_anti086.data.schema import validate_rows
from kaggle_anti086.eval.build_day4_adversarial_solver_eval import build_rows, write_eval
from kaggle_anti086.eval.run_solver_eval import run_solver_eval


def test_day4_builder_creates_schema_valid_mixed_eval(tmp_path) -> None:
    out = tmp_path / "day4.jsonl"
    rows = write_eval(out)
    assert out.exists()
    assert len(rows) >= 256
    assert validate_rows(rows)["failure_count"] == 0
    families = {row["family"] for row in rows}
    assert {"bit_manipulation", "symbol_mapping", "char_cipher", "word_cipher", "unit_conversion", "numeric_formula", "gravity_numeric", "roman_numeral"} <= families
    behaviors = {row["metadata"]["expected_solver_behavior"] for row in rows}
    assert behaviors == {"answer", "abstain"}


def test_day4_builder_path_safety_enforced() -> None:
    with pytest.raises(SystemExit):
        write_eval("/kaggle/input/bad/day4.jsonl")


def test_day4_solver_eval_reports_unsafe_answer_metrics(tmp_path) -> None:
    input_path = tmp_path / "day4.jsonl"
    write_eval(input_path)
    report = run_solver_eval(
        input_path,
        tmp_path / "report.json",
        tmp_path / "predictions.jsonl",
        min_behavior_accuracy=0.80,
        min_answerable_exact_match=0.70,
        max_unsafe_answer_rate=0.05,
        min_correct_abstain_rate=0.80,
    )
    assert report["execution_status"] == "PASS"
    assert report["quality_status"] == "PASS"
    assert "unsafe_answer_rate" in report
    assert report["expected_abstain_count"] > 0
    assert report["answerable_count"] > 0
    prediction = json.loads((tmp_path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert "expected_solver_behavior" in prediction
