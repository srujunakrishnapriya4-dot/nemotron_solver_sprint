import json

import pytest

from kaggle_anti086.eval.run_solver_eval import run_solver_eval
from kaggle_anti086.data.schema import RowValidationError


def _row(row_id: str, family: str, prompt: str, answer: str, subfamily: str = "smoke") -> dict:
    return {
        "id": row_id,
        "family": family,
        "subfamily": subfamily,
        "rule_id": f"{family}_rule",
        "prompt": prompt,
        "answer": answer,
        "source": "day3_test",
        "solver_name": "solver_ensemble",
        "verification_status": "verified",
        "difficulty": 1,
        "split": "dev",
        "leakage_group": row_id,
    }


def test_solver_eval_writes_report_predictions_and_family_metrics(tmp_path) -> None:
    rows = [
        _row("roman", "roman_numeral", "11 -> XI; 15 -> XV. Convert 38", "38 -> XXXVIII", "standard_roman"),
        _row("unit", "unit_conversion", "10 -> 15; 20 -> 25; 30 -> 35. Query 40?", "45", "linear_offset_round_0"),
        _row("word", "word_cipher", '"aaa bbb" -> "cat dog"; encrypted: aaa bbb plaintext: ?', "cat dog", "word_substitution"),
        _row("miss", "roman_numeral", "11 -> XI; 15 -> XV.", "XXXVIII", "standard_roman"),
    ]
    input_path = tmp_path / "rows.jsonl"
    input_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report_path = tmp_path / "report.json"
    predictions_path = tmp_path / "predictions.jsonl"
    report = run_solver_eval(input_path, report_path, predictions_path)
    assert report_path.exists()
    assert predictions_path.exists()
    assert report["row_count"] == 4
    assert report["correct_count"] == 3
    assert report["abstained_count"] == 1
    assert report["by_family"]["roman_numeral"]["count"] == 2
    assert report["by_family"]["roman_numeral"]["correct"] == 1
    prediction_rows = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()]
    assert len(prediction_rows) == 4
    assert prediction_rows[0]["correct"] is True


def test_solver_eval_invalid_row_fails_unless_allowed(tmp_path) -> None:
    input_path = tmp_path / "bad.jsonl"
    input_path.write_text(json.dumps({"id": "bad", "family": "roman_numeral"}) + "\n", encoding="utf-8")
    with pytest.raises(RowValidationError):
        run_solver_eval(input_path, tmp_path / "report.json", tmp_path / "preds.jsonl")
    report = run_solver_eval(input_path, tmp_path / "report2.json", tmp_path / "preds2.jsonl", allow_invalid_rows=True)
    assert report["row_count"] == 1
