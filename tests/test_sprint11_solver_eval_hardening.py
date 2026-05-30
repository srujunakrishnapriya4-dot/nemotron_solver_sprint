import json
import subprocess
import sys

import pytest

from kaggle_anti086.eval.run_solver_eval import run_solver_eval


def _row(prompt: str = "11 -> XI; 15 -> XV. Convert 38", answer: str = "XXXVIII") -> dict:
    return {
        "id": "eval_harden",
        "family": "roman_numeral",
        "subfamily": "standard_roman",
        "rule_id": "roman",
        "prompt": prompt,
        "answer": answer,
        "source": "test",
        "solver_name": "solver_ensemble",
        "verification_status": "verified",
        "difficulty": 1,
        "split": "dev",
        "leakage_group": "eval_harden",
    }


def test_quality_status_fail_when_threshold_not_met(tmp_path) -> None:
    input_path = tmp_path / "rows.jsonl"
    input_path.write_text(json.dumps(_row(prompt="11 -> XI; 15 -> XV.", answer="XXXVIII")) + "\n", encoding="utf-8")
    report = run_solver_eval(input_path, tmp_path / "report.json", tmp_path / "predictions.jsonl", min_exact_match=0.8, min_attempt_rate=0.8)
    assert report["execution_status"] == "PASS"
    assert report["quality_status"] == "FAIL"


def test_fail_on_quality_gate_exits_nonzero(tmp_path) -> None:
    input_path = tmp_path / "rows.jsonl"
    input_path.write_text(json.dumps(_row(prompt="11 -> XI; 15 -> XV.", answer="XXXVIII")) + "\n", encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "kaggle_anti086.eval.run_solver_eval",
            "--input",
            str(input_path),
            "--out-report",
            str(tmp_path / "report.json"),
            "--out-predictions",
            str(tmp_path / "predictions.jsonl"),
            "--min-exact-match",
            "0.8",
            "--fail-on-quality-gate",
        ],
        cwd=".",
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 2


def test_eval_rejects_kaggle_input_output_path(tmp_path) -> None:
    input_path = tmp_path / "rows.jsonl"
    input_path.write_text(json.dumps(_row()) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        run_solver_eval(input_path, "/kaggle/input/bad/report.json", tmp_path / "predictions.jsonl")


def test_predictions_include_normalized_fields_and_family_metrics(tmp_path) -> None:
    input_path = tmp_path / "rows.jsonl"
    input_path.write_text(json.dumps(_row(answer="38 -> XXXVIII")) + "\n", encoding="utf-8")
    report = run_solver_eval(input_path, tmp_path / "report.json", tmp_path / "predictions.jsonl")
    assert report["by_family"]["roman_numeral"]["correct"] == 1
    prediction = json.loads((tmp_path / "predictions.jsonl").read_text(encoding="utf-8"))
    assert prediction["normalized_expected"] == "XXXVIII"
    assert prediction["normalized_prediction"] == "XXXVIII"
