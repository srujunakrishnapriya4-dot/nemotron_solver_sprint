import json
import subprocess
import sys

from kaggle_anti086.eval.run_solver_eval import run_solver_eval


def _row(row_id: str, prompt: str, answer: str, behavior: str = "answer") -> dict:
    return {
        "id": row_id,
        "family": "roman_numeral",
        "subfamily": "standard_roman",
        "rule_id": f"rule_{row_id}",
        "prompt": prompt,
        "answer": answer,
        "source": "test",
        "solver_name": "solver_ensemble",
        "verification_status": "verified",
        "difficulty": 1,
        "split": "dev",
        "leakage_group": f"lg_{row_id}",
        "metadata": {"expected_solver_behavior": behavior},
    }


def test_behavior_accuracy_counts_correct_abstains(tmp_path) -> None:
    rows = [
        _row("answer", "11 -> XI; 15 -> XV. solve for 38", "XXXVIII"),
        _row("abstain", "11 -> XII; 15 -> XV. solve for 38", "ABSTAIN", "abstain"),
    ]
    input_path = tmp_path / "rows.jsonl"
    input_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report = run_solver_eval(input_path, tmp_path / "report.json", tmp_path / "predictions.jsonl")
    assert report["behavior_accuracy"] == 1.0
    assert report["answerable_exact_match"] == 1.0
    assert report["correct_abstain_rate"] == 1.0


def test_quality_gate_uses_behavior_not_exact_match_only(tmp_path) -> None:
    rows = [_row(f"abstain_{i}", "11 -> XII; 15 -> XV. solve for 38", "ABSTAIN", "abstain") for i in range(3)]
    input_path = tmp_path / "rows.jsonl"
    input_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report = run_solver_eval(input_path, tmp_path / "report.json", tmp_path / "predictions.jsonl", min_behavior_accuracy=1.0, min_correct_abstain_rate=1.0)
    assert report["exact_match"] == 0.0
    assert report["quality_status"] == "PASS"


def test_quality_gate_fails_on_unsafe_answers(tmp_path) -> None:
    row = _row("unsafe", "11 -> XI; 15 -> XV. solve for 38", "ABSTAIN", "abstain")
    input_path = tmp_path / "rows.jsonl"
    input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    report = run_solver_eval(input_path, tmp_path / "report.json", tmp_path / "predictions.jsonl", max_unsafe_answer_rate=0.0)
    assert report["unsafe_answer_count"] == 1
    assert report["quality_status"] == "FAIL"


def test_fail_on_low_answerable_exact_match_exits_nonzero(tmp_path) -> None:
    row = _row("wrong", "11 -> XI; 15 -> XV.", "XXXVIII")
    input_path = tmp_path / "rows.jsonl"
    input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
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
            "--min-answerable-exact-match",
            "0.9",
            "--fail-on-quality-gate",
        ],
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 2
