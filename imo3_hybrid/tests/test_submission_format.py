from __future__ import annotations

import csv
import json

import pytest

from scripts import build_submission
from src.common.schemas import CandidateAnswer, FinalPrediction
from src.online.submission_formatter import (
    SubmissionFormattingError,
    SubmissionRow,
    format_submission_batch,
    format_submission_row,
    render_submission_csv,
    submission_rows_from_records,
)


def _candidate(answer: str) -> CandidateAnswer:
    return CandidateAnswer(
        answer=answer,
        answer_canonical=answer,
        branch_ids=[f"branch::{answer}"],
        verifier_score=0.82,
        tool_consistency=0.79,
        answer_agreement=0.74,
        branch_novelty=0.31,
        symbolic_check=1.0,
        composite_score=0.83,
        cluster_size=1,
        entropy_penalty=0.0,
    )


def _prediction(problem_id: str, answer: int, *, confidence: float = 0.84) -> FinalPrediction:
    return FinalPrediction(
        problem_id=problem_id,
        final_answer=answer,
        confidence=confidence,
        winning_cluster=_candidate(str(answer)),
        num_branches_generated=12,
        num_branches_survived=4,
        solve_time_sec=1.25,
        method_used="weighted_cluster_final_selector",
    )


def test_submission_rows_are_typed_and_csv_output_is_deterministically_sorted() -> None:
    predictions = [
        _prediction("10", 210),
        _prediction("2", 17),
        _prediction("A1", 99),
    ]

    bundle = format_submission_batch(
        predictions,
        expected_problem_ids=["A1", "10", "2"],
    )
    csv_text = render_submission_csv(bundle.rows)

    assert all(isinstance(row, SubmissionRow) for row in bundle.rows)
    assert [row.id for row in bundle.rows] == ["2", "10", "A1"]
    assert [row.answer for row in bundle.rows] == [17, 210, 99]
    assert csv_text == "id,answer\n2,17\n10,210\nA1,99\n"


def test_submission_row_and_record_validation_rejects_malformed_or_unsafe_values() -> None:
    row = format_submission_row(_prediction("5digit", 99999))
    assert row.answer == 99999

    with pytest.raises(SubmissionFormattingError, match="Duplicate problem ids"):
        submission_rows_from_records(
            [
                {"id": "2", "answer": 17},
                {"id": "2", "answer": 18},
            ]
        )

    with pytest.raises(SubmissionFormattingError, match="must contain 'id' and 'answer'"):
        submission_rows_from_records([{"problem_id": "2", "final_answer": 17}])

    with pytest.raises(SubmissionFormattingError, match="Boolean answers are invalid"):
        submission_rows_from_records([{"id": "2", "answer": True}])

    with pytest.raises(SubmissionFormattingError, match="Unsafe non-integer answer string"):
        submission_rows_from_records([{"id": "2", "answer": "1/2"}])

    with pytest.raises(SubmissionFormattingError, match="invalid CSV delimiter"):
        format_submission_row(_prediction("bad,id", 17))


def test_build_submission_main_writes_strict_sorted_outputs_and_sidecars(tmp_path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    output_csv = tmp_path / "submission.csv"
    debug_csv = tmp_path / "submission.debug.csv"
    summary_json = tmp_path / "submission.summary.json"
    manifest_json = tmp_path / "submission.manifest.json"
    expected_ids_csv = tmp_path / "expected.csv"

    predictions = [
        _prediction("10", 210).model_dump(mode="json"),
        _prediction("2", 17).model_dump(mode="json"),
    ]
    predictions_path.write_text(
        "\n".join(json.dumps(item, sort_keys=True) for item in predictions) + "\n",
        encoding="utf-8",
    )
    expected_ids_csv.write_text("id,problem\n10,p10\n2,p2\n", encoding="utf-8")

    exit_code = build_submission.main(
        [
            "--predictions",
            str(predictions_path),
            "--output-csv",
            str(output_csv),
            "--debug-csv",
            str(debug_csv),
            "--summary-json",
            str(summary_json),
            "--manifest-json",
            str(manifest_json),
            "--expected-ids-csv",
            str(expected_ids_csv),
            "--include-debug-sidecar",
        ]
    )

    assert exit_code == 0
    assert output_csv.read_text(encoding="utf-8") == "id,answer\n2,17\n10,210\n"

    with debug_csv.open("r", encoding="utf-8", newline="") as handle:
        debug_rows = list(csv.DictReader(handle))
    assert [row["problem_id"] for row in debug_rows] == ["2", "10"]
    assert [row["submission_answer"] for row in debug_rows] == ["17", "210"]

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))

    assert summary["row_count"] == 2
    assert summary["prediction_count"] == 2
    assert summary["prebuilt_row_count"] == 0
    assert list(summary["problem_ids"]) == ["2", "10"]
    assert manifest["summary"]["row_count"] == 2
    assert manifest["sources"][0]["record_count"] == 2
    assert manifest["output_csv_path"] == str(output_csv)
    assert manifest["debug_csv_path"] == str(debug_csv)


def test_build_submission_rejects_malformed_prediction_artifacts_explicitly(tmp_path, capsys) -> None:
    bad_predictions = tmp_path / "bad_predictions.jsonl"
    output_csv = tmp_path / "submission.csv"

    bad_predictions.write_text(
        json.dumps({"problem_id": "2", "confidence": 0.5}) + "\n",
        encoding="utf-8",
    )

    exit_code = build_submission.main(
        [
            "--predictions",
            str(bad_predictions),
            "--output-csv",
            str(output_csv),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 2
    assert not output_csv.exists()
    assert "Malformed prediction record" in captured.err
