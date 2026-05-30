from __future__ import annotations

from dataclasses import replace
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.sprint_solvers import (  # noqa: E402
    FailureMiningError,
    mine_failures,
    run_predictions,
    validate_failure_mining_report,
)


REQUIRED = (
    "solver_attempts.jsonl",
    "predictions.jsonl",
    "disagreements.jsonl",
    "abstentions.jsonl",
    "family_metrics.json",
)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def make_run(tmp_path: Path) -> tuple[Path, Path]:
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "out"
    write_jsonl(
        input_path,
        [
            {"id": "p1", "prompt": "1 -> 2\n2 -> 3\nTarget: 3 -> ?", "expected_answer": 4},
            {"id": "p2", "prompt": "1 -> 2\n2 -> 3\nTarget: 3 -> ?", "expected_answer": 5},
            {"id": "p3", "prompt": "1 -> 2"},
            {"id": "p4", "prompt": "bit\n00 -> 00\n11 -> 11\nTarget: 01 -> ?"},
            {"id": "p5", "prompt": "abc -> cba\nTarget: def -> ?", "gold": 1},
        ],
    )
    run_predictions(input_path, output_dir)
    return input_path, output_dir


def test_reads_runner_artifacts_and_builds_deterministic_report(tmp_path: Path) -> None:
    _, output_dir = make_run(tmp_path)

    first = mine_failures(output_dir)
    second = mine_failures(output_dir)

    assert first.report_hash == second.report_hash
    assert first.total_predictions == 2
    assert first.total_abstentions >= 1


def test_missing_artifacts_rejected(tmp_path: Path) -> None:
    with pytest.raises(FailureMiningError):
        mine_failures(tmp_path)


def test_malformed_jsonl_rejected(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    for name in REQUIRED:
        (output_dir / name).write_text("{}\n" if name.endswith(".json") else "", encoding="utf-8")
    (output_dir / "solver_attempts.jsonl").write_text("{bad\n", encoding="utf-8")

    with pytest.raises(FailureMiningError):
        mine_failures(output_dir)


def test_non_object_jsonl_row_rejected(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    for name in REQUIRED:
        (output_dir / name).write_text("{}\n" if name.endswith(".json") else "", encoding="utf-8")
    (output_dir / "predictions.jsonl").write_text("[]\n", encoding="utf-8")

    with pytest.raises(FailureMiningError):
        mine_failures(output_dir)


def test_computes_top_abstention_reasons(tmp_path: Path) -> None:
    _, output_dir = make_run(tmp_path)

    report = mine_failures(output_dir)

    reasons = dict(report.top_abstention_reasons)
    assert reasons["parser_failed"] == 1
    assert reasons["symbolic_disagreement"] == 1


def test_computes_top_rejected_reasons(tmp_path: Path) -> None:
    _, output_dir = make_run(tmp_path)

    report = mine_failures(output_dir)

    reasons = dict(report.top_rejected_reasons)
    assert "invalid_output_format" in reasons or "failed_example_or_target_validation" in reasons


def test_optional_labeled_accuracy_works(tmp_path: Path) -> None:
    input_path, output_dir = make_run(tmp_path)

    report = mine_failures(output_dir, labeled_input_path=input_path)

    assert report.optional_accuracy is not None
    assert report.optional_accuracy["labeled_count"] == 3
    assert report.optional_accuracy["correct_count"] == 1
    assert report.optional_accuracy["incorrect_count"] == 1


def test_wrong_prediction_ids_reported(tmp_path: Path) -> None:
    input_path, output_dir = make_run(tmp_path)

    report = mine_failures(output_dir, labeled_input_path=input_path)

    assert report.wrong_prediction_ids == ("p2",)
    assert "p2" in report.high_priority_failure_ids


def test_gold_fields_used_only_for_analysis_not_mutation(tmp_path: Path) -> None:
    input_path, output_dir = make_run(tmp_path)
    predictions_path = output_dir / "predictions.jsonl"
    before = predictions_path.read_text(encoding="utf-8")

    report = mine_failures(output_dir, labeled_input_path=input_path)

    assert report.optional_accuracy is not None
    assert predictions_path.read_text(encoding="utf-8") == before


def test_failure_mining_report_forged_hash_rejected(tmp_path: Path) -> None:
    _, output_dir = make_run(tmp_path)
    report = mine_failures(output_dir)

    with pytest.raises(FailureMiningError):
        replace(report, report_hash="forged")


def test_validate_report_rejects_tampered_total_predictions(tmp_path: Path) -> None:
    _, output_dir = make_run(tmp_path)
    report = mine_failures(output_dir)
    tampered = replace(report, total_predictions=99, report_hash="")

    with pytest.raises(FailureMiningError):
        validate_failure_mining_report(tampered, output_dir)


def test_validate_report_rejects_tampered_top_abstention_reasons(tmp_path: Path) -> None:
    _, output_dir = make_run(tmp_path)
    report = mine_failures(output_dir)
    tampered = replace(report, top_abstention_reasons=(("fake", 99),), report_hash="")

    with pytest.raises(FailureMiningError):
        validate_failure_mining_report(tampered, output_dir)


def test_validate_report_rejects_tampered_wrong_prediction_ids(tmp_path: Path) -> None:
    input_path, output_dir = make_run(tmp_path)
    report = mine_failures(output_dir, labeled_input_path=input_path)
    tampered = replace(report, wrong_prediction_ids=("p999",), report_hash="")

    with pytest.raises(FailureMiningError):
        validate_failure_mining_report(tampered, output_dir, labeled_input_path=input_path)


def test_no_output_mutation_unless_explicitly_requested(tmp_path: Path) -> None:
    _, output_dir = make_run(tmp_path)
    before = {path.name for path in output_dir.iterdir()}

    mine_failures(output_dir)

    assert {path.name for path in output_dir.iterdir()} == before
    report_path = output_dir / "failure_mining_report.json"
    mine_failures(output_dir, output_path=report_path)
    assert report_path.exists()


def test_external_output_path_rejected_by_default(tmp_path: Path) -> None:
    _, output_dir = make_run(tmp_path)

    with pytest.raises(FailureMiningError):
        mine_failures(output_dir, output_path=tmp_path / "external_report.json")


def test_valid_mining_report_validates_against_artifacts(tmp_path: Path) -> None:
    input_path, output_dir = make_run(tmp_path)
    report = mine_failures(output_dir, labeled_input_path=input_path)

    assert validate_failure_mining_report(report, output_dir, labeled_input_path=input_path) == report
