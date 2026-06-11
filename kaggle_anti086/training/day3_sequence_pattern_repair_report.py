from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.v2_corpus_io import file_record, read_jsonl, write_json_checked
from kaggle_anti086.solvers.sequence_pattern_solver import SequencePatternSolver


DEFAULT_PREDICTION_FILES = (
    "artifacts/sprint11/day2_predictions/core_eval_solver_only_predictions.jsonl",
    "artifacts/sprint11/day2_predictions/family_eval_solver_only_predictions.jsonl",
    "artifacts/sprint11/day2_predictions/rule_holdout_solver_only_predictions.jsonl",
    "artifacts/sprint11/day2_predictions/anti_leak_solver_only_predictions.jsonl",
)


def build_sequence_pattern_repair_report(prediction_files: list[str | Path] | None = None) -> dict[str, Any]:
    paths = [Path(path) for path in (prediction_files or list(DEFAULT_PREDICTION_FILES))]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing Day2 prediction artifacts for sequence measurement: {missing}")

    solver = SequencePatternSolver()
    sequence_rows: list[dict[str, Any]] = []
    for path in paths:
        dataset = path.name.replace("_solver_only_predictions.jsonl", "")
        for row in read_jsonl(path):
            if row.get("family") == "sequence_pattern":
                sequence_rows.append(dict(row) | {"dataset": dataset})

    previous_correct = sum(1 for row in sequence_rows if bool(row.get("correct")))
    new_correct = 0
    unsafe_answer_count = 0
    recovered: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for row in sequence_rows:
        expected = _expected_answer(row)
        result = solver.solve({"family": "sequence_pattern", "prompt": row.get("prompt", "")})
        if result.abstained or not result.candidates:
            blocked.append(_example(row, expected=expected, solver_answer="", reason=result.reason))
            continue
        answer = result.candidates[0].answer
        if expected == "ABSTAIN":
            unsafe_answer_count += 1
            blocked.append(_example(row, expected=expected, solver_answer=answer, reason="unsafe_answer_on_expected_abstain"))
            continue
        if answer == expected:
            new_correct += 1
            recovered.append(_example(row, expected=expected, solver_answer=answer, reason="recovered"))
        else:
            blocked.append(_example(row, expected=expected, solver_answer=answer, reason="wrong_answer"))

    newly_recovered = max(0, new_correct - previous_correct)
    projected_total = 1383 + newly_recovered if sequence_rows else None
    projected_accuracy = (projected_total / 1792) if projected_total is not None else None
    status = "FAIL" if unsafe_answer_count else ("WARN" if newly_recovered == 0 else "PASS")
    return {
        "schema_version": 1,
        "created_by": "DAY3_SEQUENCE_PATTERN_REPAIR_REPORT",
        "status": status,
        "family": "sequence_pattern",
        "inputs": {str(path): file_record(path) for path in paths},
        "rows_seen": len(sequence_rows),
        "previous_correct": previous_correct,
        "new_correct": new_correct,
        "newly_recovered": newly_recovered,
        "remaining_wrong_or_abstain": max(0, len(sequence_rows) - new_correct),
        "unsafe_answer_count": unsafe_answer_count,
        "examples_recovered": recovered[:10],
        "examples_still_blocked": blocked[:10],
        "projected_total_correct_if_applied": projected_total,
        "projected_raw_exact_match_if_applied": projected_accuracy,
        "leaderboard_claim": False,
        "package_authorized": False,
        "submission_authorized": False,
        "v2a_150_authorized": False,
        "known_unresolved_failures": [
            {
                "test": "tests/test_sprint11_numeric_formula_solver.py::test_numeric_formula_ambiguous_not_low_risk_verified",
                "classification": "pre_existing_unresolved_with_evidence",
            }
        ],
    }


def _expected_answer(row: dict[str, Any]) -> str:
    for key in ("gold_answer", "expected_answer", "expected", "answer"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _example(row: dict[str, Any], *, expected: str, solver_answer: str, reason: str) -> dict[str, Any]:
    return {
        "dataset": row.get("dataset"),
        "row_id": row.get("row_id") or row.get("id"),
        "prompt": row.get("prompt"),
        "expected": expected,
        "previous_output": row.get("normalized_answer") or row.get("extracted_answer") or row.get("raw_output"),
        "solver_answer": solver_answer,
        "reason": reason,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure standalone Day3 sequence_pattern solver recovery on Day2 artifacts.")
    parser.add_argument("--prediction-file", action="append", dest="prediction_files", default=[])
    parser.add_argument("--out", default="artifacts/sprint11/day3_sequence_pattern_repair_report.json")
    args = parser.parse_args(argv)
    report = build_sequence_pattern_repair_report(args.prediction_files or None)
    write_json_checked(args.out, report, field_name="day3_sequence_pattern_repair_report")
    print(
        json.dumps(
            {
                "status": report["status"],
                "rows_seen": report["rows_seen"],
                "newly_recovered": report["newly_recovered"],
                "unsafe_answer_count": report["unsafe_answer_count"],
                "out": args.out,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
