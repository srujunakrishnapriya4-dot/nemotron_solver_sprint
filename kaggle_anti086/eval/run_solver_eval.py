from __future__ import annotations

import argparse
import json
from pathlib import Path

from kaggle_anti086.data.schema import RowValidationError, validate_row
from kaggle_anti086.solvers.answer_normalizer import answers_match
from kaggle_anti086.solvers.solver_ensemble import ANSWER_TYPE_BY_FAMILY, SolverEnsemble


def run_solver_eval(
    input_path: str | Path,
    out_report: str | Path,
    out_predictions: str | Path,
    *,
    allow_invalid_rows: bool = False,
) -> dict:
    rows = _read_jsonl(Path(input_path))
    ensemble = SolverEnsemble()
    predictions: list[dict] = []
    by_family: dict[str, dict] = {}
    failure_examples: list[dict] = []

    for idx, row in enumerate(rows):
        try:
            parsed = validate_row(row, context=f"row[{idx}]")
        except RowValidationError:
            if not allow_invalid_rows:
                raise
            parsed = None
        family = str(row.get("family", "unknown"))
        expected = "" if row.get("answer") is None else str(row.get("answer"))
        result = ensemble.run_all(row)
        best = None if result.abstained or not result.candidates else result.candidates[0]
        answer_type = ANSWER_TYPE_BY_FAMILY.get(family)
        prediction = "" if best is None else best.answer
        correct = False if best is None else answers_match(prediction, expected, answer_type=answer_type)
        prediction_row = {
            "id": row.get("id", f"row_{idx}"),
            "family": family,
            "expected": expected,
            "prediction": prediction,
            "correct": bool(correct),
            "abstained": bool(best is None),
            "best_source": "" if best is None else best.source,
            "confidence": 0.0 if best is None else best.confidence,
            "risk": "" if best is None else best.risk,
            "verified": False if best is None else best.verified,
            "candidate_count": 0 if result.abstained else len(result.candidates),
        }
        predictions.append(prediction_row)
        bucket = by_family.setdefault(
            family,
            {"count": 0, "attempted": 0, "verified": 0, "correct": 0, "exact_match": 0.0},
        )
        bucket["count"] += 1
        if best is not None:
            bucket["attempted"] += 1
            if best.verified:
                bucket["verified"] += 1
        if correct:
            bucket["correct"] += 1
        if not correct and len(failure_examples) < 20:
            failure_examples.append(
                {
                    "id": prediction_row["id"],
                    "family": family,
                    "abstained": prediction_row["abstained"],
                    "expected": expected,
                    "prediction": prediction,
                    "prompt_excerpt": str(row.get("prompt", ""))[:300],
                    "reason": result.reason if result.abstained else "wrong_prediction",
                }
            )

    for bucket in by_family.values():
        bucket["exact_match"] = 0.0 if bucket["count"] == 0 else bucket["correct"] / bucket["count"]
    row_count = len(rows)
    attempted_count = sum(0 if item["abstained"] else 1 for item in predictions)
    verified_count = sum(1 for item in predictions if item["verified"])
    correct_count = sum(1 for item in predictions if item["correct"])
    report = {
        "status": "PASS",
        "row_count": row_count,
        "attempted_count": attempted_count,
        "verified_count": verified_count,
        "correct_count": correct_count,
        "exact_match": 0.0 if row_count == 0 else correct_count / row_count,
        "abstained_count": row_count - attempted_count,
        "by_family": dict(sorted(by_family.items())),
        "failure_examples": failure_examples,
    }
    _write_jsonl(Path(out_predictions), predictions)
    Path(out_report).parent.mkdir(parents=True, exist_ok=True)
    Path(out_report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic solver eval smoke.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-report", required=True)
    parser.add_argument("--out-predictions", required=True)
    parser.add_argument("--allow-invalid-rows", action="store_true")
    args = parser.parse_args(argv)
    report = run_solver_eval(
        args.input,
        args.out_report,
        args.out_predictions,
        allow_invalid_rows=args.allow_invalid_rows,
    )
    print(json.dumps({"status": report["status"], "row_count": report["row_count"], "exact_match": report["exact_match"]}, sort_keys=True))
    return 0


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
