from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.schema import RowValidationError, validate_row
from kaggle_anti086.solvers.answer_normalizer import answers_match, normalize_answer
from kaggle_anti086.solvers.solver_ensemble import ANSWER_TYPE_BY_FAMILY, SolverEnsemble

try:
    from kaggle_anti086.kaggle_path_safety import require_writable_output_path, safe_write_text
except Exception:  # pragma: no cover - fallback for isolated local copies
    def require_writable_output_path(path: str | Path, *, field_name: str, **_: object) -> Path:
        candidate = Path(path)
        normalized = str(candidate).replace("\\", "/")
        if normalized.startswith("/kaggle/input") or normalized.startswith("/tmp/"):
            raise SystemExit(f"{field_name} is not a safe durable eval output path: {candidate}")
        return candidate

    def safe_write_text(path: str | Path, content: str, *, field_name: str, encoding: str = "utf-8") -> Path:
        target = require_writable_output_path(path, field_name=field_name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding=encoding)
        return target


def run_solver_eval(
    input_path: str | Path,
    out_report: str | Path,
    out_predictions: str | Path,
    *,
    allow_invalid_rows: bool = False,
    min_exact_match: float = 0.0,
    min_attempt_rate: float = 0.0,
    max_unsafe_answer_rate: float = 1.0,
    min_correct_abstain_rate: float = 0.0,
) -> dict:
    rows = _read_jsonl(Path(input_path))
    out_report = require_writable_output_path(out_report, field_name="out_report")
    out_predictions = require_writable_output_path(out_predictions, field_name="out_predictions")
    ensemble = SolverEnsemble()
    predictions: list[dict] = []
    by_family: dict[str, dict] = {}
    failure_examples: list[dict] = []
    answerable_count = 0
    expected_abstain_count = 0
    correct_abstain_count = 0
    wrong_abstain_count = 0
    unsafe_answer_count = 0
    wrong_answer_count = 0

    for idx, row in enumerate(rows):
        try:
            parsed = validate_row(row, context=f"row[{idx}]")
        except RowValidationError:
            if not allow_invalid_rows:
                raise
            parsed = None
        family = str(row.get("family", "unknown"))
        expected = "" if row.get("answer") is None else str(row.get("answer"))
        expected_behavior = str(row.get("metadata", {}).get("expected_solver_behavior", "answer"))
        result = ensemble.run_all(row)
        best = None if result.abstained or not result.candidates else result.candidates[0]
        answer_type = ANSWER_TYPE_BY_FAMILY.get(family)
        prediction = "" if best is None else best.answer
        normalized_expected = normalize_answer(expected, expected_type=answer_type).normalized
        normalized_prediction = normalize_answer(prediction, expected_type=answer_type).normalized if prediction else ""
        answer_correct = False if best is None or expected_behavior == "abstain" else answers_match(prediction, expected, answer_type=answer_type)
        if expected_behavior == "abstain":
            expected_abstain_count += 1
            if best is None:
                correct_abstain_count += 1
            else:
                unsafe_answer_count += 1
        else:
            answerable_count += 1
            if best is None:
                wrong_abstain_count += 1
            elif not answer_correct:
                wrong_answer_count += 1
        prediction_row = {
            "id": row.get("id", f"row_{idx}"),
            "family": family,
            "subfamily": row.get("subfamily", ""),
            "expected": expected,
            "prediction": prediction,
            "correct": bool(answer_correct),
            "behavior_correct": bool(best is None) if expected_behavior == "abstain" else bool(answer_correct),
            "expected_solver_behavior": expected_behavior,
            "abstained": bool(best is None),
            "best_source": "" if best is None else best.source,
            "confidence": 0.0 if best is None else best.confidence,
            "risk": "" if best is None else best.risk,
            "verified": False if best is None else best.verified,
            "candidate_count": 0 if result.abstained else len(result.candidates),
            "reason": result.reason,
            "normalized_expected": normalized_expected,
            "normalized_prediction": normalized_prediction,
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
        if answer_correct:
            bucket["correct"] += 1
        if not prediction_row["behavior_correct"] and len(failure_examples) < 20:
            failure_examples.append(
                {
                    "id": prediction_row["id"],
                    "family": family,
                    "abstained": prediction_row["abstained"],
                    "expected": expected,
                    "prediction": prediction,
                    "prompt_excerpt": str(row.get("prompt", ""))[:300],
                    "reason": result.reason if result.abstained else ("unsafe_answer" if expected_behavior == "abstain" else "wrong_prediction"),
                }
            )

    for bucket in by_family.values():
        bucket["exact_match"] = 0.0 if bucket["count"] == 0 else bucket["correct"] / bucket["count"]
    row_count = len(rows)
    attempted_count = sum(0 if item["abstained"] else 1 for item in predictions)
    verified_count = sum(1 for item in predictions if item["verified"])
    correct_count = sum(1 for item in predictions if item["correct"])
    exact_match = 0.0 if row_count == 0 else correct_count / row_count
    attempt_rate = 0.0 if row_count == 0 else attempted_count / row_count
    unsafe_answer_rate = 0.0 if row_count == 0 else unsafe_answer_count / row_count
    correct_abstain_rate = 1.0 if expected_abstain_count == 0 else correct_abstain_count / expected_abstain_count
    quality_status = _quality_status(
        row_count,
        exact_match,
        attempt_rate,
        min_exact_match,
        min_attempt_rate,
        unsafe_answer_rate,
        correct_abstain_rate,
        max_unsafe_answer_rate,
        min_correct_abstain_rate,
    )
    report = {
        "status": "PASS",
        "execution_status": "PASS",
        "quality_status": quality_status,
        "row_count": row_count,
        "attempted_count": attempted_count,
        "attempt_rate": attempt_rate,
        "verified_count": verified_count,
        "correct_count": correct_count,
        "exact_match": exact_match,
        "answerable_count": answerable_count,
        "expected_abstain_count": expected_abstain_count,
        "correct_abstain_count": correct_abstain_count,
        "wrong_abstain_count": wrong_abstain_count,
        "wrong_answer_count": wrong_answer_count,
        "unsafe_answer_count": unsafe_answer_count,
        "unsafe_answer_rate": unsafe_answer_rate,
        "correct_abstain_rate": correct_abstain_rate,
        "abstained_count": row_count - attempted_count,
        "by_family": dict(sorted(by_family.items())),
        "failure_examples": failure_examples,
        "thresholds": {
            "min_exact_match": min_exact_match,
            "min_attempt_rate": min_attempt_rate,
            "max_unsafe_answer_rate": max_unsafe_answer_rate,
            "min_correct_abstain_rate": min_correct_abstain_rate,
        },
    }
    _write_jsonl(Path(out_predictions), predictions)
    safe_write_text(Path(out_report), json.dumps(report, indent=2, sort_keys=True) + "\n", field_name="out_report")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic solver eval smoke.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-report", required=True)
    parser.add_argument("--out-predictions", required=True)
    parser.add_argument("--allow-invalid-rows", action="store_true")
    parser.add_argument("--min-exact-match", type=float, default=0.0)
    parser.add_argument("--min-attempt-rate", type=float, default=0.0)
    parser.add_argument("--max-unsafe-answer-rate", type=float, default=1.0)
    parser.add_argument("--min-correct-abstain-rate", type=float, default=0.0)
    parser.add_argument("--fail-on-quality-gate", action="store_true")
    args = parser.parse_args(argv)
    report = run_solver_eval(
        args.input,
        args.out_report,
        args.out_predictions,
        allow_invalid_rows=args.allow_invalid_rows,
        min_exact_match=args.min_exact_match,
        min_attempt_rate=args.min_attempt_rate,
        max_unsafe_answer_rate=args.max_unsafe_answer_rate,
        min_correct_abstain_rate=args.min_correct_abstain_rate,
    )
    print(
        json.dumps(
            {
                "execution_status": report["execution_status"],
                "quality_status": report["quality_status"],
                "row_count": report["row_count"],
                "exact_match": report["exact_match"],
                "attempt_rate": report["attempt_rate"],
                "unsafe_answer_rate": report["unsafe_answer_rate"],
                "correct_abstain_rate": report["correct_abstain_rate"],
            },
            sort_keys=True,
        )
    )
    if args.fail_on_quality_gate and report["quality_status"] == "FAIL":
        return 2
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
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    safe_write_text(path, payload, field_name="out_predictions")


def _quality_status(
    row_count: int,
    exact_match: float,
    attempt_rate: float,
    min_exact_match: float,
    min_attempt_rate: float,
    unsafe_answer_rate: float,
    correct_abstain_rate: float,
    max_unsafe_answer_rate: float = 1.0,
    min_correct_abstain_rate: float = 0.0,
) -> str:
    if row_count == 0:
        return "WARN"
    if (
        exact_match < min_exact_match
        or attempt_rate < min_attempt_rate
        or unsafe_answer_rate > max_unsafe_answer_rate
        or correct_abstain_rate < min_correct_abstain_rate
    ):
        return "FAIL"
    return "PASS"


if __name__ == "__main__":
    raise SystemExit(main())
