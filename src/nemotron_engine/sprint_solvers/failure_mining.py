"""Failure mining for Sprint 1 prediction-run artifacts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, fields
import json
from pathlib import Path
from typing import Any

from nemotron_engine.core.schemas import stable_hash
from nemotron_engine.scoring.answer_extractor import normalize_answer_candidate


class FailureMiningError(ValueError):
    """Raised when Sprint failure mining cannot safely complete."""


_REQUIRED_ARTIFACTS = (
    "solver_attempts.jsonl",
    "predictions.jsonl",
    "disagreements.jsonl",
    "abstentions.jsonl",
    "family_metrics.json",
)


@dataclass(frozen=True)
class FailureMiningReport:
    total_predictions: int
    total_abstentions: int
    total_disagreements: int
    top_abstention_reasons: tuple[tuple[str, int], ...]
    top_rejected_reasons: tuple[tuple[str, int], ...]
    solver_error_counts: Mapping[str, int]
    optional_accuracy: Mapping[str, Any] | None
    wrong_prediction_ids: tuple[str, ...]
    high_priority_failure_ids: tuple[str, ...]
    report_hash: str = ""

    def __post_init__(self) -> None:
        for field_name in ("total_predictions", "total_abstentions", "total_disagreements"):
            value = int(getattr(self, field_name))
            if value < 0:
                raise FailureMiningError(f"{field_name} cannot be negative.")
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "top_abstention_reasons", _normalize_pairs(self.top_abstention_reasons))
        object.__setattr__(self, "top_rejected_reasons", _normalize_pairs(self.top_rejected_reasons))
        object.__setattr__(self, "solver_error_counts", _normalize_counts(self.solver_error_counts))
        object.__setattr__(self, "optional_accuracy", None if self.optional_accuracy is None else dict(self.optional_accuracy))
        object.__setattr__(self, "wrong_prediction_ids", tuple(str(item) for item in self.wrong_prediction_ids))
        object.__setattr__(self, "high_priority_failure_ids", tuple(str(item) for item in self.high_priority_failure_ids))
        _set_or_check_hash(self, "report_hash")


def mine_failures(
    output_dir: str | Path,
    *,
    labeled_input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    allow_external_output_path: bool = False,
) -> FailureMiningReport:
    """Read prediction artifacts and summarize abstentions, disagreements, and optional gold accuracy."""

    out_dir = Path(output_dir).resolve()
    artifacts = _read_required_artifacts(out_dir)
    attempts = artifacts["solver_attempts.jsonl"]
    predictions = artifacts["predictions.jsonl"]
    disagreements = artifacts["disagreements.jsonl"]
    abstentions = artifacts["abstentions.jsonl"]

    top_abstention_reasons = _top_counts(row.get("reason", "unknown") for row in abstentions)
    top_rejected_reasons = _top_counts(_iter_rejected_reasons(attempts))
    solver_error_counts = _solver_error_counts(attempts)
    optional_accuracy = None
    wrong_prediction_ids: tuple[str, ...] = ()
    high_priority_ids = {str(row.get("problem_id")) for row in disagreements}
    high_priority_ids.update(str(row.get("problem_id")) for row in abstentions)

    if labeled_input_path is not None:
        optional_accuracy = _compute_optional_accuracy(labeled_input_path, predictions, abstentions, disagreements)
        wrong_prediction_ids = tuple(optional_accuracy["wrong_prediction_ids"])
        high_priority_ids.update(wrong_prediction_ids)

    report = FailureMiningReport(
        total_predictions=len(predictions),
        total_abstentions=len(abstentions),
        total_disagreements=len(disagreements),
        top_abstention_reasons=top_abstention_reasons,
        top_rejected_reasons=top_rejected_reasons,
        solver_error_counts=solver_error_counts,
        optional_accuracy=optional_accuracy,
        wrong_prediction_ids=wrong_prediction_ids,
        high_priority_failure_ids=tuple(sorted(item for item in high_priority_ids if item and item != "None")),
    )
    validate_failure_mining_report(report, out_dir, labeled_input_path=labeled_input_path)
    if output_path is not None:
        resolved_output = Path(output_path).resolve()
        if not allow_external_output_path and not _is_within(resolved_output, out_dir):
            raise FailureMiningError("external output_path is rejected by default.")
        _write_report(resolved_output, report)
    return report


def validate_failure_mining_report(
    report: FailureMiningReport,
    output_dir: str | Path | None = None,
    *,
    labeled_input_path: str | Path | None = None,
) -> FailureMiningReport:
    """Validate a failure mining report hash and, optionally, artifact contents."""

    _validate_hash(report, "report_hash")
    if report.total_predictions < 0 or report.total_abstentions < 0 or report.total_disagreements < 0:
        raise FailureMiningError("report counts cannot be negative.")
    if output_dir is None:
        return report
    out_dir = Path(output_dir).resolve()
    artifacts = _read_required_artifacts(out_dir)
    attempts = artifacts["solver_attempts.jsonl"]
    predictions = artifacts["predictions.jsonl"]
    disagreements = artifacts["disagreements.jsonl"]
    abstentions = artifacts["abstentions.jsonl"]
    expected_top_abstentions = _top_counts(row.get("reason", "unknown") for row in abstentions)
    expected_top_rejected = _top_counts(_iter_rejected_reasons(attempts))
    expected_solver_errors = _solver_error_counts(attempts)
    if report.total_predictions != len(predictions):
        raise FailureMiningError("total_predictions does not match artifacts.")
    if report.total_abstentions != len(abstentions):
        raise FailureMiningError("total_abstentions does not match artifacts.")
    if report.total_disagreements != len(disagreements):
        raise FailureMiningError("total_disagreements does not match artifacts.")
    if report.top_abstention_reasons != expected_top_abstentions:
        raise FailureMiningError("top_abstention_reasons does not match artifacts.")
    if report.top_rejected_reasons != expected_top_rejected:
        raise FailureMiningError("top_rejected_reasons does not match artifacts.")
    if dict(report.solver_error_counts) != expected_solver_errors:
        raise FailureMiningError("solver_error_counts does not match artifacts.")
    if labeled_input_path is not None:
        expected_accuracy = _compute_optional_accuracy(labeled_input_path, predictions, abstentions, disagreements)
        if report.optional_accuracy != expected_accuracy:
            raise FailureMiningError("optional_accuracy does not match labeled artifacts.")
        if report.wrong_prediction_ids != tuple(expected_accuracy["wrong_prediction_ids"]):
            raise FailureMiningError("wrong_prediction_ids does not match labeled artifacts.")
    return report


def _read_required_artifacts(output_dir: Path) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for name in _REQUIRED_ARTIFACTS:
        path = output_dir / name
        if not path.exists():
            raise FailureMiningError(f"missing required artifact: {name}")
        artifacts[name] = _read_json(path) if name.endswith(".json") else _read_jsonl(path)
    return artifacts


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise FailureMiningError(f"malformed JSONL in {path.name} line {line_number}: {exc.msg}") from exc
                if not isinstance(item, dict):
                    raise FailureMiningError(f"malformed JSONL in {path.name} line {line_number}: row is not an object.")
                rows.append(item)
    except OSError as exc:
        raise FailureMiningError(f"cannot read artifact: {path.name}") from exc
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            item = json.load(handle)
    except json.JSONDecodeError as exc:
        raise FailureMiningError(f"malformed JSON artifact {path.name}: {exc.msg}") from exc
    except OSError as exc:
        raise FailureMiningError(f"cannot read artifact: {path.name}") from exc
    if not isinstance(item, dict):
        raise FailureMiningError(f"malformed JSON artifact {path.name}: object expected.")
    return item


def _top_counts(values: Any) -> tuple[tuple[str, int], ...]:
    counter = Counter(str(value) for value in values if str(value).strip())
    return tuple(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _iter_rejected_reasons(attempts: list[dict[str, Any]]) -> tuple[str, ...]:
    reasons: list[str] = []
    for row in attempts:
        reason = str(row.get("reason", ""))
        if row.get("status") == "solved":
            continue
        reasons.extend(item for item in reason.split(",") if item)
    return tuple(reasons)


def _solver_error_counts(attempts: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in attempts:
        reason = str(row.get("reason", ""))
        status = str(row.get("status", ""))
        if status == "error" or "candidate_exception" in reason:
            solver_name = str(row.get("solver_name", "unknown"))
            counts[solver_name] = counts.get(solver_name, 0) + 1
    return counts


def _compute_optional_accuracy(
    labeled_input_path: str | Path,
    predictions: list[dict[str, Any]],
    abstentions: list[dict[str, Any]],
    disagreements: list[dict[str, Any]],
) -> dict[str, Any]:
    labels = _read_labels(labeled_input_path)
    prediction_by_id = {str(row["id"]): row for row in predictions if "id" in row}
    abstention_ids = {str(row.get("problem_id")) for row in abstentions}
    disagreement_ids = {str(row.get("problem_id")) for row in disagreements}
    correct_count = 0
    wrong_predictions: list[dict[str, Any]] = []
    for problem_id, expected in labels.items():
        row = prediction_by_id.get(problem_id)
        if row is None:
            continue
        predicted = normalize_answer_candidate(row.get("answer"))
        if predicted == expected:
            correct_count += 1
        else:
            wrong_predictions.append({"id": problem_id, "predicted": predicted, "expected": expected})
    incorrect_count = len(wrong_predictions)
    labeled_count = len(labels)
    return {
        "labeled_count": labeled_count,
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "accuracy": (correct_count / labeled_count) if labeled_count else None,
        "wrong_predictions": wrong_predictions,
        "wrong_prediction_ids": tuple(item["id"] for item in wrong_predictions),
        "abstained_labeled_count": sum(1 for problem_id in labels if problem_id in abstention_ids),
        "disagreement_labeled_count": sum(1 for problem_id in labels if problem_id in disagreement_ids),
    }


def _read_labels(path: str | Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    for index, row in enumerate(_read_jsonl(Path(path)), start=1):
        problem_id = row.get("id", row.get("problem_id"))
        if problem_id is None:
            raise FailureMiningError(f"labeled row {index} is missing id/problem_id.")
        label = row.get("expected_answer", row.get("gold"))
        if label is None:
            continue
        labels[str(problem_id)] = normalize_answer_candidate(label)
    return labels


def _write_report(path: Path, report: FailureMiningReport) -> None:
    payload = {item.name: getattr(report, item.name) for item in fields(report)}
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        handle.write("\n")


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _normalize_pairs(pairs: Any) -> tuple[tuple[str, int], ...]:
    return tuple((str(key), int(value)) for key, value in pairs)


def _normalize_counts(counts: Mapping[str, int]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counts.items(), key=lambda item: str(item[0]))}


def _set_or_check_hash(instance: object, hash_field: str) -> None:
    expected = _payload_hash(instance, hash_field)
    current = getattr(instance, hash_field)
    if not current:
        object.__setattr__(instance, hash_field, expected)
    elif current != expected:
        raise FailureMiningError(f"{hash_field} does not match payload.")


def _validate_hash(instance: object, hash_field: str) -> None:
    if getattr(instance, hash_field) != _payload_hash(instance, hash_field):
        raise FailureMiningError(f"{hash_field} does not match payload.")


def _payload_hash(instance: object, hash_field: str) -> str:
    return stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})


__all__ = [
    "FailureMiningError",
    "FailureMiningReport",
    "mine_failures",
    "validate_failure_mining_report",
]
