"""Failure mining for competition sprint artifacts."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, fields
import json
from pathlib import Path
from typing import Any

from nemotron_engine.core.schemas import stable_hash


class CompetitionFailureMiningError(ValueError):
    """Raised when competition failure mining artifacts are invalid."""


@dataclass(frozen=True)
class CompetitionFailureMiningReport:
    total_rows: int
    prediction_count: int
    abstention_count: int
    accuracy: float | None
    accuracy_by_family: dict[str, float | None]
    abstentions_by_family: dict[str, int]
    wrong_predictions_by_family: dict[str, int]
    top_error_reasons: tuple[tuple[str, int], ...]
    report_hash: str = ""

    def __post_init__(self) -> None:
        for field_name in ("total_rows", "prediction_count", "abstention_count"):
            value = int(getattr(self, field_name))
            if value < 0:
                raise CompetitionFailureMiningError(f"{field_name} cannot be negative.")
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "accuracy_by_family", dict(sorted(self.accuracy_by_family.items())))
        object.__setattr__(self, "abstentions_by_family", {str(k): int(v) for k, v in sorted(self.abstentions_by_family.items())})
        object.__setattr__(self, "wrong_predictions_by_family", {str(k): int(v) for k, v in sorted(self.wrong_predictions_by_family.items())})
        object.__setattr__(self, "top_error_reasons", tuple((str(k), int(v)) for k, v in self.top_error_reasons))
        _set_or_check_hash(self, "report_hash")


def mine_competition_failures(output_dir: str | Path) -> CompetitionFailureMiningReport:
    """Summarize train-eval prediction and abstention artifacts."""

    out = Path(output_dir)
    predictions = _read_jsonl(out / "competition_predictions_train.jsonl")
    abstentions = _read_jsonl(out / "competition_abstentions_train.jsonl")
    errors = _read_jsonl(out / "competition_errors_train.jsonl")
    _read_json(out / "competition_family_metrics_train.json")
    correct = sum(1 for row in predictions if row.get("correct") is True)
    family_total: dict[str, int] = defaultdict(int)
    family_correct: dict[str, int] = defaultdict(int)
    wrong_by_family: dict[str, int] = defaultdict(int)
    abstain_by_family: dict[str, int] = defaultdict(int)
    for row in predictions:
        family = str(row.get("family", "unknown"))
        family_total[family] += 1
        if row.get("correct") is True:
            family_correct[family] += 1
        elif "correct" in row:
            wrong_by_family[family] += 1
    for row in abstentions:
        abstain_by_family[str(row.get("family", "unknown"))] += 1
    error_reasons = Counter(str(row.get("reason", "unknown")) for row in [*abstentions, *errors])
    report = CompetitionFailureMiningReport(
        total_rows=len(predictions) + len(abstentions) + len(errors),
        prediction_count=len(predictions),
        abstention_count=len(abstentions),
        accuracy=(correct / len(predictions)) if predictions else None,
        accuracy_by_family={family: (family_correct[family] / count if count else None) for family, count in family_total.items()},
        abstentions_by_family=dict(abstain_by_family),
        wrong_predictions_by_family=dict(wrong_by_family),
        top_error_reasons=tuple(sorted(error_reasons.items(), key=lambda item: (-item[1], item[0]))),
    )
    validate_competition_failure_mining_report(report)
    return report


def validate_competition_failure_mining_report(report: CompetitionFailureMiningReport) -> CompetitionFailureMiningReport:
    """Validate report hash."""

    _validate_hash(report, "report_hash")
    return report


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise CompetitionFailureMiningError(f"missing artifact: {path.name}")
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise CompetitionFailureMiningError(f"malformed artifact: {path.name}")
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise CompetitionFailureMiningError(f"missing artifact: {path.name}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise CompetitionFailureMiningError(f"malformed artifact: {path.name}")
    return payload


def _set_or_check_hash(instance: object, hash_field: str) -> None:
    expected = _payload_hash(instance, hash_field)
    current = getattr(instance, hash_field)
    if not current:
        object.__setattr__(instance, hash_field, expected)
    elif current != expected:
        raise CompetitionFailureMiningError(f"{hash_field} does not match payload.")


def _validate_hash(instance: object, hash_field: str) -> None:
    if getattr(instance, hash_field) != _payload_hash(instance, hash_field):
        raise CompetitionFailureMiningError(f"{hash_field} does not match payload.")


def _payload_hash(instance: object, hash_field: str) -> str:
    return stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})


__all__ = [
    "CompetitionFailureMiningError",
    "CompetitionFailureMiningReport",
    "mine_competition_failures",
    "validate_competition_failure_mining_report",
]
