"""Private-like regression gate for Pass 8 promotion."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import math
from typing import Any, Mapping, Sequence

from nemotron_engine.core.schemas import stable_hash
from nemotron_engine.scoring.local_scorer import score_completion

from .transfer_harness import TransferEvaluationReport, TransferExample, _failure_kind, _format_policy


class PrivateLikeError(ValueError):
    """Raised when private-like gate reports are inconsistent."""


@dataclass(frozen=True)
class PrivateLikeGateConfig:
    min_examples: int = 2
    accuracy_threshold: float = 0.90
    max_regression: float = 0.08
    max_format_error_rate: float = 0.0
    max_extraction_error_rate: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    config_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.min_examples, int) or isinstance(self.min_examples, bool) or self.min_examples < 0:
            raise PrivateLikeError("min_examples must be a non-negative integer.")
        for name in ("accuracy_threshold", "max_regression", "max_format_error_rate", "max_extraction_error_rate"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise PrivateLikeError(f"{name} must be a non-negative number.")
            object.__setattr__(self, name, float(value))
        object.__setattr__(self, "metadata", dict(self.metadata))
        expected = _payload_hash(self, "config_hash")
        if not self.config_hash:
            object.__setattr__(self, "config_hash", expected)
        elif self.config_hash != expected:
            raise PrivateLikeError("config_hash does not match private-like config payload.")


@dataclass(frozen=True)
class PrivateLikeReport:
    total: int
    correct: int
    accuracy: float
    baseline_accuracy: float
    regression_vs_baseline: float
    format_error_rate: float
    extraction_error_rate: float
    contamination_count: int
    passed: bool
    failure_reasons: tuple[str, ...] = ()
    example_ids: tuple[str, ...] = ()
    report_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.total, int) or not isinstance(self.correct, int) or self.total < 0 or self.correct < 0 or self.correct > self.total:
            raise PrivateLikeError("private-like counts are invalid.")
        if not isinstance(self.contamination_count, int) or self.contamination_count < 0:
            raise PrivateLikeError("contamination_count must be non-negative.")
        expected_accuracy = 0.0 if self.total == 0 else self.correct / self.total
        for name in ("accuracy", "baseline_accuracy", "regression_vs_baseline", "format_error_rate", "extraction_error_rate"):
            _require_rate(getattr(self, name), name)
        if abs(float(self.accuracy) - expected_accuracy) > 1e-12:
            raise PrivateLikeError("accuracy is inconsistent with counts.")
        expected_regression = max(0.0, float(self.baseline_accuracy) - expected_accuracy)
        if abs(float(self.regression_vs_baseline) - expected_regression) > 1e-12:
            raise PrivateLikeError("regression_vs_baseline is inconsistent.")
        object.__setattr__(self, "accuracy", expected_accuracy)
        object.__setattr__(self, "baseline_accuracy", float(self.baseline_accuracy))
        object.__setattr__(self, "regression_vs_baseline", expected_regression)
        object.__setattr__(self, "format_error_rate", float(self.format_error_rate))
        object.__setattr__(self, "extraction_error_rate", float(self.extraction_error_rate))
        failures = tuple(str(item) for item in self.failure_reasons)
        if self.passed and failures:
            raise PrivateLikeError("passed=True cannot include failure_reasons.")
        if self.passed and self.contamination_count > 0:
            raise PrivateLikeError("passed=True cannot include contamination.")
        if self.passed and float(self.extraction_error_rate) > 0.0:
            raise PrivateLikeError("passed=True cannot include extraction errors.")
        if self.passed and float(self.format_error_rate) > 0.0:
            raise PrivateLikeError("passed=True cannot include format errors.")
        object.__setattr__(self, "failure_reasons", failures)
        object.__setattr__(self, "example_ids", tuple(str(item) for item in self.example_ids))
        expected = _payload_hash(self, "report_hash")
        if not self.report_hash:
            object.__setattr__(self, "report_hash", expected)
        elif self.report_hash != expected:
            raise PrivateLikeError("report_hash does not match private-like report payload.")


def evaluate_private_like_gate(
    examples: Sequence[TransferExample],
    transfer_report: TransferEvaluationReport,
    config: PrivateLikeGateConfig | None = None,
) -> PrivateLikeReport:
    """Evaluate private-like transfer safety without hidden scoring shortcuts."""

    cfg = config or PrivateLikeGateConfig()
    if not isinstance(transfer_report, TransferEvaluationReport):
        raise PrivateLikeError("transfer_report must be a TransferEvaluationReport.")
    private_slice = next((item for item in transfer_report.slice_results if item.slice_name == "private_like"), None)
    if private_slice is None:
        raise PrivateLikeError("private_like slice absent.")
    private_examples = tuple(example for example in examples if example.split == "private_like")
    if len(private_examples) != private_slice.total:
        raise PrivateLikeError("private_like examples do not match transfer report count.")
    correct = 0
    format_errors = 0
    extraction_errors = 0
    contamination_count = 0
    failures: list[str] = []
    for example in private_examples:
        if _contaminated(example.metadata):
            contamination_count += 1
            failures.append(f"{example.problem_id}:contamination")
        result = score_completion(example.completion, example.expected_answer, policy=_format_policy(example))
        if result.correct:
            correct += 1
        elif _failure_kind(result.error) == "extraction_error":
            extraction_errors += 1
        elif _failure_kind(result.error) == "format_error":
            format_errors += 1
    total = len(private_examples)
    accuracy = 0.0 if total == 0 else correct / total
    format_rate = 0.0 if total == 0 else format_errors / total
    extraction_rate = 0.0 if total == 0 else extraction_errors / total
    regression = max(0.0, private_slice.baseline_accuracy - accuracy)
    if total < cfg.min_examples:
        failures.append("insufficient_examples")
    if accuracy < cfg.accuracy_threshold:
        failures.append("accuracy_below_threshold")
    if regression > cfg.max_regression:
        failures.append("private_like_regression")
    if format_rate > cfg.max_format_error_rate:
        failures.append("format_error_rate")
    if extraction_rate > cfg.max_extraction_error_rate:
        failures.append("extraction_error_rate")
    if contamination_count:
        failures.append("contamination")
    return PrivateLikeReport(
        total=total,
        correct=correct,
        accuracy=accuracy,
        baseline_accuracy=private_slice.baseline_accuracy,
        regression_vs_baseline=regression,
        format_error_rate=format_rate,
        extraction_error_rate=extraction_rate,
        contamination_count=contamination_count,
        passed=not failures,
        failure_reasons=tuple(sorted(set(failures))),
        example_ids=tuple(example.problem_id for example in private_examples),
    )


def _contaminated(metadata: Mapping[str, Any]) -> bool:
    return bool(
        metadata.get("contamination_flags")
        or metadata.get("contaminated") is True
        or metadata.get("leakage") is True
        or metadata.get("split_leakage") is True
    )


def _require_rate(value: Any, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or math.isnan(float(value)) or math.isinf(float(value)):
        raise PrivateLikeError(f"{field_name} must be finite.")
    if float(value) < 0.0 or float(value) > 1.0:
        raise PrivateLikeError(f"{field_name} must be in [0, 1].")


def _payload_hash(instance: object, hash_field: str) -> str:
    return stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})


__all__ = [
    "PrivateLikeError",
    "PrivateLikeGateConfig",
    "PrivateLikeReport",
    "evaluate_private_like_gate",
]
