"""Split-safe transfer evaluation harness for Pass 8."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import math
from typing import Any, Mapping, Sequence

from nemotron_engine.core.schemas import stable_hash
from nemotron_engine.scoring.answer_extractor import AnswerExtractionError, find_boxed_spans
from nemotron_engine.scoring.format_policy import AnswerFormatError, AnswerType, FormatPolicy
from nemotron_engine.scoring.local_scorer import score_completion


TRANSFER_SLICE_NAMES = (
    "known_exact_family",
    "hard_known",
    "known_domain_new_composition",
    "new_primitive_known_domain",
    "new_domain_structured",
    "private_like",
    "stress_only",
    "forbidden_holdout",
)

DEFAULT_GAP_THRESHOLDS = {
    "known_exact_family": 0.02,
    "hard_known": 0.05,
    "known_domain_new_composition": 0.08,
    "new_primitive_known_domain": 0.10,
    "new_domain_structured": 0.12,
    "private_like": 0.08,
    "stress_only": 0.15,
    "forbidden_holdout": 0.15,
}


class TransferHarnessError(ValueError):
    """Raised when transfer evaluation contracts are violated."""


@dataclass(frozen=True)
class TransferExample:
    problem_id: str
    family_id: str
    primitive_family_id: str
    format_family_id: str
    split: str
    expected_answer: Any
    completion: str
    answer_type: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("problem_id", "family_id", "primitive_family_id", "format_family_id", "split", "completion", "answer_type"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise TransferHarnessError(f"{name} must be a non-empty string.")
        if self.split not in TRANSFER_SLICE_NAMES:
            raise TransferHarnessError(f"unsupported transfer slice: {self.split!r}.")
        try:
            AnswerType(str(self.answer_type))
        except ValueError as exc:
            raise TransferHarnessError(f"unsupported answer_type: {self.answer_type!r}.") from exc
        if self.expected_answer is None or (isinstance(self.expected_answer, str) and not self.expected_answer.strip()):
            raise TransferHarnessError("expected_answer must be non-empty.")
        if not isinstance(self.metadata, Mapping):
            raise TransferHarnessError("metadata must be a mapping.")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class TransferHarnessConfig:
    min_examples_per_slice: int = 2
    allow_small_slices: bool = False
    required_slices: tuple[str, ...] = TRANSFER_SLICE_NAMES
    baseline_accuracies: Mapping[str, float] = field(default_factory=dict)
    gap_thresholds: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_GAP_THRESHOLDS))
    metadata: Mapping[str, Any] = field(default_factory=dict)
    config_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.min_examples_per_slice, int) or isinstance(self.min_examples_per_slice, bool) or self.min_examples_per_slice < 0:
            raise TransferHarnessError("min_examples_per_slice must be a non-negative integer.")
        if not isinstance(self.allow_small_slices, bool):
            raise TransferHarnessError("allow_small_slices must be boolean.")
        for slice_name in self.required_slices:
            if slice_name not in TRANSFER_SLICE_NAMES:
                raise TransferHarnessError(f"unsupported required slice: {slice_name!r}.")
        object.__setattr__(self, "required_slices", tuple(self.required_slices))
        object.__setattr__(self, "baseline_accuracies", _normalize_float_mapping(self.baseline_accuracies, "baseline_accuracies"))
        object.__setattr__(self, "gap_thresholds", _normalize_float_mapping(self.gap_thresholds, "gap_thresholds"))
        object.__setattr__(self, "metadata", dict(self.metadata))
        expected = _payload_hash(self, "config_hash")
        if not self.config_hash:
            object.__setattr__(self, "config_hash", expected)
        elif self.config_hash != expected:
            raise TransferHarnessError("config_hash does not match transfer harness config payload.")


@dataclass(frozen=True)
class TransferSliceResult:
    slice_name: str
    total: int
    correct: int
    accuracy: float
    baseline_accuracy: float
    transfer_gap: float
    passed: bool
    failures: tuple[str, ...] = ()
    example_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.slice_name not in TRANSFER_SLICE_NAMES:
            raise TransferHarnessError(f"unsupported slice_name: {self.slice_name!r}.")
        if not isinstance(self.total, int) or not isinstance(self.correct, int) or self.total < 0 or self.correct < 0 or self.correct > self.total:
            raise TransferHarnessError("slice counts are invalid.")
        expected_accuracy = 0.0 if self.total == 0 else self.correct / self.total
        _require_finite(self.accuracy, "accuracy")
        _require_finite(self.baseline_accuracy, "baseline_accuracy")
        _require_finite(self.transfer_gap, "transfer_gap")
        if float(self.accuracy) < 0.0 or float(self.accuracy) > 1.0:
            raise TransferHarnessError("accuracy must be in [0, 1].")
        if float(self.baseline_accuracy) < 0.0 or float(self.baseline_accuracy) > 1.0:
            raise TransferHarnessError("baseline_accuracy must be in [0, 1].")
        if float(self.transfer_gap) < 0.0:
            raise TransferHarnessError("transfer_gap must be non-negative.")
        if abs(float(self.accuracy) - expected_accuracy) > 1e-12:
            raise TransferHarnessError("slice accuracy is inconsistent with counts.")
        expected_gap = max(0.0, float(self.baseline_accuracy) - expected_accuracy)
        if abs(float(self.transfer_gap) - expected_gap) > 1e-12:
            raise TransferHarnessError("transfer_gap is inconsistent with baseline/accuracy.")
        failures = tuple(str(item) for item in self.failures)
        if self.passed and self.total == 0:
            raise TransferHarnessError("passed=True requires at least one example.")
        if self.passed and failures:
            raise TransferHarnessError("passed=True cannot include failures.")
        object.__setattr__(self, "accuracy", expected_accuracy)
        object.__setattr__(self, "baseline_accuracy", float(self.baseline_accuracy))
        object.__setattr__(self, "transfer_gap", expected_gap)
        object.__setattr__(self, "failures", failures)
        object.__setattr__(self, "example_ids", tuple(str(item) for item in self.example_ids))


@dataclass(frozen=True)
class TransferEvaluationReport:
    config_hash: str
    total_examples: int
    slice_results: tuple[TransferSliceResult, ...]
    overall_accuracy: float
    passed: bool
    failure_reasons: tuple[str, ...] = ()
    required_slices: tuple[str, ...] = TRANSFER_SLICE_NAMES
    allow_small_slices: bool = False
    report_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.config_hash, str) or not self.config_hash.strip():
            raise TransferHarnessError("config_hash must be non-empty.")
        slices = tuple(sorted(self.slice_results, key=lambda item: item.slice_name))
        if len({item.slice_name for item in slices}) != len(slices):
            raise TransferHarnessError("slice_results cannot contain duplicate slice names.")
        total = sum(item.total for item in slices)
        correct = sum(item.correct for item in slices)
        if self.total_examples != total:
            raise TransferHarnessError("total_examples mismatch.")
        expected_accuracy = 0.0 if total == 0 else correct / total
        _require_finite(self.overall_accuracy, "overall_accuracy")
        if float(self.overall_accuracy) < 0.0 or float(self.overall_accuracy) > 1.0:
            raise TransferHarnessError("overall_accuracy must be in [0, 1].")
        if abs(float(self.overall_accuracy) - expected_accuracy) > 1e-12:
            raise TransferHarnessError("overall_accuracy inconsistent with slice totals.")
        required = tuple(self.required_slices)
        missing = tuple(slice_name for slice_name in required if slice_name not in {item.slice_name for item in slices})
        if missing and not self.allow_small_slices:
            raise TransferHarnessError(f"missing required slice(s): {missing}")
        failed_required = tuple(item.slice_name for item in slices if item.slice_name in required and not item.passed)
        if self.passed is True and failed_required:
            raise TransferHarnessError("passed=True while a required slice failed.")
        if self.passed is True and missing and not self.allow_small_slices:
            raise TransferHarnessError("passed=True while a required slice is missing.")
        passed = not failed_required and (not missing or self.allow_small_slices)
        failures = tuple(str(item) for item in self.failure_reasons)
        if self.passed is True and failures:
            raise TransferHarnessError("passed=True cannot include failure_reasons.")
        object.__setattr__(self, "slice_results", slices)
        object.__setattr__(self, "overall_accuracy", expected_accuracy)
        object.__setattr__(self, "passed", passed)
        object.__setattr__(self, "failure_reasons", () if passed else (failures or tuple(f"slice_failed:{name}" for name in failed_required) + tuple(f"missing_slice:{name}" for name in missing)))
        object.__setattr__(self, "required_slices", required)
        expected = _payload_hash(self, "report_hash")
        if not self.report_hash:
            object.__setattr__(self, "report_hash", expected)
        elif self.report_hash != expected:
            raise TransferHarnessError("report_hash does not match transfer evaluation report payload.")


def evaluate_transfer_slices(
    examples: Sequence[TransferExample],
    config: TransferHarnessConfig | None = None,
) -> TransferEvaluationReport:
    """Evaluate precomputed completions by transfer slice without model calls."""

    cfg = config or TransferHarnessConfig()
    materialized = tuple(examples)
    grouped: dict[str, list[TransferExample]] = {name: [] for name in TRANSFER_SLICE_NAMES}
    for example in materialized:
        if not isinstance(example, TransferExample):
            raise TransferHarnessError("examples must be TransferExample instances.")
        grouped[example.split].append(example)
    slice_results: list[TransferSliceResult] = []
    report_failures: list[str] = []
    for slice_name in sorted(set(cfg.required_slices) | {example.split for example in materialized}):
        items = tuple(grouped.get(slice_name, ()))
        baseline = float(cfg.baseline_accuracies.get(slice_name, 1.0))
        threshold = float(cfg.gap_thresholds.get(slice_name, DEFAULT_GAP_THRESHOLDS[slice_name]))
        failures: list[str] = []
        correct = 0
        for example in items:
            result = score_completion(example.completion, example.expected_answer, policy=_format_policy(example))
            if result.correct:
                correct += 1
                continue
            kind = _failure_kind(result.error)
            failures.append(f"{example.problem_id}:{kind}:{result.error or 'wrong answer'}")
        accuracy = 0.0 if not items else correct / len(items)
        gap = max(0.0, baseline - accuracy)
        if len(items) < cfg.min_examples_per_slice and not cfg.allow_small_slices:
            failures.append(f"{slice_name}:insufficient_examples:{len(items)}<{cfg.min_examples_per_slice}")
        if gap > threshold + 1e-12:
            failures.append(f"{slice_name}:transfer_gap:{gap:.12g}>{threshold:.12g}")
        passed = not failures
        if not passed and slice_name in cfg.required_slices:
            report_failures.append(f"slice_failed:{slice_name}")
        slice_results.append(
            TransferSliceResult(
                slice_name=slice_name,
                total=len(items),
                correct=correct,
                accuracy=accuracy,
                baseline_accuracy=baseline,
                transfer_gap=gap,
                passed=passed,
                failures=tuple(failures),
                example_ids=tuple(item.problem_id for item in items),
            )
        )
    total = sum(item.total for item in slice_results)
    overall = 0.0 if total == 0 else sum(item.correct for item in slice_results) / total
    return TransferEvaluationReport(
        config_hash=cfg.config_hash,
        total_examples=total,
        slice_results=tuple(slice_results),
        overall_accuracy=overall,
        passed=not report_failures,
        failure_reasons=tuple(report_failures),
        required_slices=cfg.required_slices,
        allow_small_slices=cfg.allow_small_slices,
    )


def _format_policy(example: TransferExample) -> FormatPolicy:
    metadata = example.metadata
    return FormatPolicy(
        answer_type=AnswerType(example.answer_type),
        answer_min=int(metadata.get("answer_min", 0)),
        answer_max=int(metadata.get("answer_max", 99999)),
        preserve_leading_zeros=bool(metadata.get("preserve_leading_zeros", False)),
        decimal_places=metadata.get("decimal_places"),
        allow_whitespace=bool(metadata.get("allow_whitespace", False)),
        preserve_case=bool(metadata.get("preserve_case", True)),
    )


def _failure_kind(error: str | None) -> str:
    text = (error or "").lower()
    if "boxed" in text or "box" in text or "extract" in text or "truncated" in text:
        return "extraction_error"
    if "invalid" in text or "unsupported" in text or "outside allowed" in text or "whitespace" in text:
        return "format_error"
    return "wrong_answer"


def _normalize_float_mapping(mapping: Mapping[str, float], field_name: str) -> dict[str, float]:
    if not isinstance(mapping, Mapping):
        raise TransferHarnessError(f"{field_name} must be a mapping.")
    result: dict[str, float] = {}
    for key, value in sorted(mapping.items()):
        if key not in TRANSFER_SLICE_NAMES:
            raise TransferHarnessError(f"unsupported {field_name} slice: {key!r}.")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TransferHarnessError(f"{field_name} values must be numeric.")
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            raise TransferHarnessError(f"{field_name} values must be finite.")
        if field_name == "baseline_accuracies" and (number < 0.0 or number > 1.0):
            raise TransferHarnessError("baseline accuracies must be in [0, 1].")
        if field_name == "gap_thresholds" and number < 0.0:
            raise TransferHarnessError("gap thresholds must be non-negative.")
        result[str(key)] = number
    return result


def _require_finite(value: Any, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or math.isnan(float(value)) or math.isinf(float(value)):
        raise TransferHarnessError(f"{field_name} must be finite.")


def _payload_hash(instance: object, hash_field: str) -> str:
    return stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})


__all__ = [
    "DEFAULT_GAP_THRESHOLDS",
    "TRANSFER_SLICE_NAMES",
    "TransferEvaluationReport",
    "TransferExample",
    "TransferHarnessConfig",
    "TransferHarnessError",
    "TransferSliceResult",
    "evaluate_transfer_slices",
]
