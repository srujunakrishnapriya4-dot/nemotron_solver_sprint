"""Bridge captured completions into the Pass 8 transfer harness."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping, Sequence

from nemotron_engine.core.schemas import stable_hash
from nemotron_engine.evaluation.transfer_harness import (
    TransferEvaluationReport,
    TransferExample,
    TransferHarnessConfig,
    evaluate_transfer_slices,
)

from .completion_capture import CompletionCaptureError, CompletionCaptureReport, validate_completion_records
from .inference_contracts import _reject_forbidden_metadata
from .prompt_batch import PromptBatch, validate_prompt_batch


class CompletionTransferError(ValueError):
    """Raised when completion transfer evaluation evidence is invalid."""


@dataclass(frozen=True)
class CompletionTransferConfig:
    transfer_harness_config: TransferHarnessConfig | None = None
    non_submission_exact: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    config_hash: str = ""

    def __post_init__(self) -> None:
        if self.transfer_harness_config is not None and not isinstance(self.transfer_harness_config, TransferHarnessConfig):
            raise CompletionTransferError("transfer_harness_config must be a TransferHarnessConfig.")
        if not isinstance(self.non_submission_exact, bool):
            raise CompletionTransferError("non_submission_exact must be boolean.")
        metadata = _safe_metadata(self.metadata)
        object.__setattr__(self, "metadata", metadata)
        expected = _payload_hash(self, "config_hash")
        if not self.config_hash:
            object.__setattr__(self, "config_hash", expected)
        elif self.config_hash != expected:
            raise CompletionTransferError("config_hash does not match completion transfer config payload.")


@dataclass(frozen=True)
class CompletionTransferReport:
    prompt_batch_hash: str
    completion_capture_hash: str
    transfer_report_hash: str
    transfer_report: TransferEvaluationReport
    transfer_examples: tuple[TransferExample, ...]
    passed: bool
    non_submission_exact: bool = False
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    report_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("prompt_batch_hash", "completion_capture_hash", "transfer_report_hash"):
            _require_non_empty(getattr(self, name), name)
        if not isinstance(self.transfer_report, TransferEvaluationReport):
            raise CompletionTransferError("transfer_report must be a TransferEvaluationReport.")
        if self.transfer_report_hash != self.transfer_report.report_hash:
            raise CompletionTransferError("transfer_report_hash mismatch.")
        examples = tuple(self.transfer_examples)
        if not all(isinstance(item, TransferExample) for item in examples):
            raise CompletionTransferError("transfer_examples must contain TransferExample values.")
        if not isinstance(self.passed, bool) or not isinstance(self.non_submission_exact, bool):
            raise CompletionTransferError("passed and non_submission_exact must be booleans.")
        errors = tuple(str(item) for item in self.errors)
        if self.passed and self.non_submission_exact:
            raise CompletionTransferError("passed=True is blocked for non-submission-exact evaluation.")
        if self.passed and self.transfer_report.passed is not True:
            raise CompletionTransferError("passed=True requires passed Pass 8 transfer report.")
        if self.passed and errors:
            raise CompletionTransferError("passed=True cannot include errors.")
        object.__setattr__(self, "transfer_examples", examples)
        object.__setattr__(self, "errors", errors)
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))
        object.__setattr__(self, "metadata", _safe_metadata(self.metadata))
        expected = _payload_hash(self, "report_hash")
        if not self.report_hash:
            object.__setattr__(self, "report_hash", expected)
        elif self.report_hash != expected:
            raise CompletionTransferError("report_hash does not match completion transfer report payload.")


def build_transfer_examples_from_completions(
    prompt_batch: PromptBatch,
    capture_report: CompletionCaptureReport,
    expected_answers: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> tuple[TransferExample, ...]:
    batch = validate_prompt_batch(prompt_batch)
    try:
        capture = validate_completion_records(capture_report, prompt_batch=batch)
    except CompletionCaptureError as exc:
        raise CompletionTransferError(str(exc)) from exc
    expected = _expected_answer_mapping(expected_answers)
    prompt_ids = {item.problem_id for item in batch.examples}
    record_ids = {item.problem_id for item in capture.records}
    if prompt_ids != record_ids:
        raise CompletionTransferError("completion record IDs must match prompt batch IDs.")
    missing_expected = tuple(sorted(prompt_ids - set(expected)))
    if missing_expected:
        raise CompletionTransferError(f"missing expected answer for problem(s): {missing_expected}")
    extra_expected = tuple(sorted(set(expected) - prompt_ids))
    if extra_expected:
        raise CompletionTransferError(f"extra expected answer for unknown problem(s): {extra_expected}")
    records = {item.problem_id: item for item in capture.records}
    examples: list[TransferExample] = []
    for prompt in sorted(batch.examples, key=lambda item: item.problem_id):
        record = records[prompt.problem_id]
        examples.append(
            TransferExample(
                problem_id=prompt.problem_id,
                family_id=prompt.family_id,
                primitive_family_id=prompt.primitive_family_id,
                format_family_id=prompt.format_family_id,
                split=prompt.split,
                expected_answer=expected[prompt.problem_id],
                completion=record.completion_text,
                answer_type=prompt.answer_type,
                metadata=dict(prompt.metadata),
            )
        )
    return tuple(examples)


def evaluate_completions_with_transfer_harness(
    prompt_batch: PromptBatch,
    capture_report: CompletionCaptureReport,
    expected_answers: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    config: CompletionTransferConfig | None = None,
) -> CompletionTransferReport:
    cfg = config or CompletionTransferConfig()
    examples = build_transfer_examples_from_completions(prompt_batch, capture_report, expected_answers)
    transfer_report = evaluate_transfer_slices(examples, cfg.transfer_harness_config)
    passed = transfer_report.passed is True and not cfg.non_submission_exact
    errors: tuple[str, ...] = ()
    if transfer_report.passed is not True:
        errors = tuple(transfer_report.failure_reasons or ("transfer_failed",))
    elif cfg.non_submission_exact:
        errors = ("non_submission_exact",)
    return CompletionTransferReport(
        prompt_batch_hash=prompt_batch.batch_hash,
        completion_capture_hash=capture_report.report_hash,
        transfer_report_hash=transfer_report.report_hash,
        transfer_report=transfer_report,
        transfer_examples=examples,
        passed=passed,
        non_submission_exact=cfg.non_submission_exact,
        errors=errors,
        metadata=dict(cfg.metadata),
    )


def _expected_answer_mapping(source: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return {str(key): value for key, value in source.items()}
    result: dict[str, Any] = {}
    for item in source:
        if not isinstance(item, Mapping):
            raise CompletionTransferError("expected answer records must be mappings.")
        problem_id = item.get("problem_id")
        if not isinstance(problem_id, str) or not problem_id.strip():
            raise CompletionTransferError("expected answer record requires problem_id.")
        if "expected_answer" not in item:
            raise CompletionTransferError("expected answer record requires expected_answer.")
        if problem_id in result:
            raise CompletionTransferError("duplicate expected answer problem_id.")
        result[problem_id] = item["expected_answer"]
    return result


def _safe_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CompletionTransferError("metadata must be a mapping.")
    metadata = dict(value)
    try:
        _reject_forbidden_metadata(metadata)
    except Exception as exc:
        raise CompletionTransferError(str(exc)) from exc
    return metadata


def _require_non_empty(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CompletionTransferError(f"{field_name} must be non-empty.")
    return value


def _payload_hash(instance: object, hash_field: str) -> str:
    return stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})


__all__ = [
    "CompletionTransferConfig",
    "CompletionTransferError",
    "CompletionTransferReport",
    "build_transfer_examples_from_completions",
    "evaluate_completions_with_transfer_harness",
]
