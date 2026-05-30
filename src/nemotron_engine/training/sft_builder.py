"""Build strict in-memory SFT rows from audited Pass 5 traces."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping, Sequence

from nemotron_engine.core.registry import split_allows_sft
from nemotron_engine.core.schemas import ImmutableProblemRecord, SplitName, stable_hash
from nemotron_engine.scoring.answer_extractor import extract_boxed_answer, find_boxed_spans
from nemotron_engine.traces.trace_auditor import audit_trace
from nemotron_engine.traces.trace_compiler import TraceAuditReport, TraceRecord


class SFTBuildError(ValueError):
    """Raised when a trace cannot become an SFT dataset row."""


@dataclass(frozen=True)
class SFTDatasetRow:
    row_id: str
    problem_id: str
    prompt: str
    completion: str
    trace_id: str
    trace_hash: str
    source_proof_hash: str
    source_program_hash: str
    split: str | None = None
    family_id: str | None = None
    primitive_family_id: str | None = None
    format_family_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    row_hash: str = ""

    def __post_init__(self) -> None:
        for name in (
            "row_id",
            "problem_id",
            "prompt",
            "completion",
            "trace_id",
            "trace_hash",
            "source_proof_hash",
            "source_program_hash",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise SFTBuildError(f"{name} must be a non-empty string.")
        object.__setattr__(self, "metadata", dict(self.metadata))
        expected_id = make_sft_row_id(
            problem_id=self.problem_id,
            prompt=self.prompt,
            completion=self.completion,
            trace_id=self.trace_id,
            trace_hash=self.trace_hash,
            split=self.split,
            family_id=self.family_id,
            primitive_family_id=self.primitive_family_id,
            format_family_id=self.format_family_id,
            metadata=self.metadata,
        )
        if self.row_id != expected_id:
            raise SFTBuildError("row_id does not match deterministic row payload.")
        expected_hash = _row_payload_hash(self)
        if not self.row_hash:
            object.__setattr__(self, "row_hash", expected_hash)
        elif self.row_hash != expected_hash:
            raise SFTBuildError("row_hash does not match row payload.")


@dataclass(frozen=True)
class SFTBuildReport:
    input_count: int
    accepted_count: int
    rejected_count: int
    rejection_reasons: Mapping[str, int] = field(default_factory=dict)
    accepted_row_ids: tuple[str, ...] = ()
    rejected_trace_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if min(self.input_count, self.accepted_count, self.rejected_count) < 0:
            raise SFTBuildError("SFT build counts must be non-negative.")
        if self.accepted_count + self.rejected_count != self.input_count:
            raise SFTBuildError("accepted_count + rejected_count must equal input_count.")
        if len(self.accepted_row_ids) != self.accepted_count:
            raise SFTBuildError("accepted_row_ids count must match accepted_count.")
        if len(self.rejected_trace_ids) != self.rejected_count:
            raise SFTBuildError("rejected_trace_ids count must match rejected_count.")
        object.__setattr__(self, "rejection_reasons", dict(self.rejection_reasons))
        if sum(self.rejection_reasons.values()) != self.rejected_count:
            raise SFTBuildError("rejection reason totals must match rejected_count.")
        object.__setattr__(self, "accepted_row_ids", tuple(str(item) for item in self.accepted_row_ids))
        object.__setattr__(self, "rejected_trace_ids", tuple(str(item) for item in self.rejected_trace_ids))


def build_sft_row(
    trace: TraceRecord,
    *,
    audit: TraceAuditReport | None = None,
    problem_record: ImmutableProblemRecord | None = None,
    prompt: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> SFTDatasetRow:
    if not isinstance(trace, TraceRecord):
        raise SFTBuildError("trace must be a TraceRecord.")
    _require_sft_trace(trace, audit)
    chosen_prompt = _resolve_prompt(trace, prompt)
    split = family_id = primitive_family_id = format_family_id = None
    if problem_record is not None:
        _require_sft_record(trace, problem_record)
        split = problem_record.split.value
        family_id = problem_record.family_id
        primitive_family_id = problem_record.primitive_family_id
        format_family_id = problem_record.format_family_id
    row_metadata = dict(metadata or {})
    row_metadata.setdefault("trace_type", trace.trace_type.value)
    row_id = make_sft_row_id(
        problem_id=trace.problem_id,
        prompt=chosen_prompt,
        completion=trace.completion,
        trace_id=trace.trace_id,
        trace_hash=trace.trace_hash,
        split=split,
        family_id=family_id,
        primitive_family_id=primitive_family_id,
        format_family_id=format_family_id,
        metadata=row_metadata,
    )
    return SFTDatasetRow(
        row_id=row_id,
        problem_id=trace.problem_id,
        prompt=chosen_prompt,
        completion=trace.completion,
        trace_id=trace.trace_id,
        trace_hash=trace.trace_hash,
        source_proof_hash=str(trace.source_proof_hash),
        source_program_hash=str(trace.source_program_hash),
        split=split,
        family_id=family_id,
        primitive_family_id=primitive_family_id,
        format_family_id=format_family_id,
        metadata=row_metadata,
    )


def build_sft_dataset(
    traces: Sequence[TraceRecord],
    *,
    audits: Mapping[str, TraceAuditReport] | None = None,
    problem_records: Mapping[str, ImmutableProblemRecord] | None = None,
    prompts: Mapping[str, str] | None = None,
) -> tuple[tuple[SFTDatasetRow, ...], SFTBuildReport]:
    rows: list[SFTDatasetRow] = []
    rejection_reasons: dict[str, int] = {}
    rejected_trace_ids: list[str] = []
    materialized = tuple(traces)
    for index, trace in enumerate(materialized):
        try:
            row = build_sft_row(
                trace,
                audit=(audits or {}).get(trace.trace_id),
                problem_record=(problem_records or {}).get(trace.problem_id),
                prompt=(prompts or {}).get(trace.problem_id),
            )
        except SFTBuildError as exc:
            reason = _reason_key(str(exc))
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            rejected_trace_ids.append(getattr(trace, "trace_id", f"trace_index_{index}"))
            continue
        rows.append(row)
    report = SFTBuildReport(
        input_count=len(materialized),
        accepted_count=len(rows),
        rejected_count=len(rejected_trace_ids),
        rejection_reasons=rejection_reasons,
        accepted_row_ids=tuple(row.row_id for row in rows),
        rejected_trace_ids=tuple(rejected_trace_ids),
    )
    return tuple(rows), report


def make_sft_row_id(
    *,
    problem_id: str,
    prompt: str,
    completion: str,
    trace_id: str,
    trace_hash: str,
    split: str | None,
    family_id: str | None,
    primitive_family_id: str | None,
    format_family_id: str | None,
    metadata: Mapping[str, Any],
) -> str:
    return "sft-" + stable_hash(
        {
            "problem_id": problem_id,
            "prompt": prompt,
            "completion": completion,
            "trace_id": trace_id,
            "trace_hash": trace_hash,
            "split": split,
            "family_id": family_id,
            "primitive_family_id": primitive_family_id,
            "format_family_id": format_family_id,
            "metadata": dict(metadata),
        }
    )[:24]


def _require_sft_trace(trace: TraceRecord, audit: TraceAuditReport | None) -> None:
    if trace.is_positive is not True:
        raise SFTBuildError("trace_not_positive")
    if _is_negative_metadata(trace.metadata):
        raise SFTBuildError("trace_marked_negative")
    if len(find_boxed_spans(trace.completion)) != 1:
        raise SFTBuildError("trace_requires_one_box")
    if not trace.source_proof_hash:
        raise SFTBuildError("missing_source_proof_hash")
    if not trace.source_program_hash:
        raise SFTBuildError("missing_source_program_hash")
    if not trace.boxed_answer:
        raise SFTBuildError("missing_boxed_answer")
    if not trace.target_output:
        raise SFTBuildError("missing_target_output")
    if trace.boxed_answer != trace.target_output:
        raise SFTBuildError("boxed_answer_target_mismatch")
    if extract_boxed_answer(trace.completion).raw != trace.boxed_answer:
        raise SFTBuildError("completion_box_mismatch")
    if _has_unsafe_trace_metadata(trace.metadata):
        raise SFTBuildError("unsafe_trace_metadata")
    report = audit if audit is not None else audit_trace(trace)
    if report.trace_id != trace.trace_id:
        raise SFTBuildError("audit_trace_id_mismatch")
    if report.valid is not True:
        raise SFTBuildError("invalid_audit")


def _require_sft_record(trace: TraceRecord, record: ImmutableProblemRecord) -> None:
    if not isinstance(record, ImmutableProblemRecord):
        raise SFTBuildError("problem_record_must_be_immutable")
    if record.problem_id != trace.problem_id:
        raise SFTBuildError("problem_record_id_mismatch")
    if record.split is not SplitName.TRAIN:
        raise SFTBuildError("problem_record_not_train")
    if record.allowed_for_sft is not True:
        raise SFTBuildError("problem_record_sft_not_allowed")
    if record.contamination_flags:
        raise SFTBuildError("problem_record_contaminated")
    if split_allows_sft(record) is not True:
        raise SFTBuildError("split_does_not_allow_sft")


def _resolve_prompt(trace: TraceRecord, prompt: str | None) -> str:
    selected = prompt if prompt is not None else trace.prompt
    if not isinstance(selected, str) or not selected.strip():
        raise SFTBuildError("missing_prompt")
    return selected


def _is_negative_metadata(metadata: Mapping[str, Any]) -> bool:
    return bool(metadata.get("negative_type") or metadata.get("is_negative") is True)


def _has_unsafe_trace_metadata(metadata: Mapping[str, Any]) -> bool:
    flags = metadata.get("contamination_flags")
    if flags:
        return True
    unsafe_true = ("has_leakage", "leakage", "ambiguous", "invalid_format")
    if any(metadata.get(key) is True for key in unsafe_true):
        return True
    if metadata.get("format_valid") is False:
        return True
    return False


def _row_payload_hash(row: SFTDatasetRow) -> str:
    return stable_hash({item.name: getattr(row, item.name) for item in fields(SFTDatasetRow) if item.name != "row_hash"})


def _reason_key(message: str) -> str:
    return message.split(":", 1)[0].strip() or "rejected"


__all__ = [
    "SFTBuildError",
    "SFTBuildReport",
    "SFTDatasetRow",
    "build_sft_dataset",
    "build_sft_row",
    "make_sft_row_id",
]
