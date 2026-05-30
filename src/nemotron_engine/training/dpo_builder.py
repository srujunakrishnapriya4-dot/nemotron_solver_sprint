"""Build strict in-memory DPO pairs from audited Pass 5 traces."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping, Sequence

from nemotron_engine.core.schemas import ImmutableProblemRecord, SplitName, stable_hash
from nemotron_engine.traces.trace_auditor import audit_trace
from nemotron_engine.traces.trace_compiler import TraceAuditReport, TraceRecord


ALLOWED_NEGATIVE_TYPES = (
    "false_rule",
    "fake_verification",
    "decoded_vs_encoded",
    "format_box",
    "target_leak",
    "overlong_reasoning",
    "unnecessary_ambiguity",
)


class DPOBuildError(ValueError):
    """Raised when traces cannot become a DPO pair."""


@dataclass(frozen=True)
class DPOPair:
    pair_id: str
    problem_id: str
    prompt: str
    chosen: str
    rejected: str
    chosen_trace_id: str
    rejected_trace_id: str | None
    negative_type: str
    difficulty_match_score: float
    length_ratio: float
    verifier_delta: float
    source_proof_hash: str | None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    pair_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("pair_id", "problem_id", "prompt", "chosen", "rejected", "chosen_trace_id", "negative_type"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise DPOBuildError(f"{name} must be a non-empty string.")
        if self.negative_type not in ALLOWED_NEGATIVE_TYPES:
            raise DPOBuildError("unsupported_negative_type")
        if self.chosen.strip() == self.rejected.strip():
            raise DPOBuildError("chosen_rejected_identical")
        if not 0.0 <= self.difficulty_match_score <= 1.0:
            raise DPOBuildError("difficulty_match_score_out_of_range")
        if self.length_ratio <= 0.0:
            raise DPOBuildError("length_ratio_must_be_positive")
        max_ratio = 8.0 if self.negative_type == "overlong_reasoning" else 3.0
        if self.length_ratio < 0.25 or self.length_ratio > max_ratio:
            raise DPOBuildError("length_ratio_uncontrolled")
        if self.verifier_delta <= 0.0:
            raise DPOBuildError("verifier_delta_not_meaningful")
        object.__setattr__(self, "metadata", dict(self.metadata))
        expected_id = make_dpo_pair_id(
            problem_id=self.problem_id,
            prompt=self.prompt,
            chosen=self.chosen,
            rejected=self.rejected,
            chosen_trace_id=self.chosen_trace_id,
            rejected_trace_id=self.rejected_trace_id,
            negative_type=self.negative_type,
            metadata=self.metadata,
        )
        if self.pair_id != expected_id:
            raise DPOBuildError("pair_id does not match deterministic pair payload.")
        expected_hash = _pair_payload_hash(self)
        if not self.pair_hash:
            object.__setattr__(self, "pair_hash", expected_hash)
        elif self.pair_hash != expected_hash:
            raise DPOBuildError("pair_hash does not match pair payload.")


@dataclass(frozen=True)
class DPOBuildReport:
    input_positive_count: int
    input_negative_count: int
    accepted_pair_count: int
    rejected_pair_count: int
    rejection_reasons: Mapping[str, int] = field(default_factory=dict)
    negative_type_counts: Mapping[str, int] = field(default_factory=dict)
    pair_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if min(self.input_positive_count, self.input_negative_count, self.accepted_pair_count, self.rejected_pair_count) < 0:
            raise DPOBuildError("DPO build counts must be non-negative.")
        if len(self.pair_ids) != self.accepted_pair_count:
            raise DPOBuildError("pair_ids count must match accepted_pair_count.")
        object.__setattr__(self, "rejection_reasons", dict(self.rejection_reasons))
        object.__setattr__(self, "negative_type_counts", dict(self.negative_type_counts))
        if sum(self.rejection_reasons.values()) != self.rejected_pair_count:
            raise DPOBuildError("rejection reason totals must match rejected_pair_count.")
        if sum(self.negative_type_counts.values()) != self.accepted_pair_count:
            raise DPOBuildError("negative type totals must match accepted_pair_count.")
        object.__setattr__(self, "pair_ids", tuple(str(item) for item in self.pair_ids))


def build_dpo_pair(
    *,
    prompt: str,
    chosen_trace: TraceRecord,
    rejected_trace: TraceRecord,
    negative_type: str,
    chosen_audit: TraceAuditReport | None = None,
    rejected_audit: TraceAuditReport | None = None,
    metadata: dict[str, Any] | None = None,
) -> DPOPair:
    if not isinstance(prompt, str) or not prompt.strip():
        raise DPOBuildError("empty_prompt")
    if not negative_type:
        raise DPOBuildError("missing_negative_type")
    if negative_type not in ALLOWED_NEGATIVE_TYPES:
        raise DPOBuildError("unsupported_negative_type")
    if chosen_trace.problem_id != rejected_trace.problem_id:
        raise DPOBuildError("problem_id_mismatch")
    chosen_report = _require_chosen_trace(chosen_trace, chosen_audit)
    rejected_report = rejected_audit if rejected_audit is not None else audit_trace(rejected_trace)
    _require_rejected_trace(chosen_trace, rejected_trace, rejected_report, negative_type)
    ratio = _length_ratio(chosen_trace.completion, rejected_trace.completion)
    difficulty = min(1.0, min(ratio, 1.0 / ratio))
    verifier_delta = (
        1.0
        if not rejected_report.valid
        or rejected_trace.is_positive is not True
        or (negative_type == "fake_verification" and bool(rejected_report.warnings))
        else 0.0
    )
    pair_metadata = dict(metadata or {})
    pair_metadata.setdefault("chosen_audit_valid", chosen_report.valid)
    pair_metadata.setdefault("rejected_audit_valid", rejected_report.valid)
    pair_id = make_dpo_pair_id(
        problem_id=chosen_trace.problem_id,
        prompt=prompt,
        chosen=chosen_trace.completion,
        rejected=rejected_trace.completion,
        chosen_trace_id=chosen_trace.trace_id,
        rejected_trace_id=rejected_trace.trace_id,
        negative_type=negative_type,
        metadata=pair_metadata,
    )
    return DPOPair(
        pair_id=pair_id,
        problem_id=chosen_trace.problem_id,
        prompt=prompt,
        chosen=chosen_trace.completion,
        rejected=rejected_trace.completion,
        chosen_trace_id=chosen_trace.trace_id,
        rejected_trace_id=rejected_trace.trace_id,
        negative_type=negative_type,
        difficulty_match_score=difficulty,
        length_ratio=ratio,
        verifier_delta=verifier_delta,
        source_proof_hash=chosen_trace.source_proof_hash,
        metadata=pair_metadata,
    )


def build_dpo_dataset(
    positive_traces: Sequence[TraceRecord],
    negative_traces: Sequence[TraceRecord],
    *,
    positive_audits: Mapping[str, TraceAuditReport] | None = None,
    negative_audits: Mapping[str, TraceAuditReport] | None = None,
    prompts: Mapping[str, str] | None = None,
    problem_records: Mapping[str, ImmutableProblemRecord] | None = None,
    max_pairs_per_problem: int = 8,
) -> tuple[tuple[DPOPair, ...], DPOBuildReport]:
    if not isinstance(max_pairs_per_problem, int) or isinstance(max_pairs_per_problem, bool) or max_pairs_per_problem < 0:
        raise DPOBuildError("max_pairs_per_problem must be a non-negative integer.")
    positives = tuple(positive_traces)
    negatives = tuple(negative_traces)
    negative_by_problem: dict[str, list[TraceRecord]] = {}
    for trace in negatives:
        negative_by_problem.setdefault(trace.problem_id, []).append(trace)
    pairs: list[DPOPair] = []
    per_problem_counts: dict[str, int] = {}
    rejection_reasons: dict[str, int] = {}
    negative_type_counts: dict[str, int] = {}

    for chosen in positives:
        matching_negatives = negative_by_problem.get(chosen.problem_id, [])
        if not matching_negatives:
            _count(rejection_reasons, "no_matching_negative")
            continue
        record = (problem_records or {}).get(chosen.problem_id)
        if record is not None:
            try:
                _require_dpo_record(record)
            except DPOBuildError as exc:
                for _ in range(max(1, len(matching_negatives))):
                    _count(rejection_reasons, str(exc))
                continue
        prompt = (prompts or {}).get(chosen.problem_id) or chosen.prompt
        for rejected in matching_negatives:
            if per_problem_counts.get(chosen.problem_id, 0) >= max_pairs_per_problem:
                _count(rejection_reasons, "max_pairs_per_problem")
                break
            negative_type = str(rejected.metadata.get("negative_type", ""))
            try:
                pair = build_dpo_pair(
                    prompt=prompt or "",
                    chosen_trace=chosen,
                    rejected_trace=rejected,
                    negative_type=negative_type,
                    chosen_audit=(positive_audits or {}).get(chosen.trace_id),
                    rejected_audit=(negative_audits or {}).get(rejected.trace_id),
                )
            except DPOBuildError as exc:
                _count(rejection_reasons, str(exc))
                continue
            pairs.append(pair)
            per_problem_counts[chosen.problem_id] = per_problem_counts.get(chosen.problem_id, 0) + 1
            _count(negative_type_counts, pair.negative_type)

    report = DPOBuildReport(
        input_positive_count=len(positives),
        input_negative_count=len(negatives),
        accepted_pair_count=len(pairs),
        rejected_pair_count=sum(rejection_reasons.values()),
        rejection_reasons=rejection_reasons,
        negative_type_counts=negative_type_counts,
        pair_ids=tuple(pair.pair_id for pair in pairs),
    )
    return tuple(pairs), report


def make_dpo_pair_id(
    *,
    problem_id: str,
    prompt: str,
    chosen: str,
    rejected: str,
    chosen_trace_id: str,
    rejected_trace_id: str | None,
    negative_type: str,
    metadata: Mapping[str, Any],
) -> str:
    return "dpo-" + stable_hash(
        {
            "problem_id": problem_id,
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "chosen_trace_id": chosen_trace_id,
            "rejected_trace_id": rejected_trace_id,
            "negative_type": negative_type,
            "metadata": dict(metadata),
        }
    )[:24]


def _require_chosen_trace(trace: TraceRecord, audit: TraceAuditReport | None) -> TraceAuditReport:
    report = audit if audit is not None else audit_trace(trace)
    if trace.is_positive is not True:
        raise DPOBuildError("chosen_not_positive")
    if report.trace_id != trace.trace_id:
        raise DPOBuildError("chosen_audit_trace_id_mismatch")
    if report.valid is not True:
        raise DPOBuildError("chosen_audit_invalid")
    if not trace.boxed_answer or not trace.source_proof_hash or not trace.source_program_hash:
        raise DPOBuildError("chosen_missing_verified_fields")
    if _is_negative_metadata(trace.metadata):
        raise DPOBuildError("chosen_marked_negative")
    return report


def _require_rejected_trace(chosen: TraceRecord, rejected: TraceRecord, audit: TraceAuditReport, negative_type: str) -> None:
    if not isinstance(rejected.completion, str) or not rejected.completion.strip():
        raise DPOBuildError("rejected_empty")
    if chosen.completion.strip() == rejected.completion.strip():
        raise DPOBuildError("chosen_rejected_identical")
    if audit.trace_id != rejected.trace_id:
        raise DPOBuildError("rejected_audit_trace_id_mismatch")
    if _is_negative_metadata(rejected.metadata) and audit.valid and not audit.warnings and rejected.is_positive:
        raise DPOBuildError("metadata_negative_otherwise_valid_positive")
    if audit.valid and not audit.warnings and rejected.is_positive:
        raise DPOBuildError("rejected_valid_positive")
    if not _negative_type_matches(negative_type, rejected, audit):
        raise DPOBuildError("negative_type_mismatch")
    ratio = _length_ratio(chosen.completion, rejected.completion)
    max_ratio = 8.0 if negative_type == "overlong_reasoning" else 3.0
    if ratio < 0.25 or ratio > max_ratio:
        raise DPOBuildError("length_ratio_uncontrolled")


def _require_dpo_record(record: ImmutableProblemRecord) -> None:
    if not isinstance(record, ImmutableProblemRecord):
        raise DPOBuildError("problem_record_must_be_immutable")
    if record.split is not SplitName.TRAIN:
        raise DPOBuildError("problem_record_not_train")
    if record.contamination_flags:
        raise DPOBuildError("problem_record_contaminated")
    if record.allowed_for_dpo is not True:
        raise DPOBuildError("problem_record_dpo_not_allowed")


def _negative_type_matches(negative_type: str, trace: TraceRecord, audit: TraceAuditReport) -> bool:
    errors = set(audit.errors)
    if negative_type == "target_leak":
        return "answer_leakage_before_application" in errors or "target_leakage_before_application" in errors
    if negative_type == "format_box":
        return "missing_application_marker" in errors or "positive_requires_exactly_one_box" in errors
    if negative_type == "overlong_reasoning":
        return "trace_too_long" in errors or "too_many_lines" in errors or _is_negative_metadata(trace.metadata)
    if negative_type == "fake_verification":
        return "fake_verification_claim" in errors or bool(audit.warnings) or _is_negative_metadata(trace.metadata)
    return _is_negative_metadata(trace.metadata) or trace.is_positive is not True or not audit.valid


def _is_negative_metadata(metadata: Mapping[str, Any]) -> bool:
    return bool(metadata.get("negative_type") or metadata.get("is_negative") is True)


def _length_ratio(chosen: str, rejected: str) -> float:
    chosen_len = max(1, len(chosen.strip()))
    rejected_len = len(rejected.strip())
    return rejected_len / chosen_len


def _pair_payload_hash(pair: DPOPair) -> str:
    return stable_hash({item.name: getattr(pair, item.name) for item in fields(DPOPair) if item.name != "pair_hash"})


def _count(mapping: dict[str, int], key: str) -> None:
    mapping[key] = mapping.get(key, 0) + 1


__all__ = [
    "ALLOWED_NEGATIVE_TYPES",
    "DPOBuildError",
    "DPOBuildReport",
    "DPOPair",
    "build_dpo_dataset",
    "build_dpo_pair",
    "make_dpo_pair_id",
]
