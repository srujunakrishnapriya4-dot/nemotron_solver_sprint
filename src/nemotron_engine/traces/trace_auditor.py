"""Audit Pass 5 traces against proof artifacts."""

from __future__ import annotations

import re
from typing import Any

from nemotron_engine.core.schemas import ExecutableProof, stable_hash
from nemotron_engine.programs.leakage import detect_target_leakage
from nemotron_engine.scoring.answer_extractor import AnswerExtractionError, extract_boxed_answer, find_boxed_spans

from .trace_compiler import TraceAuditReport, TraceRecord, TraceType


class TraceAuditError(ValueError):
    """Raised when a trace audit must fail loudly."""


def audit_trace(
    trace: TraceRecord,
    *,
    proof: ExecutableProof | None = None,
    max_chars: int = 1200,
    max_lines: int = 30,
    max_boilerplate_ratio: float = 0.65,
) -> TraceAuditReport:
    errors: list[str] = []
    warnings: list[str] = []
    completion = trace.completion
    lines = completion.splitlines()
    spans = find_boxed_spans(completion)
    metrics: dict[str, Any] = {
        "char_count": len(completion),
        "line_count": len(lines),
        "box_count": len(spans),
        "boilerplate_ratio": _boilerplate_ratio(lines),
    }

    if len(completion) > max_chars:
        errors.append("trace_too_long")
    if len(lines) > max_lines:
        errors.append("too_many_lines")
    if metrics["boilerplate_ratio"] > max_boilerplate_ratio:
        errors.append("excessive_boilerplate")

    if trace.is_positive:
        if trace.trace_type is TraceType.HARD_REJECTION:
            errors.append("positive_trace_has_hard_rejection_type")
        _audit_positive_box_policy(trace, proof, spans, errors)
        _audit_application_marker(trace, errors)
        if proof is not None:
            _audit_proof_agreement(trace, proof, errors)
    else:
        if trace.trace_type is not TraceType.HARD_REJECTION:
            errors.append("non_positive_trace_not_hard_rejection")
        _audit_rejection_policy(trace, spans, errors)

    if proof is not None and _claims_verification(completion):
        if not proof.example_executions or any(item.passed is not True for item in proof.example_executions):
            errors.append("fake_verification_claim")
    if proof is None and trace.is_positive and _claims_verification(completion):
        warnings.append("verification_claim_without_proof")
    if not trace.is_positive and _claims_verification(completion):
        errors.append("rejection_claims_verification")

    return TraceAuditReport(trace.trace_id, valid=not errors, errors=tuple(errors), warnings=tuple(warnings), metrics=metrics)


def assert_trace_valid(
    trace: TraceRecord,
    *,
    proof: ExecutableProof | None = None,
    max_chars: int = 1200,
    max_lines: int = 30,
    max_boilerplate_ratio: float = 0.65,
) -> TraceAuditReport:
    report = audit_trace(
        trace,
        proof=proof,
        max_chars=max_chars,
        max_lines=max_lines,
        max_boilerplate_ratio=max_boilerplate_ratio,
    )
    if not report.valid:
        raise TraceAuditError(";".join(report.errors))
    return report


def _audit_positive_box_policy(
    trace: TraceRecord,
    proof: ExecutableProof | None,
    spans,
    errors: list[str],
) -> None:
    if len(spans) != 1:
        errors.append("positive_requires_exactly_one_box")
        return
    try:
        boxed = extract_boxed_answer(trace.completion).raw
    except AnswerExtractionError:
        errors.append("invalid_boxed_answer")
        return
    if boxed == "":
        errors.append("empty_boxed_answer")
    if trace.boxed_answer != boxed:
        errors.append("boxed_answer_mismatch")
    if trace.target_output is not None and boxed != trace.target_output:
        errors.append("target_output_mismatch")
    if proof is not None and proof.target_execution is not None and boxed != proof.target_execution.output_value:
        errors.append("proof_target_output_mismatch")


def _audit_proof_agreement(trace: TraceRecord, proof: ExecutableProof, errors: list[str]) -> None:
    if trace.source_proof_hash != stable_hash(proof):
        errors.append("source_proof_hash_mismatch")
    if trace.source_program_hash != proof.locked_program.program_hash:
        errors.append("source_program_hash_mismatch")
    if trace.is_positive and proof.is_training_safe is not True:
        errors.append("proof_not_training_safe")
    if trace.is_positive:
        if proof.leakage_report.has_leakage:
            errors.append("proof_leakage")
        if proof.ambiguity_report.ambiguous or not proof.ambiguity_report.unique_target_output:
            errors.append("proof_ambiguous")
        if proof.format_report.valid is not True:
            errors.append("proof_invalid_format")
        if any(item.passed is not True for item in proof.example_executions):
            errors.append("proof_example_failed")


def _audit_application_marker(trace: TraceRecord, errors: list[str]) -> None:
    answer = str(trace.boxed_answer or trace.target_output or "")
    marker = _first_application_marker(trace.completion)
    if marker is None:
        errors.append("missing_application_marker")
        return
    before = trace.completion[: marker.start()]
    if answer and answer in before:
        errors.append("answer_leakage_before_application")
    leakage = detect_target_leakage("", answer, before)
    if leakage.has_leakage:
        errors.append("target_leakage_before_application")


def _audit_rejection_policy(trace: TraceRecord, spans, errors: list[str]) -> None:
    if trace.trace_type is TraceType.HARD_REJECTION:
        if trace.is_positive:
            errors.append("rejection_trace_marked_positive")
        if trace.boxed_answer is not None:
            errors.append("rejection_trace_has_boxed_answer")
        if spans:
            errors.append("rejection_trace_contains_box")
        reason = trace.metadata.get("reason")
        has_reason = isinstance(reason, str) and bool(reason.strip())
        has_reason_line = bool(re.search(r"^Reason:\s*\S+", trace.completion, flags=re.I | re.M))
        if not has_reason and not has_reason_line:
            errors.append("rejection_trace_missing_reason")


def _first_application_marker(text: str):
    return re.search(r"\b(Target application|Applied|Apply)\b", text)


def _claims_verification(text: str) -> bool:
    return bool(re.search(r"\b(verified|passed examples|proof passed|verified on examples)\b", text, flags=re.I))


def _boilerplate_ratio(lines: list[str]) -> float:
    normalized = [" ".join(line.lower().split()) for line in lines if line.strip()]
    if not normalized:
        return 1.0
    counts = {line: normalized.count(line) for line in set(normalized)}
    return max(counts.values()) / len(normalized)


__all__ = ["TraceAuditError", "assert_trace_valid", "audit_trace"]
