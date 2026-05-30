"""Deterministic short trace templates for Pass 5."""

from __future__ import annotations


def render_direct_answer_trace(*, answer: str) -> str:
    _require_non_empty(answer, "answer")
    return "\n".join(
        (
            "Rule verified.",
            "Apply to target.",
            rf"\boxed{{{answer}}}",
        )
    )


def render_minimal_known_rule_trace(
    *,
    rule_summary: str,
    example_checks: str,
    target_application: str,
    answer: str,
) -> str:
    _require_non_empty(answer, "answer")
    _reject_pre_application_answer(answer, rule_summary, "rule_summary")
    _reject_pre_application_answer(answer, example_checks, "example_checks")
    _reject_pre_application_answer(answer, target_application, "target_application")
    return "\n".join(
        (
            f"Observed rule: {_clean(rule_summary)}",
            f"Verified on examples: {_clean(example_checks)}",
            f"Applied to target: {_clean(target_application)}",
            rf"\boxed{{{answer}}}",
        )
    )


def render_minimal_induced_rule_trace(
    *,
    rule_summary: str,
    example_checks: str,
    target_application: str,
    answer: str,
) -> str:
    _require_non_empty(answer, "answer")
    _reject_pre_application_answer(answer, rule_summary, "rule_summary")
    _reject_pre_application_answer(answer, example_checks, "example_checks")
    _reject_pre_application_answer(answer, target_application, "target_application")
    return "\n".join(
        (
            f"Candidate rule: {_clean(rule_summary)}",
            f"Example checks: {_clean(example_checks)}",
            f"Target application: {_clean(target_application)}",
            rf"\boxed{{{answer}}}",
        )
    )


def render_hard_rejection_trace(
    *,
    reason: str,
    candidate: str = "unavailable",
    allow_boxed_answer: bool = False,
    boxed_answer: str | None = None,
) -> str:
    _require_non_empty(reason, "reason")
    lines = (
        f"Candidate rejected: {_clean(candidate)}",
        f"Reason: {_clean(reason)}",
        "No final training answer.",
    )
    if allow_boxed_answer:
        if boxed_answer is None or not str(boxed_answer).strip():
            raise ValueError("boxed_answer is required when allow_boxed_answer=True.")
        return "\n".join((*lines, rf"\boxed{{{boxed_answer}}}"))
    return "\n".join(lines)


def _clean(value: str) -> str:
    text = " ".join(str(value).strip().split())
    return text or "unavailable"


def _require_non_empty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


def _reject_pre_application_answer(answer: str, value: str, field_name: str) -> None:
    if str(answer) in str(value):
        raise ValueError(f"{field_name} must not contain the target answer before application.")


__all__ = [
    "render_direct_answer_trace",
    "render_hard_rejection_trace",
    "render_minimal_induced_rule_trace",
    "render_minimal_known_rule_trace",
]
