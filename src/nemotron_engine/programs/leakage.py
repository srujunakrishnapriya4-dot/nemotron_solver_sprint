"""Conservative leakage checks for proof-spine artifacts."""

from __future__ import annotations

import re

from nemotron_engine.core.schemas import LeakageReport


def detect_target_leakage(
    raw_prompt: str,
    target_answer: str | None,
    candidate_text: str | None = None,
) -> LeakageReport:
    answer = str(target_answer or "").strip()
    if not answer:
        return LeakageReport(False, (), ())

    leakage_types: list[str] = []
    details: list[str] = []
    prompt = str(raw_prompt or "")
    candidate = str(candidate_text or "")

    if _suspicious_contains(prompt, answer, require_context=len(answer) == 1):
        leakage_types.append("prompt_target_answer")
        details.append("Target answer appears in prompt with suspicious context.")

    pre_apply = re.split(r"\bAPPLY\b|\bapplication\b", candidate, maxsplit=1, flags=re.I)[0]
    if _suspicious_contains(pre_apply, answer, require_context=len(answer) == 1):
        leakage_types.append("candidate_pre_apply_target_answer")
        details.append("Target answer appears in candidate text before APPLY/application.")
    if re.search(r"answer\s+is\s+" + re.escape(answer) + r"\b", pre_apply, flags=re.I):
        leakage_types.append("suspicious_answer_pattern")
        details.append("Suspicious answer-is pattern before verification/application.")

    return LeakageReport(bool(leakage_types), tuple(sorted(set(leakage_types))), tuple(details))


def _suspicious_contains(text: str, answer: str, *, require_context: bool) -> bool:
    if not text:
        return False
    if require_context:
        return bool(re.search(r"(answer|target|final)\s+(?:is|=|:)?\s*" + re.escape(answer) + r"\b", text, re.I))
    return answer in text


__all__ = ["detect_target_leakage"]
