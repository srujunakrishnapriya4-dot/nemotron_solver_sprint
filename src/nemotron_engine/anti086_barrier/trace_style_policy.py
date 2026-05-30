from __future__ import annotations


def make_short_trace(rule_id: str, answer: str) -> str:
    return f"Rule: {rule_id}\nAnswer: {answer}"


def validate_trace_style(text: str) -> bool:
    return len(text.split()) <= 40 and "because because" not in text.lower()
