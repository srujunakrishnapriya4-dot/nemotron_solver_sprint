from __future__ import annotations

import re
from typing import Any

from kaggle_anti086.solvers.answer_normalizer import normalize_gold


DIRECT_ANSWER_INSTRUCTION = (
    "You are given examples of a transformation rule. Infer the rule and answer the query.\n"
    "Respond with only the final answer. No explanation.\n\n"
)

VERBOSE_ASSISTANT_RE = re.compile(r"\b(because|therefore|explanation|reasoning)\b|\banswer\s*:|```|\n", re.IGNORECASE)
LEAK_MARKER_RE = re.compile(r"\b(expected(?:\s+answer)?|gold(?:\s+answer)?|final\s+answer|target\s+answer)\s*[:=]", re.IGNORECASE)


def build_direct_answer_messages(row: dict[str, Any], answer: str) -> list[dict[str, str]]:
    prompt = str(row.get("prompt", "")).strip()
    assistant = str(answer).strip()
    return [
        {"role": "user", "content": DIRECT_ANSWER_INSTRUCTION + prompt},
        {"role": "assistant", "content": assistant},
    ]


def validate_training_messages(messages: list[dict[str, str]], *, answer: str | None = None) -> dict[str, Any]:
    failures: list[str] = []
    if len(messages) != 2:
        failures.append("messages_must_have_user_and_assistant_only")
    roles = [message.get("role") for message in messages]
    if roles != ["user", "assistant"]:
        failures.append("messages_roles_must_be_user_assistant")
    user = str(messages[0].get("content", "")) if messages else ""
    assistant = str(messages[1].get("content", "")) if len(messages) > 1 else ""
    if not assistant.strip():
        failures.append("assistant_empty")
    if VERBOSE_ASSISTANT_RE.search(assistant):
        failures.append("assistant_verbose_or_markdown")
    if len(user) > 0 and assistant.strip() and assistant.strip() == user.strip():
        failures.append("assistant_copies_full_prompt")
    if answer is not None and _query_answer_leaked(user, str(answer)):
        failures.append("query_gold_answer_leaked_in_user_prompt")
    return {"status": "PASS" if not failures else "FAIL", "failures": failures}


def normalized_answer_for_family(answer: str, family: str) -> str:
    return normalize_gold(answer, _answer_type_for_family(family))


def _query_answer_leaked(user_prompt: str, answer: str) -> bool:
    if not answer or not user_prompt:
        return False
    normalized_answer = answer.strip()
    for line in user_prompt.splitlines():
        if LEAK_MARKER_RE.search(line) and normalized_answer in line:
            return True
    return False


def _answer_type_for_family(family: str) -> str | None:
    if family in {"numeric_formula", "gravity_numeric", "unit_conversion"}:
        return "numeric"
    if family == "bit_manipulation":
        return "binary"
    if family == "roman_numeral":
        return "roman"
    if family in {"symbol_mapping", "custom_numeral", "format_only"}:
        return "symbol"
    if family in {"word_cipher", "char_cipher"}:
        return "text_phrase"
    return None
