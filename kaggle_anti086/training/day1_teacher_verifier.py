from __future__ import annotations

import re


_BOX_TOKEN = r"\boxed{"


def count_boxed_answers(text: str) -> int:
    return text.count(_BOX_TOKEN)


def extract_boxed_answer(text: str) -> str | None:
    if count_boxed_answers(text) != 1:
        return None
    start = text.find(_BOX_TOKEN)
    if start < 0:
        return None
    content_start = start + len(_BOX_TOKEN)
    depth = 1
    index = content_start
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                if index != len(text) - 1:
                    return None
                answer = text[content_start:index]
                return answer if answer else None
        index += 1
    return None


def normalize_answer_for_teacher(answer: str) -> str:
    text = str(answer).strip()
    if re.fullmatch(r"[+-]?\d+", text):
        return str(int(text))
    return text


def validate_target_text(target_text: str, expected_answer: str) -> tuple[bool, str | None]:
    if not isinstance(target_text, str) or not target_text.strip():
        return False, "target_text_empty"
    if "ABSTAIN" in target_text.upper():
        return False, "target_text_contains_abstain"
    if count_boxed_answers(target_text) != 1:
        return False, "target_text_must_contain_exactly_one_box"
    if not target_text.endswith("}"):
        return False, "target_text_has_text_after_final_box"
    boxed = extract_boxed_answer(target_text)
    if boxed is None:
        return False, "boxed_answer_malformed_or_empty"
    if "\n" in boxed or len(boxed.split()) > 8:
        return False, "boxed_answer_contains_explanation"
    if normalize_answer_for_teacher(boxed) != normalize_answer_for_teacher(expected_answer):
        return False, "boxed_answer_mismatch"
    return True, None

