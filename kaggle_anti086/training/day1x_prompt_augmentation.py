from __future__ import annotations

from copy import deepcopy
import hashlib
import random
import re
from typing import Any

from kaggle_anti086.training.day1_teacher_trainability_gate import verify_trainable_row
from kaggle_anti086.training.day1x_composed_teachers import (
    COMPOSED_TEACHER_REGISTRY,
    validate_composed_row,
)


PUBLIC_STYLES = (
    "direct",
    "minimal",
    "examples_query",
    "table",
    "story",
    "public_style_minimal",
    "public_style_verbose",
    "public_style_table",
    "public_style_examples_then_query",
    "public_style_noisy_context",
    "public_style_symbol_heavy",
    "public_style_mixed_notation",
    "public_style_multiline",
    "public_style_plain_english",
)


def supported_public_styles() -> list[str]:
    return list(PUBLIC_STYLES)


def augment_prompt(row: dict[str, Any], style: str, seed: int) -> dict[str, Any]:
    if style not in PUBLIC_STYLES:
        raise ValueError(f"unsupported_public_style:{style}")
    original_ok, original_errors = validate_composed_row(row)
    if not original_ok:
        raise ValueError(f"original_row_not_valid:{original_errors}")
    rng = random.Random(seed + _offset(str(row.get("id", row.get("prompt", "")))) + _offset(style))
    augmented = deepcopy(row)
    prompt = _render_style(row, style, rng)
    if _prompt_key(prompt) == _prompt_key(str(row.get("prompt", ""))):
        raise ValueError("augmentation_prompt_unchanged")
    parent_id = str(row.get("id") or row.get("record_id"))
    augmented["id"] = _stable_id("aug", parent_id, style, str(seed), prompt)
    augmented["record_id"] = _stable_id("aug-record", parent_id, style, str(seed), prompt)
    augmented["parent_id"] = parent_id
    augmented["augmentation_style"] = style
    augmented["prompt_style"] = style
    augmented["prompt"] = prompt
    augmented["source"] = "day1x_public_style_prompt_augmentation"
    augmented["generator"] = f"{row.get('generator', 'unknown')}_aug_{style}_v1"
    augmented["messages"] = [{"role": "user", "content": prompt}, {"role": "assistant", "content": augmented["target_text"]}]
    ok, errors = validate_augmented_row(row, augmented)
    if not ok:
        raise ValueError(f"augmented_row_invalid:{errors}")
    return augmented


def augment_rows(rows: list[dict[str, Any]], styles: list[str], seed: int, max_per_row: int | None = None) -> list[dict[str, Any]]:
    if not styles:
        return []
    for style in styles:
        if style not in PUBLIC_STYLES:
            raise ValueError(f"unsupported_public_style:{style}")
    augmented: list[dict[str, Any]] = []
    limit = len(styles) if max_per_row is None else max(0, min(max_per_row, len(styles)))
    for row_index, row in enumerate(rows):
        for style in styles[:limit]:
            augmented.append(augment_prompt(row, style, seed + row_index * 1009))
    return augmented


def validate_augmented_row(original: dict[str, Any], augmented: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for key in ("answer", "target_text", "family", "subfamilies", "substeps"):
        if augmented.get(key) != original.get(key):
            errors.append(f"{key}_changed")
    if not augmented.get("parent_id"):
        errors.append("missing_parent_id")
    if augmented.get("augmentation_style") not in PUBLIC_STYLES:
        errors.append("missing_or_bad_augmentation_style")
    if _prompt_key(str(augmented.get("prompt", ""))) == _prompt_key(str(original.get("prompt", ""))):
        errors.append("prompt_unchanged")
    ok, composed_errors = validate_composed_row(augmented)
    if not ok:
        errors.extend(composed_errors)
    trainability = verify_trainable_row(augmented)
    if not trainability.trainable:
        errors.append(f"trainability_gate_failed:{trainability.rejection_reason}")
    return not errors, errors


def _render_style(row: dict[str, Any], style: str, rng: random.Random) -> str:
    family = str(row["family"]).replace("composed_", "").replace("_", " ")
    subfamilies = ", ".join(str(item) for item in row.get("subfamilies", []))
    original = str(row["prompt"]).strip()
    steps = row.get("substeps", [])
    step_lines = [f"step {step['index'] + 1}: {step['family']} input {step['input']}" for step in steps]
    salt = rng.randint(1000, 9999)
    if style == "direct":
        return f"Direct composed task {salt}. {original}"
    if style == "minimal":
        return f"Solve exactly: {original} [case {salt}]"
    if style == "examples_query":
        return f"Example: verify each listed substep before answering. Query {salt}: {original}"
    if style == "table":
        return f"Task table {salt} | family={family} | subfamilies={subfamilies} | query={original}"
    if style == "story":
        return f"A worksheet note says all constants are explicit. Case {salt}: {original}"
    if style == "public_style_minimal":
        return f"{family} puzzle #{salt}: {original}"
    if style == "public_style_verbose":
        return f"Use only the rules written here, verify every intermediate value, and solve this {family} item: {original} Ref {salt}."
    if style == "public_style_table":
        return " | ".join([f"public table {salt}", f"type={family}", f"parts={subfamilies}", f"question={original}"])
    if style == "public_style_examples_then_query":
        return f"Worked-example style: first decode the stated transform, then apply the stated operation. Query {salt}: {original}"
    if style == "public_style_noisy_context":
        return f"Ignore the label code {salt}; it is not part of the math. Actual task: {original}"
    if style == "public_style_symbol_heavy":
        return f"@@ composed::{family} :: {original} :: verify[{len(steps)}] :: id[{salt}]"
    if style == "public_style_mixed_notation":
        return f"{family} / substeps -> ({'; '.join(step_lines)}). Now answer the original query: {original} #{salt}"
    if style == "public_style_multiline":
        return "\n".join([f"Public prompt {salt}", f"Family: {family}", f"Subfamilies: {subfamilies}", f"Question: {original}"])
    if style == "public_style_plain_english":
        return f"Find the answer by doing the composed operations in order. The problem is: {original} Reference {salt}."
    raise ValueError(f"unsupported_public_style:{style}")


def _stable_id(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:24]


def _offset(text: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(text))


def _prompt_key(prompt: str) -> str:
    return re.sub(r"\s+", " ", prompt.strip())


def required_family_prompt_style_support() -> dict[str, int]:
    return {family: len(teacher.supported_prompt_styles()) for family, teacher in COMPOSED_TEACHER_REGISTRY.items()}
