from __future__ import annotations

from collections import Counter, defaultdict
import re
from typing import Any, Iterable

from kaggle_anti086.training.day1_teacher_trainability_gate import (
    KNOWN_TRAINABLE_FAMILIES,
    detect_abstain_placeholder,
    detect_answer_mismatch,
    detect_multiple_boxes,
    detect_text_after_final_box,
    detect_unsafe_metadata,
    verify_trainable_row,
)


AuditResult = dict[str, Any]


def audit_format(records: Iterable[dict[str, Any]]) -> AuditResult:
    failures: list[str] = []
    counts = Counter()
    for row in records:
        rid = row.get("record_id", "unknown")
        target = str(row.get("target_text", ""))
        has_multiple_boxes = detect_multiple_boxes(target)
        has_text_after_box = detect_text_after_final_box(target)
        if has_multiple_boxes:
            counts["multiple_boxes"] += 1
            failures.append(f"{rid}: multiple_boxes")
        if has_text_after_box:
            counts["text_after_box"] += 1
            failures.append(f"{rid}: text_after_box")
        if not has_multiple_boxes and not has_text_after_box and detect_answer_mismatch(
            target, str(row.get("answer", "")), row.get("family"), row.get("metadata")
        ):
            counts["answer_mismatch"] += 1
            failures.append(f"{rid}: answer_mismatch")
        if detect_abstain_placeholder(row):
            counts["abstain_placeholder"] += 1
            failures.append(f"{rid}: abstain_placeholder")
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 2:
            counts["missing_messages"] += 1
            failures.append(f"{rid}: missing_messages")
        elif messages[0].get("content") != row.get("prompt") or messages[1].get("content") != row.get("target_text"):
            counts["message_mismatch"] += 1
            failures.append(f"{rid}: message_mismatch")
    return _result("PASS" if not failures else "FAIL", counts, failures, [])


def audit_dedup_leakage(
    train: Iterable[dict[str, Any]],
    eval_rows: Iterable[dict[str, Any]],
    probe: Iterable[dict[str, Any]],
    pairs: Iterable[dict[str, Any]],
    *,
    strict_pair_prompts: bool = True,
) -> AuditResult:
    failures: list[str] = []
    counts = Counter()
    splits = {"train": list(train), "eval": list(eval_rows), "probe": list(probe)}
    prompts_by_split: dict[str, set[str]] = {}
    all_records = [row for rows in splits.values() for row in rows]
    for split, rows in splits.items():
        prompts = [_prompt_key(row.get("prompt", "")) for row in rows]
        duplicates = len(prompts) - len(set(prompts))
        if duplicates:
            counts[f"{split}_duplicate_prompts"] = duplicates
            failures.append(f"{split}: duplicate_prompts={duplicates}")
        prompts_by_split[split] = set(prompts)
    for left, right in (("train", "eval"), ("train", "probe"), ("eval", "probe")):
        overlap = prompts_by_split[left] & prompts_by_split[right]
        if overlap:
            counts[f"{left}_{right}_prompt_overlap"] = len(overlap)
            failures.append(f"{left}/{right}: prompt_overlap={len(overlap)}")
    record_ids = [row.get("record_id") for row in all_records]
    if len(record_ids) != len(set(record_ids)):
        counts["duplicate_record_ids"] = len(record_ids) - len(set(record_ids))
        failures.append("duplicate_record_ids")
    pair_list = list(pairs)
    pair_ids = [row.get("pair_id") for row in pair_list]
    if len(pair_ids) != len(set(pair_ids)):
        counts["duplicate_pair_ids"] = len(pair_ids) - len(set(pair_ids))
        failures.append("duplicate_pair_ids")
    if strict_pair_prompts:
        positive_prompts = set().union(*prompts_by_split.values())
        pair_prompts = [_prompt_key(row.get("prompt", "")) for row in pair_list]
        prompt_overlap = positive_prompts & set(pair_prompts)
        pair_duplicates = len(pair_prompts) - len(set(pair_prompts))
        if prompt_overlap:
            counts["positive_pair_prompt_overlap"] = len(prompt_overlap)
            failures.append(f"positive_pair_prompt_overlap={len(prompt_overlap)}")
        if pair_duplicates:
            counts["pair_prompt_duplicates"] = pair_duplicates
            failures.append(f"pair_prompt_duplicates={pair_duplicates}")
    return _result("PASS" if not failures else "FAIL", counts, failures, [])


def audit_family_balance(records: Iterable[dict[str, Any]], expected_per_family: dict[str, int] | None = None) -> AuditResult:
    rows = list(records)
    counts = Counter(row.get("family", "UNKNOWN") for row in rows)
    failures: list[str] = []
    warnings: list[str] = []
    for family in sorted(KNOWN_TRAINABLE_FAMILIES):
        if counts.get(family, 0) == 0:
            failures.append(f"{family}: missing")
    if expected_per_family:
        for family, expected in sorted(expected_per_family.items()):
            actual = counts.get(family, 0)
            if actual != expected:
                failures.append(f"{family}: expected={expected} actual={actual}")
    return _result("PASS" if not failures else "FAIL", counts, failures, warnings)


def audit_prompt_diversity(records: Iterable[dict[str, Any]]) -> AuditResult:
    rows = list(records)
    counts = Counter(row.get("prompt_style", "UNKNOWN") for row in rows)
    failures: list[str] = []
    warnings: list[str] = []
    total = len(rows)
    if total:
        style, count = counts.most_common(1)[0]
        if count / total > 0.45:
            failures.append(f"prompt_style_dominates:{style}:{count}/{total}")
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row.get("family"))].append(row)
    for family, family_rows in sorted(by_family.items()):
        styles = {str(row.get("prompt_style")) for row in family_rows}
        difficulties = {str(row.get("difficulty")) for row in family_rows}
        medium_hard = sum(1 for row in family_rows if row.get("difficulty") in {"medium", "hard"})
        if len(styles) < 3:
            warnings.append(f"{family}: prompt_styles={len(styles)}")
        if medium_hard < max(1, int(0.2 * len(family_rows))):
            failures.append(f"{family}: medium_hard_below_20_percent")
        if difficulties == {"easy"}:
            failures.append(f"{family}: all_easy")
    status = "FAIL" if failures else ("WARN" if warnings else "PASS")
    return _result(status, counts, failures, warnings)


def audit_teacher_verification(records: Iterable[dict[str, Any]]) -> AuditResult:
    failures: list[str] = []
    counts = Counter()
    seen: set[str] | None = None
    for row in records:
        rid = row.get("record_id", "unknown")
        check = verify_trainable_row(row, seen)
        if not check.trainable:
            counts[check.rejection_reason or "unknown"] += 1
            failures.append(f"{rid}: {check.rejection_reason}")
        if row.get("verification_status") != "PASS":
            counts["verification_not_pass"] += 1
        if row.get("ambiguity_count") != 0:
            counts["ambiguity_nonzero"] += 1
        if row.get("trainable") is not True:
            counts["trainable_not_true"] += 1
        if detect_unsafe_metadata(dict(row.get("metadata") or {})):
            counts["unsafe_metadata"] += 1
    return _result("PASS" if not failures else "FAIL", counts, failures, [])


def _result(status: str, counts: Counter[str] | dict[str, int], failures: list[str], warnings: list[str]) -> AuditResult:
    return {
        "status": status,
        "counts": dict(sorted(dict(counts).items())),
        "failures": failures[:100],
        "warnings": warnings[:100],
    }


def _prompt_key(prompt: str) -> str:
    return re.sub(r"\s+", " ", str(prompt).strip())
