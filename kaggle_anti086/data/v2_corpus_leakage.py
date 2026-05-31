from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Any


UNSUPPORTED_DIRECT_FAMILIES = {"custom_numeral", "equation_operator", "sequence_pattern", "permutation_sorting", "unknown"}


def prompt_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", str(text).strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def build_leakage_report(direct_rows: list[dict[str, Any]], eval_rows_by_name: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    train_rule_ids = {str(row.get("rule_id", "")) for row in direct_rows}
    train_leakage_groups = {str(row.get("leakage_group", "")) for row in direct_rows}
    train_ids = {str(row.get("id", "")) for row in direct_rows}
    train_prompt_hashes = {_training_prompt_hash(row) for row in direct_rows}
    source_ids = [str(row.get("source_row_id", "")) for row in direct_rows]
    prompt_answer_pairs = [(prompt_hash(str(row.get("prompt", ""))), str(row.get("normalized_answer", row.get("answer", "")))) for row in direct_rows]

    rule_holdout_rule_ids = {str(row.get("rule_id", "")) for row in eval_rows_by_name.get("rule_holdout", [])}
    anti_leak_groups = {str(row.get("leakage_group", "")) for row in eval_rows_by_name.get("anti_leak", [])}
    eval_ids = {str(row.get("id", "")) for rows in eval_rows_by_name.values() for row in rows}
    eval_prompt_hashes = {prompt_hash(str(row.get("prompt", ""))) for rows in eval_rows_by_name.values() for row in rows}

    failures: list[dict[str, Any]] = []
    rule_overlap = sorted(train_rule_ids & rule_holdout_rule_ids)
    leakage_overlap = sorted(train_leakage_groups & anti_leak_groups)
    prompt_overlap = sorted(train_prompt_hashes & eval_prompt_hashes)
    id_overlap = sorted(train_ids & eval_ids)
    duplicate_source = [item for item, count in Counter(source_ids).items() if item and count > 1]
    duplicate_prompt_answer = [item for item, count in Counter(prompt_answer_pairs).items() if count > 1]
    unsupported = [row.get("id") for row in direct_rows if row.get("family") in UNSUPPORTED_DIRECT_FAMILIES]

    _append_failures(failures, "rule_id_overlap", rule_overlap)
    _append_failures(failures, "leakage_group_overlap", leakage_overlap)
    _append_failures(failures, "prompt_hash_overlap", prompt_overlap)
    _append_failures(failures, "id_overlap", id_overlap)
    _append_failures(failures, "duplicate_source_row_id", duplicate_source)
    _append_failures(failures, "duplicate_prompt_answer", duplicate_prompt_answer)
    _append_failures(failures, "unsupported_family_direct_row", unsupported)

    return {
        "status": "PASS" if not failures else "FAIL",
        "rule_id_overlap_count": len(rule_overlap),
        "leakage_group_overlap_count": len(leakage_overlap),
        "prompt_hash_overlap_count": len(prompt_overlap),
        "id_overlap_count": len(id_overlap),
        "duplicate_source_row_id_count": len(duplicate_source),
        "duplicate_prompt_answer_count": len(duplicate_prompt_answer),
        "unsupported_family_direct_rows": len(unsupported),
        "failures": failures,
    }


def _training_prompt_hash(row: dict[str, Any]) -> str:
    messages = row.get("messages", [])
    if messages and isinstance(messages, list):
        return prompt_hash(str(messages[0].get("content", "")))
    return prompt_hash(str(row.get("prompt", "")))


def _append_failures(failures: list[dict[str, Any]], code: str, examples: list[Any]) -> None:
    if examples:
        failures.append({"code": code, "count": len(examples), "examples": examples[:20]})
