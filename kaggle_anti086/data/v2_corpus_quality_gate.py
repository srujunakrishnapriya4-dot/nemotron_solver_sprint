from __future__ import annotations

from collections import Counter
from typing import Any

from kaggle_anti086.data.v2_corpus_leakage import UNSUPPORTED_DIRECT_FAMILIES
from kaggle_anti086.data.v2_prompt_templates import validate_training_messages


PER_FAMILY_MINIMUMS = {
    "bit_manipulation": 100,
    "symbol_mapping": 100,
    "char_cipher": 100,
    "unit_conversion": 100,
    "numeric_formula": 80,
    "gravity_numeric": 80,
    "word_cipher": 100,
    "roman_numeral": 50,
    "format_only": 20,
}


def build_quality_gate(
    direct_rows: list[dict[str, Any]],
    abstain_rows: list[dict[str, Any]],
    hard_negative_rows: list[dict[str, Any]],
    leakage_report: dict[str, Any],
    mixture_report: dict[str, Any],
) -> dict[str, Any]:
    family_counts = Counter(str(row.get("family", "unknown")) for row in direct_rows)
    source_counts = Counter(str(row.get("source_row_id", "")) for row in direct_rows)
    assistant_only_rows = sum(1 for row in direct_rows if row.get("loss_scope") == "assistant_only")
    full_prompt_loss_rows = sum(1 for row in direct_rows if row.get("loss_scope") == "full_prompt")
    unsupported_direct_rows = sum(1 for row in direct_rows if row.get("family") in UNSUPPORTED_DIRECT_FAMILIES)
    verbose_assistant_rows = 0
    empty_assistant_rows = 0
    for row in direct_rows:
        report = validate_training_messages(row.get("messages", []), answer=str(row.get("answer", "")))
        if "assistant_verbose_or_markdown" in report["failures"]:
            verbose_assistant_rows += 1
        if "assistant_empty" in report["failures"]:
            empty_assistant_rows += 1

    gates = {
        "verified_direct_answer_rows": _min_gate(len(direct_rows), 1000),
        "eligible_abstain_safety_rows": _min_gate(len(abstain_rows), 200),
        "hard_negative_rows": _min_gate(len(hard_negative_rows), 20),
        "unsupported_family_direct_rows": _eq_gate(unsupported_direct_rows, 0),
        "rule_id_overlap_count": _eq_gate(int(leakage_report.get("rule_id_overlap_count", 0)), 0),
        "leakage_group_overlap_count": _eq_gate(int(leakage_report.get("leakage_group_overlap_count", 0)), 0),
        "prompt_hash_overlap_count": _eq_gate(int(leakage_report.get("prompt_hash_overlap_count", 0)), 0),
        "assistant_only_loss_rows": _eq_gate(assistant_only_rows, len(direct_rows)),
        "full_prompt_loss_rows": _eq_gate(full_prompt_loss_rows, 0),
        "verbose_assistant_rows": _eq_gate(verbose_assistant_rows, 0),
        "empty_assistant_rows": _eq_gate(empty_assistant_rows, 0),
        "duplicate_source_row_ids": _eq_gate(sum(1 for count in source_counts.values() if count > 1), 0),
        "leakage_report_status": {"value": leakage_report.get("status"), "threshold": "PASS", "status": "PASS" if leakage_report.get("status") == "PASS" else "FAIL"},
        "mixture_report_status": {"value": mixture_report.get("status"), "threshold": "PASS", "status": "PASS" if mixture_report.get("status") == "PASS" else "FAIL"},
    }
    for family, minimum in PER_FAMILY_MINIMUMS.items():
        gates[f"{family}_rows"] = _min_gate(family_counts.get(family, 0), minimum)
    remaining = [name for name, gate in gates.items() if gate["status"] != "PASS"]
    status = "PASS" if not remaining else "FAIL"
    return {
        "status": status,
        "decision": "ALLOW_DAY7_TRAINING_CONFIG_PREP" if status == "PASS" else "BLOCK_DAY7_TRAINING_CONFIG_PREP",
        "training_allowed": False,
        "quality_gates": gates,
        "family_counts": dict(sorted(family_counts.items())),
        "remaining_blockers": remaining,
    }


def _min_gate(value: int, threshold: int) -> dict[str, Any]:
    return {"value": value, "threshold": threshold, "status": "PASS" if value >= threshold else "FAIL"}


def _eq_gate(value: int, threshold: int) -> dict[str, Any]:
    return {"value": value, "threshold": threshold, "status": "PASS" if value == threshold else "FAIL"}
