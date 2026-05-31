from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.v2_corpus_io import read_json, read_jsonl, write_json_checked


ALLOWED_SFT_FAMILIES = {
    "bit_manipulation",
    "char_cipher",
    "format_only",
    "gravity_numeric",
    "numeric_formula",
    "roman_numeral",
    "symbol_mapping",
    "unit_conversion",
    "word_cipher",
}
UNSUPPORTED_SFT_FAMILIES = {"equation_operator", "sequence_pattern", "permutation_sorting", "custom_numeral", "unknown"}
TARGET_MIXTURE = {
    "bit_manipulation": 0.15,
    "numeric_formula": 0.12,
    "gravity_numeric": 0.08,
    "unit_conversion": 0.12,
    "word_cipher": 0.13,
    "char_cipher": 0.10,
    "symbol_mapping": 0.12,
    "roman_numeral": 0.06,
    "format_only": 0.04,
}
HARD_MINIMUMS = {
    "bit_manipulation": 100,
    "numeric_formula": 80,
    "gravity_numeric": 60,
    "unit_conversion": 80,
    "word_cipher": 80,
    "char_cipher": 80,
    "symbol_mapping": 80,
    "roman_numeral": 40,
    "format_only": 40,
}


def build_sampling_policy(
    direct_rows: list[dict[str, Any]],
    solver_corrected_rows: list[dict[str, Any]],
    abstain_rows: list[dict[str, Any]],
    hard_negative_rows: list[dict[str, Any]],
    independence_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    independence_report = independence_report or {}
    overlay = independence_report.get("policy_overlay", {})
    row_policy = {}
    violations = Counter()
    for row in direct_rows:
        policy = _direct_policy()
        _validate_sft_policy(row, policy, violations)
        row_policy[str(row.get("id"))] = policy
    for row in solver_corrected_rows:
        correction_policy = overlay.get(str(row.get("id")), {})
        policy = _solver_corrected_policy(correction_policy)
        _validate_sft_policy(row, policy, violations)
        row_policy[str(row.get("id"))] = policy
    for row in abstain_rows:
        policy = _abstain_policy()
        if policy["allowed_for_sft"]:
            violations["abstain_safety_sft_violations"] += 1
        row_policy[str(row.get("id"))] = policy
    for row in hard_negative_rows:
        policy = _hard_negative_policy()
        if policy["allowed_for_sft"]:
            violations["hard_negative_sft_violations"] += 1
        row_policy[str(row.get("id"))] = policy

    family_counts = Counter(str(row.get("family", "unknown")) for row in direct_rows)
    total = len(direct_rows)
    actual = {family: (family_counts.get(family, 0) / total if total else 0.0) for family in sorted(TARGET_MIXTURE)}
    underrepresented = []
    overrepresented = []
    warnings = []
    for family, target in TARGET_MIXTURE.items():
        count = family_counts.get(family, 0)
        share = actual.get(family, 0.0)
        if share < 0.50 * target and count < HARD_MINIMUMS[family]:
            underrepresented.append(family)
        if share > 1.75 * target:
            overrepresented.append(family)
    if underrepresented:
        warnings.append("underrepresented_families_present")
    if overrepresented:
        warnings.append("overrepresented_families_present")
    failures = []
    for key in (
        "loss_policy_violations",
        "unsupported_sft_family_violations",
        "hard_negative_sft_violations",
        "abstain_safety_sft_violations",
        "train_on_user_violations",
        "unverified_sft_violations",
    ):
        if violations[key] > 0:
            failures.append({"code": key, "count": violations[key]})
    return {
        "status": "FAIL" if failures else ("WARN" if warnings else "PASS"),
        "target_mixture": TARGET_MIXTURE,
        "actual_mixture": actual,
        "family_counts": dict(sorted(family_counts.items())),
        "underrepresented_families": underrepresented,
        "overrepresented_families": overrepresented,
        "sft_allowed_count": len(direct_rows) + len(solver_corrected_rows),
        "sft_blocked_count": len(abstain_rows) + len(hard_negative_rows),
        "loss_policy_violations": violations["loss_policy_violations"],
        "unsupported_sft_family_violations": violations["unsupported_sft_family_violations"],
        "hard_negative_sft_violations": violations["hard_negative_sft_violations"],
        "abstain_safety_sft_violations": violations["abstain_safety_sft_violations"],
        "train_on_user_violations": violations["train_on_user_violations"],
        "unverified_sft_violations": violations["unverified_sft_violations"],
        "row_policy": row_policy,
        "warnings": warnings,
        "failures": failures,
    }


def _direct_policy() -> dict[str, Any]:
    return {"allowed_for_sft": True, "allowed_for_dpo": False, "allowed_for_eval": False, "train_on_user": False, "train_on_assistant": True, "loss_scope": "assistant_only", "sampling_weight": 1.0, "counted_as_independent": True}


def _solver_corrected_policy(overlay: dict[str, Any]) -> dict[str, Any]:
    derived = bool(overlay.get("derived_from_direct", True))
    return {"allowed_for_sft": True, "allowed_for_dpo": False, "allowed_for_eval": False, "train_on_user": False, "train_on_assistant": True, "loss_scope": "assistant_only", "sampling_weight": 0.5 if derived else 1.0, "counted_as_independent": not derived, "metadata": {"derived_from_direct": derived}}


def _abstain_policy() -> dict[str, Any]:
    return {"allowed_for_sft": False, "allowed_for_dpo": False, "allowed_for_eval": True, "allowed_for_safety_eval": True, "train_on_user": False, "train_on_assistant": False, "loss_scope": "none", "sampling_weight": 0.0, "counted_as_independent": False}


def _hard_negative_policy() -> dict[str, Any]:
    return {"allowed_for_sft": False, "allowed_for_dpo": True, "allowed_for_contrastive": True, "allowed_for_eval": False, "do_not_use_as_sft": True, "train_on_user": False, "train_on_assistant": False, "loss_scope": "none", "sampling_weight": 0.0, "counted_as_independent": False}


def _validate_sft_policy(row: dict[str, Any], policy: dict[str, Any], violations: Counter[str]) -> None:
    if not policy.get("allowed_for_sft"):
        return
    if policy.get("train_on_user"):
        violations["train_on_user_violations"] += 1
    if policy.get("loss_scope") != "assistant_only":
        violations["loss_policy_violations"] += 1
    if row.get("family") in UNSUPPORTED_SFT_FAMILIES or row.get("family") not in ALLOWED_SFT_FAMILIES:
        violations["unsupported_sft_family_violations"] += 1
    if row.get("verification_status") != "verified":
        violations["unverified_sft_violations"] += 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct", default="artifacts/sprint11/train_v2_verified_direct_answer.jsonl")
    parser.add_argument("--solver-corrected", default="artifacts/sprint11/train_v2_solver_corrected.jsonl")
    parser.add_argument("--abstain-safety", default="artifacts/sprint11/train_v2_abstain_safety.jsonl")
    parser.add_argument("--hard-negative", default="artifacts/sprint11/train_v2_hard_negative.jsonl")
    parser.add_argument("--independence-report", default="artifacts/sprint11/train_v2_independence_report.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = build_sampling_policy(
        read_jsonl(args.direct),
        read_jsonl(args.solver_corrected),
        read_jsonl(args.abstain_safety),
        read_jsonl(args.hard_negative),
        read_json(args.independence_report) if Path(args.independence_report).exists() else {},
    )
    write_json_checked(args.out, report, field_name="train_v2_sampling_policy")
    print(json.dumps({"status": report["status"], "out": args.out}, sort_keys=True))
    return 0 if report["status"] != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
