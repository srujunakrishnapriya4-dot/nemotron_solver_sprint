from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from nemotron_engine.core.schemas import stable_hash


FAMILIES = ("equation_symbolic", "bit_manipulation", "gravity_numeric", "unit_conversion", "cipher_text")


def analyze_hard_family_roi(
    input_dir: str | Path = "artifacts/win_system",
    output_path: str | Path = "artifacts/win_system/hard_family_roi_report.json",
) -> dict[str, Any]:
    root = Path(input_dir)
    coverage = _read_json(root / "solver_coverage_report.json")
    equation_taxonomy = _read_json(root / "equation_failure_taxonomy.json")
    digit_taxonomy = _read_json(root / "digit_symbol_failure_taxonomy.json")
    adversarial = _read_json(root / "adversarial_v2_manifest.json")
    corpus = _read_json(root / "win_corpus_manifest.json")
    bit_budget = _read_jsonl(root / "failure_samples_bit_budget.jsonl")
    numeric_precision = _read_jsonl(root / "failure_samples_numeric_precision.jsonl")

    correct = coverage.get("solver_verified_correct_by_family", {})
    wrong = coverage.get("solver_wrong_by_family", {})
    abstain_reasons = coverage.get("top_abstention_reasons", {})
    rows: dict[str, dict[str, Any]] = {}
    rows["equation_symbolic"] = _equation_roi(correct, wrong, abstain_reasons, equation_taxonomy, digit_taxonomy, adversarial)
    rows["bit_manipulation"] = _bit_roi(correct, wrong, abstain_reasons, bit_budget, adversarial)
    rows["gravity_numeric"] = _numeric_roi("gravity_numeric", correct, wrong, numeric_precision)
    rows["unit_conversion"] = _numeric_roi("unit_conversion", correct, wrong, numeric_precision)
    rows["cipher_text"] = {
        "current_verified_correct": int(correct.get("cipher_text", 0)),
        "remaining_failures": 0,
        "failure_type_concentration": {"low_current_blocker": True},
        "estimated_implementation_difficulty": "medium",
        "estimated_score_leverage": "low",
        "risk_of_unsafe_verification": "medium",
        "expected_return_next_1_day": "low",
        "recommended_action": "QUARANTINE_FOR_NOW",
    }
    report = {
        "families": rows,
        "recommended_next_sprint": _recommend(rows),
        "inputs": {
            "has_corpus_manifest": bool(corpus),
            "has_adversarial_manifest": bool(adversarial),
        },
    }
    report["report_hash"] = stable_hash(report)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, sort_keys=True, indent=2), encoding="utf-8")
    return report


def _equation_roi(correct, wrong, abstain_reasons, equation_taxonomy, digit_taxonomy, adversarial) -> dict[str, Any]:
    remaining = int(abstain_reasons.get("unsupported_equation_transform", 0))
    enough = int(digit_taxonomy.get("constraint_summary", {}).get("enough_constraints", 0))
    under = int(digit_taxonomy.get("constraint_summary", {}).get("underdetermined_or_unseen_target_operator", 0))
    return {
        "current_verified_correct": int(correct.get("equation_symbolic", 0)),
        "remaining_failures": remaining,
        "failure_type_concentration": equation_taxonomy.get("cluster_counts", {}),
        "digit_binary_constraints": {"enough_constraints": enough, "underdetermined_or_unseen_target_operator": under},
        "estimated_implementation_difficulty": "very_high",
        "estimated_score_leverage": "high_if_solved",
        "risk_of_unsafe_verification": "high",
        "expected_return_next_1_day": "low",
        "adversarial_filtered_count": int(adversarial.get("filtered_by_family", {}).get("equation_symbolic", 0)) if adversarial else 0,
        "recommended_action": "OFFLINE_SYNTHESIS" if enough else "QUARANTINE_FOR_NOW",
    }


def _bit_roi(correct, wrong, abstain_reasons, bit_budget, adversarial) -> dict[str, Any]:
    remaining = int(abstain_reasons.get("bit_candidate_budget_exhausted", 0))
    return {
        "current_verified_correct": int(correct.get("bit_manipulation", 0)),
        "remaining_failures": remaining,
        "failure_type_concentration": dict(Counter(row.get("failure_reason", "unknown") for row in bit_budget).most_common()),
        "estimated_implementation_difficulty": "medium_high",
        "estimated_score_leverage": "high",
        "risk_of_unsafe_verification": "medium",
        "expected_return_next_1_day": "medium_high",
        "adversarial_filtered_count": int(adversarial.get("filtered_by_family", {}).get("bit_manipulation", 0)) if adversarial else 0,
        "recommended_action": "PIVOT_TO_THIS_FAMILY" if remaining else "CONTINUE_SOLVER",
    }


def _numeric_roi(family: str, correct, wrong, numeric_precision) -> dict[str, Any]:
    failures = [row for row in numeric_precision if row.get("family") == family]
    return {
        "current_verified_correct": int(correct.get(family, 0)),
        "remaining_failures": len(failures),
        "verified_wrong": int(wrong.get(family, 0)),
        "failure_type_concentration": dict(Counter(row.get("failure_reason", "unknown") for row in failures).most_common()),
        "estimated_implementation_difficulty": "medium",
        "estimated_score_leverage": "medium_high",
        "risk_of_unsafe_verification": "medium",
        "expected_return_next_1_day": "medium",
        "recommended_action": "PIVOT_TO_THIS_FAMILY" if failures else "CONTINUE_SOLVER",
    }


def _recommend(rows: dict[str, dict[str, Any]]) -> str:
    if rows.get("bit_manipulation", {}).get("recommended_action") == "PIVOT_TO_THIS_FAMILY":
        return "SPRINT-7.7_BIT_BREAKTHROUGH"
    if rows.get("unit_conversion", {}).get("remaining_failures", 0) or rows.get("gravity_numeric", {}).get("remaining_failures", 0):
        return "SPRINT-7.7_NUMERIC_PRECISION"
    return "SPRINT-7.7_EQUATION_OFFLINE_SYNTHESIS"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
