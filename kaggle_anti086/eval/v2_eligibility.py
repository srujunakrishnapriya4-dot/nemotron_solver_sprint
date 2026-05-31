from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.schema import RowValidationError, validate_row
from kaggle_anti086.eval.day5_eval_factory import read_jsonl
from kaggle_anti086.kaggle_path_safety import safe_write_text


CLASSES = (
    "eligible_verified_answer",
    "eligible_hard_negative",
    "eligible_abstain_safety",
    "blocked_unsupported_family",
    "blocked_solver_failure",
    "blocked_leakage_risk",
    "blocked_eval_generator_bug",
    "blocked_missing_metadata",
)

UNSUPPORTED_DIRECT_FAMILIES = {"custom_numeral", "equation_operator", "permutation_sorting", "sequence_pattern"}
THRESHOLDS = {
    "eligible_verified_answer_count": 1000,
    "eligible_abstain_safety_count": 200,
    "unsafe_answer_rate": 0.0,
}
PER_FAMILY_THRESHOLDS = {
    "bit_manipulation": 100,
    "symbol_mapping": 100,
    "char_cipher": 100,
    "unit_conversion": 100,
    "numeric_formula": 80,
    "gravity_numeric": 80,
    "word_cipher": 100,
    "roman_numeral": 50,
}


def build_v2_eligibility_report(pairs: dict[str, tuple[Path, Path]]) -> dict:
    items = []
    seen_rule_signatures: Counter[str] = Counter()
    seen_leakage: Counter[str] = Counter()
    seen_prompts: Counter[str] = Counter()
    for eval_name, (eval_path, prediction_path) in pairs.items():
        predictions = {str(row.get("id")): row for row in read_jsonl(prediction_path)}
        for row in read_jsonl(eval_path):
            metadata = row.get("metadata", {})
            seen_rule_signatures[str(metadata.get("rule_signature", ""))] += 1
            seen_leakage[str(row.get("leakage_group", ""))] += 1
            seen_prompts[_normalized_prompt(str(row.get("prompt", "")))] += 1
            items.append((eval_name, row, predictions.get(str(row.get("id")), {})))

    classifications = []
    class_counts: Counter[str] = Counter()
    by_family: dict[str, Counter[str]] = defaultdict(Counter)
    by_subfamily: dict[str, Counter[str]] = defaultdict(Counter)
    direct_dist: Counter[str] = Counter()
    hard_negative_dist: Counter[str] = Counter()
    blocked_dist: Counter[str] = Counter()
    leakage_risk_count = 0
    unsafe_answer_count = 0

    for eval_name, row, pred in items:
        family = str(row.get("family", "unknown"))
        subfamily = str(row.get("subfamily", "unknown"))
        metadata = row.get("metadata", {})
        missing = _missing_metadata(row)
        leakage_risk = (
            seen_rule_signatures[str(metadata.get("rule_signature", ""))] > 1
            or seen_leakage[str(row.get("leakage_group", ""))] > 1
            or seen_prompts[_normalized_prompt(str(row.get("prompt", "")))] > 1
        )
        if leakage_risk:
            leakage_risk_count += 1
        if row.get("metadata", {}).get("expected_solver_behavior") == "abstain" and not bool(pred.get("abstained", True)):
            unsafe_answer_count += 1
        klass = _classify(row, pred, missing=missing, leakage_risk=leakage_risk)
        class_counts[klass] += 1
        by_family[family][klass] += 1
        by_subfamily[f"{family}::{subfamily}"][klass] += 1
        if klass == "eligible_verified_answer":
            direct_dist[family] += 1
        elif klass == "eligible_hard_negative":
            hard_negative_dist[family] += 1
        elif klass.startswith("blocked_"):
            blocked_dist[family] += 1
        classifications.append(
            {
                "eval": eval_name,
                "id": row.get("id"),
                "family": family,
                "subfamily": subfamily,
                "classification": klass,
                "prediction": pred.get("prediction", ""),
                "correct": bool(pred.get("correct", False)),
                "abstained": bool(pred.get("abstained", True)),
                "leakage_risk": leakage_risk,
                "missing_metadata": missing,
            }
        )

    blocked_counts = {name: class_counts[name] for name in CLASSES if name.startswith("blocked_")}
    unsafe_answer_rate = 0.0 if not items else unsafe_answer_count / len(items)
    per_family_gate_status = {
        family: {
            "value": direct_dist.get(family, 0),
            "threshold": threshold,
            "status": "PASS" if direct_dist.get(family, 0) >= threshold else "FAIL",
        }
        for family, threshold in PER_FAMILY_THRESHOLDS.items()
    }
    remaining_blockers = _remaining_blockers(class_counts, blocked_counts, direct_dist, unsafe_answer_rate)
    status = "PASS" if not remaining_blockers else "FAIL"
    day6_decision = "ALLOW_CORPUS_BUILD" if status == "PASS" else "BLOCK_CORPUS_BUILD"
    return {
        "schema_version": 2,
        "created_by": "SPRINT-11D.2",
        "status": status,
        "training_allowed": False,
        "thresholds": THRESHOLDS,
        "total_rows_seen": len(items),
        "eligible_verified_answer_count": class_counts["eligible_verified_answer"],
        "eligible_hard_negative_count": class_counts["eligible_hard_negative"],
        "eligible_abstain_safety_count": class_counts["eligible_abstain_safety"],
        "blocked_counts": blocked_counts,
        "classification_counts": {name: class_counts[name] for name in CLASSES},
        "by_family": {family: dict(counter) for family, counter in sorted(by_family.items())},
        "by_subfamily": {subfamily: dict(counter) for subfamily, counter in sorted(by_subfamily.items())},
        "direct_answer_family_distribution": dict(sorted(direct_dist.items())),
        "eligible_by_family": dict(sorted(direct_dist.items())),
        "hard_negative_family_distribution": dict(sorted(hard_negative_dist.items())),
        "blocked_family_distribution": dict(sorted(blocked_dist.items())),
        "leakage_risk_count": leakage_risk_count,
        "unsafe_answer_count": unsafe_answer_count,
        "unsafe_answer_rate": unsafe_answer_rate,
        "per_family_thresholds": PER_FAMILY_THRESHOLDS,
        "per_family_gate_status": per_family_gate_status,
        "remaining_blockers": remaining_blockers,
        "day6_decision": day6_decision,
        "row_classifications": classifications[:2000],
    }


def _remaining_blockers(class_counts: Counter[str], blocked_counts: dict[str, int], direct_dist: Counter[str], unsafe_answer_rate: float) -> list[str]:
    blockers = []
    if class_counts["eligible_verified_answer"] < THRESHOLDS["eligible_verified_answer_count"]:
        blockers.append("eligible_verified_answer_count_below_threshold")
    if class_counts["eligible_abstain_safety"] < THRESHOLDS["eligible_abstain_safety_count"]:
        blockers.append("eligible_abstain_safety_count_below_threshold")
    for key in ("blocked_leakage_risk", "blocked_missing_metadata", "blocked_eval_generator_bug"):
        if blocked_counts.get(key, 0) > 0:
            blockers.append(f"{key}_present")
    if unsafe_answer_rate > THRESHOLDS["unsafe_answer_rate"]:
        blockers.append("unsafe_answer_rate_above_zero")
    for family, threshold in PER_FAMILY_THRESHOLDS.items():
        if direct_dist.get(family, 0) < threshold:
            blockers.append(f"{family}_eligible_verified_answer_below_{threshold}")
    return blockers


def _classify(row: dict, pred: dict, *, missing: list[str], leakage_risk: bool) -> str:
    if missing:
        return "blocked_missing_metadata"
    try:
        validate_row(row, context=str(row.get("id", "")))
    except RowValidationError:
        return "blocked_eval_generator_bug"
    if leakage_risk:
        return "blocked_leakage_risk"
    family = str(row.get("family", ""))
    behavior = str(row.get("metadata", {}).get("expected_solver_behavior", "answer"))
    verified = row.get("verification_status") == "verified"
    correct = bool(pred.get("correct", False))
    abstained = bool(pred.get("abstained", True))
    if behavior == "abstain":
        return "eligible_abstain_safety" if abstained else "blocked_solver_failure"
    if family in UNSUPPORTED_DIRECT_FAMILIES and not correct:
        return "blocked_unsupported_family"
    if not verified:
        return "blocked_eval_generator_bug"
    if correct:
        return "eligible_verified_answer"
    if not abstained and family not in {"equation_operator", "sequence_pattern", "permutation_sorting"}:
        return "eligible_hard_negative"
    return "blocked_solver_failure"


def _missing_metadata(row: dict) -> list[str]:
    required = ("id", "family", "subfamily", "rule_id", "leakage_group", "verification_status", "metadata")
    missing = [field for field in required if field not in row or row.get(field) in (None, "")]
    metadata = row.get("metadata", {})
    for field in ("expected_solver_behavior", "rule_signature", "generator_id"):
        if not isinstance(metadata, dict) or not metadata.get(field):
            missing.append(f"metadata.{field}")
    return missing


def _normalized_prompt(prompt: str) -> str:
    return re.sub(r"\s+", " ", prompt.strip().lower())


def write_v2_eligibility_report(report: dict, path: str | Path) -> None:
    safe_write_text(path, json.dumps(report, indent=2, sort_keys=True) + "\n", field_name="v2_eligibility_report")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Day 5.1 v2 corpus eligibility report.")
    for name in ("private-like", "rule-holdout", "family-hard", "anti-leak"):
        parser.add_argument(f"--{name}")
        parser.add_argument(f"--{name}-predictions")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    pairs = {}
    for attr, name in (
        ("private_like", "private_like"),
        ("rule_holdout", "rule_holdout"),
        ("family_hard", "family_hard"),
        ("anti_leak", "anti_leak"),
    ):
        eval_path = getattr(args, attr)
        pred_path = getattr(args, f"{attr}_predictions")
        if eval_path and pred_path:
            pairs[name] = (Path(eval_path), Path(pred_path))
    if not pairs:
        raise SystemExit("at least one eval/prediction pair is required")
    report = build_v2_eligibility_report(pairs)
    write_v2_eligibility_report(report, args.out)
    print(
        json.dumps(
            {
                "status": report["status"],
                "day6_decision": report["day6_decision"],
                "eligible_verified_answer_count": report["eligible_verified_answer_count"],
                "out": args.out,
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
