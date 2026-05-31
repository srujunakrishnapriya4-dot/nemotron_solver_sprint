from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.eval.day5_eval_factory import read_jsonl
from kaggle_anti086.kaggle_path_safety import safe_write_text


UNSUPPORTED_FAMILIES = {"custom_numeral", "equation_operator", "permutation_sorting", "sequence_pattern"}


def build_weak_subfamily_report(pairs: dict[str, tuple[Path, Path]]) -> dict:
    buckets: dict[tuple[str, str], dict] = defaultdict(_bucket)
    for eval_name, (eval_path, prediction_path) in pairs.items():
        rows = {str(row["id"]): row for row in read_jsonl(eval_path)}
        for pred in read_jsonl(prediction_path):
            row = rows.get(str(pred.get("id")))
            if row is None:
                continue
            family = str(row["family"])
            subfamily = str(row["subfamily"])
            bucket = buckets[(family, subfamily)]
            behavior = str(row.get("metadata", {}).get("expected_solver_behavior", "answer"))
            attempted = not bool(pred.get("abstained", True))
            correct = bool(pred.get("correct", False))
            unsafe = behavior == "abstain" and attempted
            bucket["family"] = family
            bucket["subfamily"] = subfamily
            bucket["evals"].add(eval_name)
            bucket["count"] += 1
            if behavior == "answer":
                bucket["answerable_count"] += 1
            if attempted:
                bucket["attempted"] += 1
            else:
                bucket["abstained_count"] += 1
            if correct:
                bucket["correct"] += 1
            if attempted and behavior == "answer" and not correct:
                bucket["wrong_answer_count"] += 1
            if unsafe:
                bucket["unsafe_answer_count"] += 1
            if not bool(pred.get("behavior_correct", correct)):
                bucket["failure_reasons"].append(str(pred.get("reason", "")) or "solver_failure")
    rows = []
    for bucket in buckets.values():
        count = bucket["count"]
        answerable = bucket["answerable_count"]
        attempted = bucket["attempted"]
        wrong = bucket["wrong_answer_count"]
        unsafe = bucket["unsafe_answer_count"]
        exact = 0.0 if answerable == 0 else bucket["correct"] / answerable
        attempt_rate = 0.0 if answerable == 0 else attempted / count
        wrong_rate = 0.0 if answerable == 0 else wrong / answerable
        unsafe_rate = 0.0 if count == 0 else unsafe / count
        dominant = _dominant_failure(bucket, exact, attempt_rate, wrong_rate)
        action = _day6_action(bucket, dominant, exact, attempt_rate, wrong_rate, unsafe_rate)
        rows.append(
            {
                "family": bucket["family"],
                "subfamily": bucket["subfamily"],
                "count": count,
                "answerable_count": answerable,
                "attempted": attempted,
                "correct": bucket["correct"],
                "wrong_answer_count": wrong,
                "abstained_count": bucket["abstained_count"],
                "exact_match": exact,
                "attempt_rate": attempt_rate,
                "wrong_answer_rate": wrong_rate,
                "unsafe_answer_rate": unsafe_rate,
                "dominant_failure": dominant,
                "day6_action": action,
                "evals": sorted(bucket["evals"]),
                "failure_reason_counts": dict(Counter(bucket["failure_reasons"]).most_common(8)),
            }
        )
    action_counts = Counter(row["day6_action"] for row in rows)
    return {
        "schema_version": 1,
        "created_by": "SPRINT-11D.1",
        "subfamilies": sorted(rows, key=lambda item: (item["family"], item["subfamily"])),
        "action_counts": dict(sorted(action_counts.items())),
        "status": "PASS",
    }


def _bucket() -> dict:
    return {
        "family": "",
        "subfamily": "",
        "evals": set(),
        "count": 0,
        "answerable_count": 0,
        "attempted": 0,
        "correct": 0,
        "wrong_answer_count": 0,
        "abstained_count": 0,
        "unsafe_answer_count": 0,
        "failure_reasons": [],
    }


def _dominant_failure(bucket: dict, exact: float, attempt_rate: float, wrong_rate: float) -> str:
    family = bucket["family"]
    if family in UNSUPPORTED_FAMILIES:
        return "unsupported_family"
    if bucket["unsafe_answer_count"]:
        return "router_error"
    if bucket["answerable_count"] and bucket["attempted"] == 0:
        return "all_solvers_abstained"
    if wrong_rate >= 0.10:
        return "high_wrong_answer_rate"
    if bucket["wrong_answer_count"]:
        return "solver_wrong_answer"
    if bucket["answerable_count"] and attempt_rate < 0.50:
        return "low_attempt_rate"
    if exact < 0.50 and bucket["answerable_count"]:
        return "low_attempt_rate"
    return "none"


def _day6_action(bucket: dict, dominant: str, exact: float, attempt_rate: float, wrong_rate: float, unsafe_rate: float) -> str:
    family = bucket["family"]
    if dominant == "unsupported_family":
        return "blocked_unsupported_family"
    if unsafe_rate > 0:
        return "blocked_solver_failure"
    if bucket["answerable_count"] == 0 and bucket["abstained_count"]:
        return "eligible_for_abstain_safety"
    if exact >= 0.80 and attempt_rate >= 0.70 and wrong_rate <= 0.05:
        return "eligible_for_v2_direct_answer"
    if bucket["wrong_answer_count"] and wrong_rate <= 0.20:
        return "eligible_for_v2_hard_negative"
    if family == "format_only":
        return "eligible_for_abstain_safety"
    if dominant in {"all_solvers_abstained", "low_attempt_rate"}:
        return "route_to_solver_repair_before_direct_answer_training"
    return "blocked_solver_failure"


def write_weak_subfamily_report(report: dict, path: str | Path) -> None:
    safe_write_text(path, json.dumps(report, indent=2, sort_keys=True) + "\n", field_name="weak_subfamily_report")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Day 5.1 weak subfamily report from actual predictions.")
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
    report = build_weak_subfamily_report(pairs)
    write_weak_subfamily_report(report, args.out)
    print(json.dumps({"status": report["status"], "subfamily_count": len(report["subfamilies"]), "out": args.out}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
