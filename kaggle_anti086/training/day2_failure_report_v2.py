from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.v2_corpus_io import file_record, read_jsonl, write_json_checked


CLASS_NAMES = (
    "answerable_correct",
    "answerable_wrong_answer",
    "answerable_empty_output",
    "answerable_parse_error",
    "answerable_ambiguity_bug",
    "answerable_invalid_output",
    "unsupported_answerable_or_solver_abstained",
    "abstain_correct",
    "abstain_unsafe_answer",
    "evaluator_policy_bug",
    "unknown",
)
ANSWERABLE_WRONG_CLASSES = {
    "answerable_wrong_answer",
    "answerable_empty_output",
    "answerable_parse_error",
    "answerable_ambiguity_bug",
    "answerable_invalid_output",
    "unsupported_answerable_or_solver_abstained",
    "evaluator_policy_bug",
    "unknown",
}
DEFAULT_INPUTS = {
    "core_eval": "artifacts/sprint11/day2_predictions/core_eval_solver_only_predictions.jsonl",
    "family_eval": "artifacts/sprint11/day2_predictions/family_eval_solver_only_predictions.jsonl",
    "rule_holdout": "artifacts/sprint11/day2_predictions/rule_holdout_solver_only_predictions.jsonl",
    "anti_leak": "artifacts/sprint11/day2_predictions/anti_leak_solver_only_predictions.jsonl",
}
PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}


def classify_row(row: dict[str, Any]) -> str:
    expected = _expected_answer(row)
    reason_text = _reason_text(row)
    output = _output_text(row)
    output_empty = output == ""
    expected_abstain = _is_abstain(expected)
    correct = row.get("correct")
    valid = row.get("valid")

    if expected is None:
        return "unknown"

    if correct is True and valid is False:
        return "evaluator_policy_bug"

    if expected_abstain:
        if output_empty and _mentions_abstain(reason_text):
            return "abstain_correct"
        if not output_empty:
            return "abstain_unsafe_answer"
        if correct is True:
            return "abstain_correct"
        return "unknown"

    if correct is True:
        if output_empty:
            return "evaluator_policy_bug"
        return "answerable_correct"
    if output_empty and _mentions_unsupported_or_abstain(reason_text):
        return "unsupported_answerable_or_solver_abstained"
    if "parse" in reason_text or "parser" in reason_text:
        return "answerable_parse_error"
    if "ambiguity" in reason_text or "multiple" in reason_text or "multifit" in reason_text:
        return "answerable_ambiguity_bug"
    if output_empty:
        return "answerable_empty_output"
    if valid is False:
        return "answerable_invalid_output"
    if correct is False:
        return "answerable_wrong_answer"
    return "unknown"


def build_failure_report_v2(input_files: dict[str, str | Path]) -> dict[str, Any]:
    loaded: list[dict[str, Any]] = []
    input_records: dict[str, Any] = {}
    for dataset, path_value in input_files.items():
        path = Path(path_value)
        if not path.exists():
            raise FileNotFoundError(f"missing required solver-only prediction file for {dataset}: {path}")
        rows = read_jsonl(path)
        input_records[dataset] = file_record(path, row_count=len(rows))
        for row in rows:
            item = dict(row)
            item.setdefault("dataset", dataset)
            item["_classification"] = classify_row(item)
            loaded.append(item)

    overall = _empty_bucket()
    by_dataset: dict[str, Any] = defaultdict(_empty_bucket)
    by_family: dict[str, Any] = defaultdict(_empty_family_bucket)

    for row in loaded:
        classification = row["_classification"]
        _update_bucket(overall, classification, row)
        _update_bucket(by_dataset[str(row.get("dataset", "unknown"))], classification, row)
        family = str(row.get("family") or "unknown")
        _update_bucket(by_family[family], classification, row)
        if classification in ANSWERABLE_WRONG_CLASSES:
            _append_example(by_family[family]["examples_answerable_wrong"], row)
        elif classification == "abstain_unsafe_answer":
            _append_example(by_family[family]["examples_abstain_unsafe"], row)

    finalized_overall = _finalize_bucket(overall)
    finalized_by_dataset = {name: _finalize_bucket(bucket) for name, bucket in sorted(by_dataset.items())}
    finalized_by_family = {name: _finalize_bucket(bucket) for name, bucket in sorted(by_family.items())}
    priority_actions = _build_priority_actions(finalized_by_family)
    backlog = {"p0": [], "p1": [], "p2": [], "p3": []}
    for action in priority_actions:
        backlog[action["priority"].lower()].append(action)

    return {
        "schema_version": 2,
        "created_by": "DAY2_FAILURE_REPORT_V2_POLICY_AWARE",
        "status": "PASS",
        "input_files": input_records,
        "overall": finalized_overall,
        "by_dataset": finalized_by_dataset,
        "by_family": finalized_by_family,
        "priority_actions": priority_actions,
        "repair_backlog": backlog,
        "decision": {
            "train_v2a_150_authorized": False,
            "next_action": "patch_p0_answerable_solver_families_then_rerun_solver_only",
        },
        "packaging_allowed": False,
        "submission_allowed": False,
        "leaderboard_claim": False,
        "model_results_faked": False,
    }


def write_failure_report_v2(report: dict[str, Any], out_path: str | Path) -> None:
    write_json_checked(out_path, report, field_name="day2_failure_report_v2")


def _empty_bucket() -> dict[str, Any]:
    return {
        "rows": 0,
        "answerable_rows": 0,
        "answerable_correct": 0,
        "answerable_wrong": 0,
        "answerable_accuracy": None,
        "abstain_rows": 0,
        "abstain_correct": 0,
        "abstain_unsafe_answer": 0,
        "correct_abstain_rate": None,
        "unsafe_abstain_answer_rate": 0.0,
        "class_counts": {name: 0 for name in CLASS_NAMES},
    }


def _empty_family_bucket() -> dict[str, Any]:
    bucket = _empty_bucket()
    bucket["examples_answerable_wrong"] = []
    bucket["examples_abstain_unsafe"] = []
    return bucket


def _update_bucket(bucket: dict[str, Any], classification: str, row: dict[str, Any]) -> None:
    bucket["rows"] += 1
    bucket["class_counts"][classification] += 1
    if classification.startswith("abstain_"):
        bucket["abstain_rows"] += 1
        if classification == "abstain_correct":
            bucket["abstain_correct"] += 1
        elif classification == "abstain_unsafe_answer":
            bucket["abstain_unsafe_answer"] += 1
        return
    bucket["answerable_rows"] += 1
    if classification == "answerable_correct":
        bucket["answerable_correct"] += 1
    else:
        bucket["answerable_wrong"] += 1


def _finalize_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    result = dict(bucket)
    result["class_counts"] = dict(bucket["class_counts"])
    answerable_rows = int(result["answerable_rows"])
    abstain_rows = int(result["abstain_rows"])
    result["answerable_accuracy"] = None if answerable_rows == 0 else result["answerable_correct"] / answerable_rows
    result["correct_abstain_rate"] = None if abstain_rows == 0 else result["abstain_correct"] / abstain_rows
    result["unsafe_abstain_answer_rate"] = 0.0 if abstain_rows == 0 else result["abstain_unsafe_answer"] / abstain_rows
    if "examples_answerable_wrong" in bucket:
        result["examples_answerable_wrong"] = list(bucket["examples_answerable_wrong"])
        result["examples_abstain_unsafe"] = list(bucket["examples_abstain_unsafe"])
    return result


def _build_priority_actions(by_family: dict[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for family, bucket in by_family.items():
        wrong = int(bucket["answerable_wrong"])
        if wrong <= 0:
            continue
        action = {
            "priority": _priority_for(wrong),
            "family": family,
            "answerable_wrong": wrong,
            "answerable_rows": int(bucket["answerable_rows"]),
            "answerable_accuracy": bucket["answerable_accuracy"],
            "dominant_failure": _dominant_failure(bucket["class_counts"]),
            "recommended_action": _recommended_action(family),
        }
        actions.append(action)
    actions.sort(
        key=lambda item: (
            PRIORITY_ORDER[item["priority"]],
            -int(item["answerable_wrong"]),
            1.0 if item["answerable_accuracy"] is None else float(item["answerable_accuracy"]),
            item["family"],
        )
    )
    return actions


def _priority_for(answerable_wrong: int) -> str:
    if answerable_wrong >= 50:
        return "P0"
    if answerable_wrong >= 20:
        return "P1"
    if answerable_wrong >= 5:
        return "P2"
    return "P3"


def _dominant_failure(class_counts: dict[str, int]) -> str:
    failures = {name: count for name, count in class_counts.items() if name in ANSWERABLE_WRONG_CLASSES and count > 0}
    if not failures:
        return "none"
    return sorted(failures.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _recommended_action(family: str) -> str:
    recommendations = {
        "custom_numeral": "implement custom numeral induction solver: infer digit/symbol alphabet, base/positional decode, encode/decode roundtrip, reject inconsistent alphabets.",
        "equation_operator": "implement equation/operator solver: infer binary/unary operator from examples, verify all examples, reject multi-fit operators.",
        "sequence_pattern": "implement sequence pattern solver: arithmetic/geometric/difference/alternating/small recurrence with uniqueness check.",
        "permutation_sorting": "implement permutation/sorting solver: infer index permutation or sort key, verify all examples.",
        "symbol_mapping": "patch only answerable misses; keep abstain on unseen symbols.",
        "bit_manipulation": "patch only answerable misses; keep abstain on ambiguous transforms.",
        "char_cipher": "patch only answerable misses; keep abstain on contradictory mappings.",
        "word_cipher": "patch only deterministic answerable misses; never guess semantic/unknown token mappings.",
        "unit_conversion": "audit answerable misses for rounding/unit alias bugs only.",
        "gravity_numeric": "audit answerable misses for formula/rounding bugs only.",
        "roman_numeral": "do not prioritize unless answerable_wrong > 0.",
        "numeric_formula": "do not prioritize unless answerable_wrong > 0.",
        "format_only": "do not prioritize unless answerable_wrong > 0.",
    }
    return recommendations.get(family, "inspect answerable solver misses and add deterministic repair only if examples prove a unique rule.")


def _append_example(target: list[dict[str, Any]], row: dict[str, Any], *, limit: int = 5) -> None:
    if len(target) >= limit:
        return
    target.append(
        {
            "row_id": row.get("row_id") or row.get("id"),
            "dataset": row.get("dataset"),
            "family": row.get("family"),
            "subfamily": row.get("subfamily"),
            "expected": _expected_answer(row),
            "selected": _output_text(row),
            "failure_reason": row.get("failure_reason"),
            "route_reason": row.get("route_reason"),
            "classification": row.get("_classification"),
        }
    )


def _expected_answer(row: dict[str, Any]) -> str | None:
    for key in ("expected", "gold_answer", "answer"):
        value = row.get(key)
        if value is not None:
            return str(value).strip()
    return None


def _output_text(row: dict[str, Any]) -> str:
    for key in ("normalized_answer", "extracted_answer", "selected_answer", "solver_candidate", "raw_output"):
        value = row.get(key)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return ""


def _reason_text(row: dict[str, Any]) -> str:
    values = [
        row.get("failure_reason"),
        row.get("route_reason"),
        row.get("solver_verification_status"),
        row.get("selected_source"),
    ]
    return " ".join(str(value).lower() for value in values if value is not None)


def _is_abstain(value: str | None) -> bool:
    return value is not None and value.strip().upper() == "ABSTAIN"


def _mentions_abstain(reason_text: str) -> bool:
    return "abstain" in reason_text or "all_solvers_abstained" in reason_text


def _mentions_unsupported_or_abstain(reason_text: str) -> bool:
    return (
        "abstain" in reason_text
        or "all_solvers_abstained" in reason_text
        or "no_solver" in reason_text
        or "unsupported" in reason_text
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build policy-aware Day 2 solver-only failure report V2.")
    parser.add_argument("--core-predictions", default=DEFAULT_INPUTS["core_eval"])
    parser.add_argument("--family-predictions", default=DEFAULT_INPUTS["family_eval"])
    parser.add_argument("--rule-holdout-predictions", default=DEFAULT_INPUTS["rule_holdout"])
    parser.add_argument("--anti-leak-predictions", default=DEFAULT_INPUTS["anti_leak"])
    parser.add_argument("--out", default="artifacts/sprint11/day2_failure_report_v2.json")
    args = parser.parse_args(argv)
    inputs = {
        "core_eval": args.core_predictions,
        "family_eval": args.family_predictions,
        "rule_holdout": args.rule_holdout_predictions,
        "anti_leak": args.anti_leak_predictions,
    }
    report = build_failure_report_v2(inputs)
    write_failure_report_v2(report, args.out)
    print(json.dumps({"status": report["status"], "out": args.out, "priority_actions": len(report["priority_actions"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
