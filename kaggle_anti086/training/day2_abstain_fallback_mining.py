from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.v2_corpus_io import file_record, read_json, read_jsonl, write_json_checked


DEFAULT_PREDICTIONS = {
    "core_eval": "artifacts/sprint11/day2_predictions/core_eval_solver_only_predictions.jsonl",
    "family_eval": "artifacts/sprint11/day2_predictions/family_eval_solver_only_predictions.jsonl",
    "rule_holdout": "artifacts/sprint11/day2_predictions/rule_holdout_solver_only_predictions.jsonl",
    "anti_leak": "artifacts/sprint11/day2_predictions/anti_leak_solver_only_predictions.jsonl",
}
DEFAULT_RESOLUTION_REPORT = "artifacts/sprint11/day2_abstain_resolution_report.json"
DEFAULT_FINAL_SOLVER_REPORT = "artifacts/sprint11/day2_final_solver_score_report.json"
DEFAULT_FAILURE_REPORT_V2 = "artifacts/sprint11/day2_failure_report_v2.json"
TARGET_RAW_ACCURACY = 0.93

CLASS_DETERMINISTIC = "deterministically_recoverable"
CLASS_MAYBE = "maybe_recoverable_requires_new_solver"
CLASS_UNSAFE = "unsafe_ambiguous_do_not_answer"
CLASS_MODEL = "requires_model_fallback_but_unverified"
CLASS_ARTIFACT = "evaluator_or_policy_artifact"


def build_abstain_fallback_mining_report(
    *,
    prediction_files: dict[str, str | Path] | None = None,
    resolution_report_path: str | Path = DEFAULT_RESOLUTION_REPORT,
    final_solver_report_path: str | Path = DEFAULT_FINAL_SOLVER_REPORT,
    failure_report_v2_path: str | Path = DEFAULT_FAILURE_REPORT_V2,
    target_raw_accuracy: float = TARGET_RAW_ACCURACY,
) -> dict[str, Any]:
    paths = prediction_files or DEFAULT_PREDICTIONS
    rows, inputs = _load_prediction_rows(paths)
    abstain_rows = [row for row in rows if _is_abstain(_expected(row))]
    final_report = _read_optional(final_solver_report_path)
    resolution_report = _read_optional(resolution_report_path)
    failure_report_v2 = _read_optional(failure_report_v2_path)
    score_target = _score_target(rows, abstain_rows, final_report=final_report, target_raw_accuracy=target_raw_accuracy)
    classified = [_classify_abstain_row(row) for row in abstain_rows]
    classes = _class_summary(classified)
    candidates = _candidate_fallback_families(classified, total_abstain=len(abstain_rows))
    deterministic_low_risk = [
        item for item in candidates
        if item["recoverability"] == "deterministic" and item["estimated_precision_risk"] == "LOW"
    ]
    deterministic_gain = classes[CLASS_DETERMINISTIC]["rows"]
    warnings: list[str] = []
    failures: list[str] = []
    if not abstain_rows:
        warnings.append("no_abstain_rows_found")
    if deterministic_gain < score_target["additional_correct_needed"]:
        warnings.append("deterministic_recoverable_rows_do_not_close_0_93_gap")
    if resolution_report and resolution_report.get("policy_decision", {}).get("decision") == "ABSTAIN_NOT_ALLOWED_NEEDS_FALLBACK":
        warnings.append("abstain_resolution_requires_fallback")
    status = "FAIL" if failures else ("WARN" if warnings else "PASS")
    authorized_families = [
        {
            "family": item["family"],
            "subfamily": item["subfamily"],
            "rows": item["rows"],
        }
        for item in deterministic_low_risk
    ]
    blocked_families = [
        {
            "family": item["family"],
            "subfamily": item["subfamily"],
            "recoverability": item["recoverability"],
            "rows": item["rows"],
        }
        for item in candidates
        if item["recoverability"] != "deterministic" or item["estimated_precision_risk"] != "LOW"
    ]
    return {
        "schema_version": 1,
        "created_by": "DAY2_ABSTAIN_FALLBACK_MINING",
        "status": status,
        "inputs": {
            "prediction_files": inputs,
            "abstain_resolution_report": _optional_record(resolution_report_path),
            "final_solver_score_report": _optional_record(final_solver_report_path),
            "failure_report_v2": _optional_record(failure_report_v2_path),
        },
        "score_target": score_target,
        "abstain_pool": _abstain_pool_summary(abstain_rows),
        "recoverability_classes": classes,
        "candidate_fallback_families": candidates,
        "unsafe_constraints": [
            "Do not answer contradictory mappings",
            "Do not answer unseen symbols unless inferable",
            "Do not answer multiple-valid-transform rows",
            "Do not use base/adapter fallback without measured precision",
        ],
        "decision": {
            "fallback_implementation_authorized": bool(deterministic_low_risk),
            "authorized_families": authorized_families,
            "blocked_families": blocked_families,
            "next_action": _next_action(bool(deterministic_low_risk), deterministic_gain, score_target),
        },
        "warnings": warnings,
        "failures": failures,
        "v2a_150_authorized": False,
        "package_authorized": False,
        "submission_authorized": False,
        "leaderboard_claim": False,
        "no_0_93_evidence": True,
        "no_0_95_evidence": True,
        "model_results_faked": False,
    }


def _load_prediction_rows(paths: dict[str, str | Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    inputs: dict[str, Any] = {}
    for dataset, path_value in sorted(paths.items()):
        path = Path(path_value)
        if not path.exists():
            raise FileNotFoundError(f"missing required solver-only prediction file for {dataset}: {path}")
        dataset_rows = read_jsonl(path)
        inputs[dataset] = file_record(path, row_count=len(dataset_rows))
        for row in dataset_rows:
            item = dict(row)
            item.setdefault("dataset", dataset)
            rows.append(item)
    return rows, inputs


def _score_target(
    rows: list[dict[str, Any]],
    abstain_rows: list[dict[str, Any]],
    *,
    final_report: dict[str, Any] | None,
    target_raw_accuracy: float,
) -> dict[str, Any]:
    total_rows = len(rows)
    current_correct = sum(1 for row in rows if row.get("correct") is True)
    abstain_count = len(abstain_rows)
    if final_report:
        overall = final_report.get("overall", {})
        total_rows = int(overall.get("rows", total_rows) or total_rows)
        raw = overall.get("raw_exact_match")
        if raw is not None:
            current_correct = int(round(float(raw) * total_rows))
        else:
            current_correct = int(overall.get("answerable_correct", current_correct) or current_correct)
        abstain_count = int(overall.get("abstain_rows", abstain_count) or abstain_count)
    target_correct_needed = int(math.ceil(total_rows * target_raw_accuracy))
    additional = max(0, target_correct_needed - current_correct)
    return {
        "total_rows": total_rows,
        "current_correct": current_correct,
        "target_raw_accuracy": target_raw_accuracy,
        "target_correct_needed": target_correct_needed,
        "additional_correct_needed": additional,
        "abstain_rows": abstain_count,
        "required_abstain_recovery_rate": None if abstain_count == 0 else additional / abstain_count,
    }


def _abstain_pool_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total": len(rows),
        "by_dataset": dict(sorted(Counter(str(row.get("dataset") or "unknown") for row in rows).items())),
        "by_family": dict(sorted(Counter(str(row.get("family") or "unknown") for row in rows).items())),
        "by_subfamily": dict(sorted(Counter(str(row.get("subfamily") or "unknown") for row in rows).items())),
        "by_route_reason": dict(sorted(Counter(str(row.get("route_reason") or "unknown") for row in rows).items())),
        "by_failure_reason": dict(sorted(Counter(str(row.get("failure_reason") or "unknown") for row in rows).items())),
    }


def _classify_abstain_row(row: dict[str, Any]) -> dict[str, Any]:
    family = str(row.get("family") or "unknown")
    subfamily = str(row.get("subfamily") or "unknown")
    prompt = str(row.get("prompt") or "")
    failure = str(row.get("failure_reason") or "")
    lowered = f"{prompt}\n{failure}".lower()
    if any(marker in lowered for marker in ("evaluator", "policy_artifact", "format artifact")):
        class_name = CLASS_ARTIFACT
        risk = "MEDIUM"
        action = "audit_evaluator_or_policy_before_answering"
        reason = "policy_or_evaluator_artifact_marker"
    elif any(marker in lowered for marker in ("model_fallback", "model-only", "language model")):
        class_name = CLASS_MODEL
        risk = "HIGH"
        action = "measure_model_fallback_precision_before_use"
        reason = "requires_unverified_model_fallback"
    elif _has_contradictory_examples(prompt):
        class_name = CLASS_UNSAFE
        risk = "HIGH"
        action = "keep_abstain_contradictory_examples"
        reason = "contradictory_training_examples"
    elif _query_has_unseen_symbols(prompt):
        class_name = CLASS_UNSAFE
        risk = "HIGH"
        action = "keep_abstain_unseen_query_symbols"
        reason = "query_contains_unseen_symbols"
    elif family in {"bit_manipulation", "gravity_numeric", "unit_conversion"} and subfamily == "expected_abstain":
        class_name = CLASS_UNSAFE
        risk = "HIGH"
        action = "keep_abstain_multiple_transformations_fit_examples"
        reason = "multiple_valid_transformations_possible"
    elif "not_implemented" in failure or family in {"custom_numeral", "equation_operator", "sequence_pattern", "permutation_sorting"}:
        class_name = CLASS_MAYBE
        risk = "HIGH"
        action = "requires_separate_solver_design_and_verification"
        reason = "family_requires_new_solver_not_fallback_guess"
    elif _has_unique_character_mapping(prompt):
        class_name = CLASS_DETERMINISTIC
        risk = "LOW"
        action = "candidate_for_verified_mapping_fallback"
        reason = "examples_imply_unique_character_mapping"
    else:
        class_name = CLASS_UNSAFE
        risk = "HIGH"
        action = "keep_abstain_no_unique_deterministic_rule"
        reason = "no_verified_unique_fallback_rule"
    return {
        "row": row,
        "class_name": class_name,
        "family": family,
        "subfamily": subfamily,
        "estimated_precision_risk": risk,
        "recommended_action": action,
        "reason": reason,
    }


def _class_summary(classified: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for class_name in (CLASS_DETERMINISTIC, CLASS_MAYBE, CLASS_UNSAFE, CLASS_MODEL, CLASS_ARTIFACT):
        items = [item for item in classified if item["class_name"] == class_name]
        output[class_name] = {
            "rows": len(items),
            "max_possible_gain": len(items),
            "by_family": dict(sorted(Counter(item["family"] for item in items).items())),
            "by_subfamily": dict(sorted(Counter(item["subfamily"] for item in items).items())),
            "example_rows": [_example_row(item["row"], reason=item["reason"]) for item in items[:8]],
        }
    return output


def _candidate_fallback_families(classified: list[dict[str, Any]], *, total_abstain: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in classified:
        grouped[(item["family"], item["subfamily"], item["class_name"])].append(item)
    rows: list[dict[str, Any]] = []
    for (family, subfamily, class_name), items in sorted(grouped.items()):
        recoverability = {
            CLASS_DETERMINISTIC: "deterministic",
            CLASS_MAYBE: "maybe",
            CLASS_UNSAFE: "unsafe",
            CLASS_MODEL: "model_fallback",
            CLASS_ARTIFACT: "artifact",
        }[class_name]
        risk = _max_risk(item["estimated_precision_risk"] for item in items)
        rows.append(
            {
                "family": family,
                "subfamily": subfamily,
                "rows": len(items),
                "recoverability": recoverability,
                "estimated_precision_risk": risk,
                "estimated_recall": None if total_abstain == 0 else len(items) / total_abstain,
                "recommended_action": _group_action(recoverability, risk),
                "example_rows": [_example_row(item["row"], reason=item["reason"]) for item in items[:5]],
            }
        )
    return rows


def _has_contradictory_examples(prompt: str) -> bool:
    examples, _query = _parse_examples_and_query(prompt)
    seen: dict[str, str] = {}
    for source, target in examples:
        previous = seen.get(source)
        if previous is not None and previous != target:
            return True
        seen[source] = target
    char_map: dict[str, str] = {}
    for source, target in examples:
        if len(source) != len(target):
            continue
        for src, dst in zip(source, target):
            previous = char_map.get(src)
            if previous is not None and previous != dst:
                return True
            char_map[src] = dst
    return False


def _query_has_unseen_symbols(prompt: str) -> bool:
    examples, query = _parse_examples_and_query(prompt)
    if not examples or not query:
        return False
    if query in {source for source, _target in examples}:
        return False
    char_map: dict[str, str] = {}
    for source, target in examples:
        if len(source) != len(target):
            continue
        for src, dst in zip(source, target):
            char_map[src] = dst
    if not char_map:
        return False
    return any(char not in char_map for char in query)


def _has_unique_character_mapping(prompt: str) -> bool:
    examples, query = _parse_examples_and_query(prompt)
    if len(examples) < 2 or not query:
        return False
    if _has_contradictory_examples(prompt) or _query_has_unseen_symbols(prompt):
        return False
    char_map: dict[str, str] = {}
    for source, target in examples:
        if len(source) != len(target):
            return False
        for src, dst in zip(source, target):
            char_map[src] = dst
    return bool(char_map) and all(char in char_map for char in query)


def _parse_examples_and_query(prompt: str) -> tuple[list[tuple[str, str]], str | None]:
    cleaned = re.sub(r"^\[[^\]]+\]\s*", "", prompt.strip())
    examples = [
        (match.group("source").strip(), match.group("target").strip())
        for match in re.finditer(r"(?P<source>[^\s;,.?]+)\s*->\s*(?P<target>[^\s;,.?]+)", cleaned)
    ]
    query_patterns = (
        r"(?:query|input|target number|target)\s*[:?]\s*(?P<query>[^\s;,.?]+)",
        r"Now solve:\s*(?P<query>[^\s;,.?]+)",
    )
    query: str | None = None
    for pattern in query_patterns:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            query = match.group("query").strip()
            break
    return examples, query


def _example_row(row: dict[str, Any], *, reason: str) -> dict[str, Any]:
    return {
        "dataset": row.get("dataset"),
        "row_id": row.get("row_id") or row.get("id"),
        "family": row.get("family"),
        "subfamily": row.get("subfamily"),
        "prompt": row.get("prompt"),
        "failure_reason": row.get("failure_reason"),
        "route_reason": row.get("route_reason"),
        "classification_reason": reason,
    }


def _max_risk(values: Any) -> str:
    order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    risks = list(values)
    return max(risks, key=lambda value: order.get(value, 2)) if risks else "HIGH"


def _group_action(recoverability: str, risk: str) -> str:
    if recoverability == "deterministic" and risk == "LOW":
        return "eligible_for_focused_fallback_design_after_tests"
    if recoverability == "maybe":
        return "design_new_solver_then_verify_before_answering"
    if recoverability == "model_fallback":
        return "measure_model_fallback_precision_before_any_use"
    if recoverability == "artifact":
        return "audit_policy_or_evaluator_artifact"
    return "do_not_answer_keep_blocked"


def _next_action(authorized: bool, deterministic_gain: int, score_target: dict[str, Any]) -> str:
    if not authorized:
        return "do_not_implement_fallback_without_low_risk_deterministic_subset"
    if deterministic_gain < score_target["additional_correct_needed"]:
        return "implement_low_risk_fallback_subset_then_reassess_remaining_gap"
    return "implement_low_risk_fallback_subset_with_strict_regression_tests"


def _expected(row: dict[str, Any]) -> str | None:
    for key in ("expected", "gold_answer", "answer"):
        if row.get(key) is not None:
            return str(row[key]).strip()
    return None


def _is_abstain(value: str | None) -> bool:
    return value is not None and value.strip().upper() == "ABSTAIN"


def _read_optional(path: str | Path) -> dict[str, Any] | None:
    target = Path(path)
    if not target.exists():
        return None
    try:
        return read_json(target)
    except Exception:
        return None


def _optional_record(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    return file_record(target) if target.exists() else {"path": str(target), "exists": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mine Day2 ABSTAIN rows for deterministic fallback recoverability.")
    parser.add_argument("--core-predictions", default=DEFAULT_PREDICTIONS["core_eval"])
    parser.add_argument("--family-predictions", default=DEFAULT_PREDICTIONS["family_eval"])
    parser.add_argument("--rule-holdout-predictions", default=DEFAULT_PREDICTIONS["rule_holdout"])
    parser.add_argument("--anti-leak-predictions", default=DEFAULT_PREDICTIONS["anti_leak"])
    parser.add_argument("--abstain-resolution-report", default=DEFAULT_RESOLUTION_REPORT)
    parser.add_argument("--final-solver-report", default=DEFAULT_FINAL_SOLVER_REPORT)
    parser.add_argument("--failure-report-v2", default=DEFAULT_FAILURE_REPORT_V2)
    parser.add_argument("--out", default="artifacts/sprint11/day2_abstain_fallback_mining.json")
    args = parser.parse_args(argv)
    report = build_abstain_fallback_mining_report(
        prediction_files={
            "core_eval": args.core_predictions,
            "family_eval": args.family_predictions,
            "rule_holdout": args.rule_holdout_predictions,
            "anti_leak": args.anti_leak_predictions,
        },
        resolution_report_path=args.abstain_resolution_report,
        final_solver_report_path=args.final_solver_report,
        failure_report_v2_path=args.failure_report_v2,
    )
    write_json_checked(args.out, report, field_name="day2_abstain_fallback_mining")
    print(
        json.dumps(
            {
                "status": report["status"],
                "out": args.out,
                "score_target": report["score_target"],
                "fallback_implementation_authorized": report["decision"]["fallback_implementation_authorized"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
