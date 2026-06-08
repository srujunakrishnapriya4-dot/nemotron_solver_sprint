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

from kaggle_anti086.data.v2_corpus_io import file_record, read_json, read_jsonl, write_json_checked
from kaggle_anti086.training.day2_abstain_fallback_mining import (
    CLASS_UNSAFE,
    DEFAULT_PREDICTIONS,
    _classify_abstain_row,
    _expected,
    _has_contradictory_examples,
    _is_abstain,
    _parse_examples_and_query,
    _query_has_unseen_symbols,
)


DEFAULT_MINING_REPORT = "artifacts/sprint11/day2_abstain_fallback_mining.json"
FORENSIC_CLASSES = (
    "truly_unsafe_contradiction",
    "truly_unsafe_unseen_symbols",
    "truly_unsafe_multifit",
    "possibly_recoverable_with_new_verified_solver",
    "possibly_recoverable_with_better_parser",
    "possibly_recoverable_with_relaxed_but_verified_rule",
    "policy_artifact_or_bad_label",
)


def build_unsafe_abstain_forensic_audit(
    *,
    mining_report_path: str | Path = DEFAULT_MINING_REPORT,
    prediction_files: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    mining_path = Path(mining_report_path)
    if not mining_path.exists():
        raise FileNotFoundError(f"missing required ABSTAIN fallback mining report: {mining_path}")
    mining_report = read_json(mining_path)
    rows, prediction_inputs = _load_prediction_rows(prediction_files or DEFAULT_PREDICTIONS)
    unsafe_rows = _unsafe_rows(rows)
    classified = [_forensic_classify(row) for row in unsafe_rows]
    score_gap = _score_gap(mining_report)
    forensic_classes = _forensic_class_summary(classified)
    family_reaudit = _family_reaudit(classified)
    score_projection = _score_projection(score_gap, forensic_classes)
    authorized = [
        {
            "family": item["family"],
            "rows": item["potential_recoverable_rows"],
            "forensic_label": item["new_forensic_label"],
            "precision_risk": item["precision_risk"],
        }
        for item in family_reaudit
        if item["potential_recoverable_rows"] > 0 and item["precision_risk"] in {"LOW", "MEDIUM"}
    ]
    blocked = [
        {
            "family": item["family"],
            "rows": item["rows"],
            "forensic_label": item["new_forensic_label"],
            "precision_risk": item["precision_risk"],
        }
        for item in family_reaudit
        if item["potential_recoverable_rows"] == 0 or item["precision_risk"] == "HIGH"
    ]
    warnings: list[str] = []
    if not score_projection["can_reach_0_93_without_guessing"]:
        warnings.append("forensic_recoverable_rows_do_not_close_0_93_gap")
    if not authorized:
        warnings.append("no_solver_design_family_authorized_from_unsafe_pool")
    return {
        "schema_version": 1,
        "created_by": "DAY2_UNSAFE_ABSTAIN_FORENSIC_AUDIT",
        "status": "WARN" if warnings else "PASS",
        "inputs": {
            "abstain_fallback_mining": file_record(mining_path),
            "prediction_files": prediction_inputs,
        },
        "score_gap": score_gap,
        "unsafe_pool": _unsafe_pool_summary(unsafe_rows),
        "forensic_classes": forensic_classes,
        "family_reaudit": family_reaudit,
        "score_projection": score_projection,
        "decision": {
            "authorize_solver_design": bool(authorized),
            "authorized_families": authorized,
            "blocked_families": blocked,
            "next_action": _next_action(authorized, score_projection),
        },
        "warnings": warnings,
        "failures": [],
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


def _unsafe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        if not _is_abstain(_expected(row)):
            continue
        if _classify_abstain_row(row)["class_name"] == CLASS_UNSAFE:
            output.append(row)
    output.sort(key=lambda row: (str(row.get("family")), str(row.get("dataset")), str(row.get("row_id") or row.get("id"))))
    return output


def _score_gap(mining_report: dict[str, Any]) -> dict[str, Any]:
    target = mining_report.get("score_target", {})
    classes = mining_report.get("recoverability_classes", {})
    current = int(target.get("current_correct", 0))
    target_correct = int(target.get("target_correct_needed", 0))
    maybe = int(classes.get("maybe_recoverable_requires_new_solver", {}).get("rows", 0))
    unsafe = int(classes.get("unsafe_ambiguous_do_not_answer", {}).get("rows", 0))
    additional = max(0, target_correct - current)
    return {
        "current_correct": current,
        "target_correct_0_93": target_correct,
        "additional_needed": additional,
        "maybe_recoverable": maybe,
        "unsafe_pool": unsafe,
        "additional_needed_after_maybe": max(0, additional - maybe),
        "total_rows": int(target.get("total_rows", 0)),
    }


def _unsafe_pool_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total": len(rows),
        "by_family": dict(sorted(Counter(str(row.get("family") or "unknown") for row in rows).items())),
        "by_subfamily": dict(sorted(Counter(str(row.get("subfamily") or "unknown") for row in rows).items())),
    }


def _forensic_classify(row: dict[str, Any]) -> dict[str, Any]:
    family = str(row.get("family") or "unknown")
    prompt = str(row.get("prompt") or "")
    lowered = f"{prompt}\n{row.get('failure_reason') or ''}".lower()
    if any(marker in lowered for marker in ("policy_artifact", "bad_label", "evaluator")):
        cls = "policy_artifact_or_bad_label"
        risk = "MEDIUM"
        reason = "policy_or_label_artifact_marker"
        action = "audit_label_or_policy_before_solver_work"
    elif _has_contradictory_examples(prompt):
        cls = "truly_unsafe_contradiction"
        risk = "HIGH"
        reason = "same_source_or_symbol_has_conflicting_targets"
        action = "keep_abstain_do_not_guess"
    elif family == "custom_numeral" and _custom_numeral_unique(prompt):
        cls = "possibly_recoverable_with_new_verified_solver"
        risk = "MEDIUM"
        reason = "custom_numeral_examples_are_unique_but_solver_missing"
        action = "design_custom_numeral_solver_with_verification"
    elif family == "bit_manipulation" and _known_bit_operation_marker(prompt):
        cls = "possibly_recoverable_with_new_verified_solver"
        risk = "MEDIUM"
        reason = "known_bit_operation_marker_without_solver_coverage"
        action = "design_targeted_bit_operation_solver"
    elif family == "permutation_sorting" and "sort by hidden rule" in lowered:
        cls = "possibly_recoverable_with_relaxed_but_verified_rule"
        risk = "MEDIUM"
        reason = "sort_rule_structure_may_be_verifiable_but_hidden"
        action = "test_sort_rule_solver_with_ambiguity_guard"
    elif family in {"gravity_numeric", "unit_conversion"} and any(marker in lowered for marker in ("round", "alias", "unit")):
        cls = "possibly_recoverable_with_better_parser"
        risk = "MEDIUM"
        reason = "numeric_or_unit_parser_gap_marker"
        action = "audit_parser_alias_or_rounding_rule"
    elif _query_has_unseen_symbols(prompt):
        cls = "truly_unsafe_unseen_symbols"
        risk = "HIGH"
        reason = "query_contains_symbols_or_tokens_not_supported_by_examples"
        action = "keep_abstain_unseen_query_material"
    elif family in {"bit_manipulation", "gravity_numeric", "unit_conversion"}:
        cls = "truly_unsafe_multifit"
        risk = "HIGH"
        reason = "multiple_transformations_or_formulas_fit_examples"
        action = "keep_abstain_multifit_row"
    elif family == "word_cipher" and "unknown" in lowered:
        cls = "truly_unsafe_unseen_symbols"
        risk = "HIGH"
        reason = "encrypted_query_contains_unknown_token"
        action = "keep_abstain_unknown_token"
    else:
        cls = "truly_unsafe_multifit"
        risk = "HIGH"
        reason = "no_unique_verified_recovery_rule_identified"
        action = "keep_abstain_until_specific_solver_evidence_exists"
    return {
        "row": row,
        "family": family,
        "subfamily": str(row.get("subfamily") or "unknown"),
        "forensic_class": cls,
        "precision_risk": risk,
        "classification_reason": reason,
        "recommended_action": action,
    }


def _forensic_class_summary(classified: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for cls in FORENSIC_CLASSES:
        items = [item for item in classified if item["forensic_class"] == cls]
        output[cls] = {
            "rows": len(items),
            "max_possible_gain": len(items),
            "by_family": dict(sorted(Counter(item["family"] for item in items).items())),
            "by_subfamily": dict(sorted(Counter(item["subfamily"] for item in items).items())),
            "example_rows": [_example(item) for item in items[:8]],
        }
    return output


def _family_reaudit(classified: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in classified:
        grouped[item["family"]].append(item)
    output: list[dict[str, Any]] = []
    for family, items in sorted(grouped.items()):
        potential = [item for item in items if item["forensic_class"].startswith("possibly_recoverable")]
        risk = _max_risk(item["precision_risk"] for item in items)
        label = _majority_label(item["forensic_class"] for item in items)
        output.append(
            {
                "family": family,
                "rows": len(items),
                "current_label": "unsafe",
                "new_forensic_label": label,
                "potential_recoverable_rows": len(potential),
                "precision_risk": risk,
                "recommended_next_action": _family_action(label, risk, len(potential)),
                "example_rows": [_example(item) for item in items[:6]],
            }
        )
    return output


def _score_projection(score_gap: dict[str, Any], classes: dict[str, Any]) -> dict[str, Any]:
    total = int(score_gap.get("total_rows", 0))
    current = int(score_gap["current_correct"])
    maybe = int(score_gap["maybe_recoverable"])
    low = sum(
        classes[name]["rows"]
        for name in (
            "possibly_recoverable_with_new_verified_solver",
            "possibly_recoverable_with_better_parser",
            "possibly_recoverable_with_relaxed_but_verified_rule",
        )
        if _class_risk(name) == "LOW"
    )
    medium = sum(
        classes[name]["rows"]
        for name in (
            "possibly_recoverable_with_new_verified_solver",
            "possibly_recoverable_with_better_parser",
            "possibly_recoverable_with_relaxed_but_verified_rule",
        )
        if _class_risk(name) in {"LOW", "MEDIUM"}
    )
    if total <= 0:
        return {
            "if_only_maybe_rows_solved": None,
            "if_low_risk_forensic_rows_solved": None,
            "if_medium_risk_rows_solved": None,
            "max_plausible_without_model_guessing": None,
            "can_reach_0_93_without_guessing": False,
        }
    only_maybe = (current + maybe) / total
    low_score = (current + maybe + low) / total
    medium_score = (current + maybe + medium) / total
    max_score = medium_score
    return {
        "if_only_maybe_rows_solved": only_maybe,
        "if_low_risk_forensic_rows_solved": low_score,
        "if_medium_risk_rows_solved": medium_score,
        "max_plausible_without_model_guessing": max_score,
        "can_reach_0_93_without_guessing": max_score >= 0.93,
    }


def _class_risk(class_name: str) -> str:
    # Current forensic recoverability classes are medium-risk by construction:
    # they authorize solver design only, not answer emission.
    return "MEDIUM" if class_name.startswith("possibly_recoverable") else "HIGH"


def _custom_numeral_unique(prompt: str) -> bool:
    examples, _query = _parse_examples_and_query(prompt)
    if not examples or _has_contradictory_examples(prompt):
        return False
    sources = [source for source, _target in examples]
    targets = [target for _source, target in examples]
    return len(set(sources)) == len(sources) and len(set(targets)) == len(targets)


def _known_bit_operation_marker(prompt: str) -> bool:
    lowered = prompt.lower()
    return any(marker in lowered for marker in ("rotate", "reverse bits", "xor", "parity", "popcount", "count ones", "bitwise not", "complement"))


def _example(item: dict[str, Any]) -> dict[str, Any]:
    row = item["row"]
    return {
        "dataset": row.get("dataset"),
        "row_id": row.get("row_id") or row.get("id"),
        "family": row.get("family"),
        "subfamily": row.get("subfamily"),
        "prompt": row.get("prompt"),
        "failure_reason": row.get("failure_reason"),
        "forensic_class": item["forensic_class"],
        "classification_reason": item["classification_reason"],
    }


def _max_risk(values: Any) -> str:
    order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    risks = list(values)
    return max(risks, key=lambda value: order.get(value, 2)) if risks else "HIGH"


def _majority_label(values: Any) -> str:
    counts = Counter(values)
    if not counts:
        return "unknown"
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _family_action(label: str, risk: str, potential: int) -> str:
    if potential and risk in {"LOW", "MEDIUM"}:
        return "authorize_solver_design_only_with_strict_verification"
    if label.startswith("truly_unsafe"):
        return "keep_abstain_do_not_answer"
    return "keep_blocked_until_more_evidence"


def _next_action(authorized: list[dict[str, Any]], score_projection: dict[str, Any]) -> str:
    if not authorized:
        return "do_not_design_solver_from_unsafe_pool"
    if not score_projection["can_reach_0_93_without_guessing"]:
        return "solver_design_may_reduce_risk_but_cannot_close_0_93_gap_without_more_evidence"
    return "design_authorized_solvers_then_rerun_day2_evidence"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Forensically reaudit Day2 unsafe ABSTAIN rows.")
    parser.add_argument("--mining-report", default=DEFAULT_MINING_REPORT)
    parser.add_argument("--core-predictions", default=DEFAULT_PREDICTIONS["core_eval"])
    parser.add_argument("--family-predictions", default=DEFAULT_PREDICTIONS["family_eval"])
    parser.add_argument("--rule-holdout-predictions", default=DEFAULT_PREDICTIONS["rule_holdout"])
    parser.add_argument("--anti-leak-predictions", default=DEFAULT_PREDICTIONS["anti_leak"])
    parser.add_argument("--out", default="artifacts/sprint11/day2_unsafe_abstain_forensic_audit.json")
    args = parser.parse_args(argv)
    report = build_unsafe_abstain_forensic_audit(
        mining_report_path=args.mining_report,
        prediction_files={
            "core_eval": args.core_predictions,
            "family_eval": args.family_predictions,
            "rule_holdout": args.rule_holdout_predictions,
            "anti_leak": args.anti_leak_predictions,
        },
    )
    write_json_checked(args.out, report, field_name="day2_unsafe_abstain_forensic_audit")
    print(
        json.dumps(
            {
                "status": report["status"],
                "out": args.out,
                "score_projection": report["score_projection"],
                "authorize_solver_design": report["decision"]["authorize_solver_design"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
