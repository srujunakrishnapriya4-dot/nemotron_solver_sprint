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

from kaggle_anti086.data.v2_corpus_io import file_record, read_json, read_jsonl, write_json_checked


DEFAULT_FINAL_SOLVER_REPORT = "artifacts/sprint11/day2_final_solver_score_report.json"
DEFAULT_ABSTAIN_POLICY_GATE = "artifacts/sprint11/day2_abstain_policy_gate.json"
DEFAULT_PREDICTIONS = {
    "core_eval": "artifacts/sprint11/day2_predictions/core_eval_solver_only_predictions.jsonl",
    "family_eval": "artifacts/sprint11/day2_predictions/family_eval_solver_only_predictions.jsonl",
    "rule_holdout": "artifacts/sprint11/day2_predictions/rule_holdout_solver_only_predictions.jsonl",
    "anti_leak": "artifacts/sprint11/day2_predictions/anti_leak_solver_only_predictions.jsonl",
}
POLICY_FILES = {
    "answer_extractor": "src/nemotron_engine/scoring/answer_extractor.py",
    "format_policy": "src/nemotron_engine/scoring/format_policy.py",
    "local_scorer": "src/nemotron_engine/scoring/local_scorer.py",
    "submission_validator": "src/nemotron_engine/submission/submission_validator.py",
    "serving_config": "src/nemotron_engine/runtime/serving_config.py",
}


def build_abstain_resolution_report(
    *,
    final_solver_report_path: str | Path = DEFAULT_FINAL_SOLVER_REPORT,
    abstain_policy_gate_path: str | Path = DEFAULT_ABSTAIN_POLICY_GATE,
    prediction_files: dict[str, str | Path] | None = None,
    repo_root: str | Path = ".",
    max_examples: int = 12,
) -> dict[str, Any]:
    final_path = Path(final_solver_report_path)
    if not final_path.exists():
        raise FileNotFoundError(f"missing required final solver score report: {final_path}")

    final_report = read_json(final_path)
    gate_path = Path(abstain_policy_gate_path)
    gate_report = read_json(gate_path) if gate_path.exists() else None
    predictions = prediction_files or DEFAULT_PREDICTIONS
    abstain_rows, prediction_inputs, prediction_warnings = _load_abstain_rows(predictions, max_examples=max_examples)
    policy = inspect_submission_policy(repo_root)
    policy_decision = _decide_policy(policy, abstain_total=len(abstain_rows))
    fallback_need = _fallback_need(policy_decision, abstain_rows)
    score_interpretation = _score_interpretation(final_report, policy_decision)
    failures: list[str] = []
    warnings = list(prediction_warnings)

    overall = final_report.get("overall", {})
    if _float(overall.get("unsafe_abstain_answer_rate", 0.0)) > 0:
        failures.append("unsafe_abstain_answer_rate_nonzero")
    if len(abstain_rows) == 0 and int(overall.get("abstain_rows", 0) or 0) > 0:
        warnings.append("abstain_prediction_rows_unavailable_counts_from_final_report_only")
    if policy_decision["decision"] == "ABSTAIN_POLICY_UNKNOWN_BLOCK_SUBMISSION":
        warnings.append("final_submission_abstain_policy_unknown")
    if policy_decision["decision"] == "ABSTAIN_NOT_ALLOWED_NEEDS_FALLBACK":
        warnings.append("abstain_or_blank_rejected_by_repository_submission_policy")

    status = "FAIL" if failures else ("WARN" if warnings else "PASS")
    next_action = _next_action(policy_decision)
    return {
        "schema_version": 1,
        "created_by": "DAY2_ABSTAIN_RESOLUTION_REPORT",
        "status": status,
        "inputs": {
            "final_solver_score_report": file_record(final_path),
            "abstain_policy_gate": file_record(gate_path) if gate_path.exists() else {"path": str(gate_path), "exists": False},
            "prediction_files": prediction_inputs,
        },
        "submission_policy_evidence": policy,
        "abstain_rows": _abstain_row_summary(abstain_rows, final_report=final_report),
        "policy_decision": policy_decision,
        "fallback_need": fallback_need,
        "score_interpretation": score_interpretation,
        "decision": {
            "v2a_150_authorized": False,
            "package_authorized": False,
            "submission_authorized": False,
            "next_action": next_action,
        },
        "warnings": warnings,
        "failures": failures,
        "model_results_faked": False,
        "leaderboard_claim": False,
        "no_0_93_evidence": True,
        "no_0_95_evidence": True,
    }


def inspect_submission_policy(repo_root: str | Path = ".") -> dict[str, Any]:
    root = Path(repo_root)
    evidence_notes: list[str] = []
    found: dict[str, bool] = {}
    texts: dict[str, str] = {}
    for name, relative in POLICY_FILES.items():
        path = root / relative
        exists = path.exists()
        found[name] = exists
        if not exists:
            evidence_notes.append(f"{name}_not_found:{relative}")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        texts[name] = text
        evidence_notes.append(f"{name}_found:{relative}")

    joined = "\n".join(texts.values())
    lowered = joined.lower()
    explicit_abstain_support = any(
        marker in lowered
        for marker in (
            "abstain_allowed = true",
            "allow_abstain = true",
            "supports_abstain = true",
            "abstain_literal_supported = true",
        )
    )
    explicit_blank_support = any(
        marker in lowered
        for marker in (
            "blank_answer_supported = true",
            "allow_blank_answer = true",
            "allow_blank = true",
        )
    )
    concrete_answer_only = any(
        marker in joined
        for marker in (
            "normalize_answer_candidate",
            "answer: int",
            "Answer cannot be empty",
            "Boxed answer cannot be empty",
            "Unsafe non-integer answer",
            "Final answer must appear in exactly one",
        )
    )
    boxed_required = None
    if "extract_boxed_answer" in joined or "\\boxed" in joined:
        boxed_required = True
    if explicit_abstain_support and not concrete_answer_only:
        abstain_literal_supported: bool | None = True
    elif concrete_answer_only:
        abstain_literal_supported = False
    else:
        abstain_literal_supported = None
    if explicit_blank_support and not concrete_answer_only:
        blank_answer_supported: bool | None = True
    elif concrete_answer_only:
        blank_answer_supported = False
    else:
        blank_answer_supported = None

    if concrete_answer_only:
        evidence_notes.append("repository_policy_requires_non_empty_concrete_answer")
    if explicit_abstain_support:
        evidence_notes.append("repository_policy_contains_explicit_abstain_support_marker")
    if explicit_blank_support:
        evidence_notes.append("repository_policy_contains_explicit_blank_support_marker")
    if boxed_required:
        evidence_notes.append("local_scorer_or_extractor_requires_single_boxed_answer")
    return {
        "submission_validator_found": found["submission_validator"],
        "format_policy_found": found["format_policy"],
        "local_scorer_found": found["local_scorer"],
        "abstain_literal_supported": abstain_literal_supported,
        "blank_answer_supported": blank_answer_supported,
        "single_boxed_answer_required": boxed_required,
        "policy_files": {
            name: {"path": relative, "exists": found[name]}
            for name, relative in POLICY_FILES.items()
        },
        "evidence_notes": sorted(evidence_notes),
    }


def _load_abstain_rows(
    prediction_files: dict[str, str | Path],
    *,
    max_examples: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    rows: list[dict[str, Any]] = []
    inputs: dict[str, Any] = {}
    warnings: list[str] = []
    for dataset, path_value in sorted(prediction_files.items()):
        path = Path(path_value)
        if not path.exists():
            inputs[dataset] = {"path": str(path), "exists": False}
            warnings.append(f"missing_solver_prediction_file:{dataset}")
            continue
        dataset_rows = read_jsonl(path)
        inputs[dataset] = file_record(path, row_count=len(dataset_rows))
        for row in dataset_rows:
            if _is_abstain(_expected(row)):
                item = dict(row)
                item.setdefault("dataset", dataset)
                rows.append(item)
    rows.sort(key=lambda row: (str(row.get("dataset", "")), str(row.get("family", "")), str(row.get("row_id") or row.get("id") or "")))
    return rows, inputs, warnings


def _abstain_row_summary(abstain_rows: list[dict[str, Any]], *, final_report: dict[str, Any]) -> dict[str, Any]:
    by_dataset = Counter(str(row.get("dataset") or "unknown") for row in abstain_rows)
    by_family = Counter(str(row.get("family") or "unknown") for row in abstain_rows)
    total = len(abstain_rows)
    if total == 0:
        total = int(final_report.get("overall", {}).get("abstain_rows", 0) or 0)
    return {
        "total": total,
        "by_dataset": dict(sorted(by_dataset.items())),
        "by_family": dict(sorted(by_family.items())),
        "example_rows": [_example_row(row) for row in abstain_rows[:12]],
    }


def _example_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": row.get("dataset"),
        "row_id": row.get("row_id") or row.get("id"),
        "family": row.get("family"),
        "subfamily": row.get("subfamily"),
        "prompt": row.get("prompt"),
        "gold_answer": _expected(row),
        "raw_output": row.get("raw_output"),
        "extracted_answer": row.get("extracted_answer"),
        "normalized_answer": row.get("normalized_answer"),
        "correct": row.get("correct"),
        "valid": row.get("valid"),
        "failure_reason": row.get("failure_reason"),
        "route_reason": row.get("route_reason"),
    }


def _decide_policy(policy: dict[str, Any], *, abstain_total: int) -> dict[str, Any]:
    reasons: list[str] = []
    abstain_supported = policy.get("abstain_literal_supported")
    blank_supported = policy.get("blank_answer_supported")
    boxed_required = policy.get("single_boxed_answer_required")
    if abstain_supported is True or blank_supported is True:
        decision = "ABSTAIN_ALLOWED"
        known = True
        reasons.append("repository_policy_explicitly_supports_abstain_or_blank")
    elif abstain_supported is False or blank_supported is False or boxed_required is True:
        decision = "ABSTAIN_NOT_ALLOWED_NEEDS_FALLBACK"
        known = True
        reasons.append("repository_policy_rejects_blank_or_abstain_outputs")
        if boxed_required:
            reasons.append("single_non_empty_boxed_answer_required_by_local_scorer")
        if abstain_total > 0:
            reasons.append("abstain_rows_present_require_fallback_before_submission")
    else:
        decision = "ABSTAIN_POLICY_UNKNOWN_BLOCK_SUBMISSION"
        known = False
        reasons.append("insufficient_repository_evidence_for_final_abstain_scoring")
        if abstain_total > 0:
            reasons.append("abstain_rows_present_under_unknown_policy")
    return {
        "final_submission_policy_known": known,
        "decision": decision,
        "reason_codes": sorted(set(reasons)),
    }


def _fallback_need(policy_decision: dict[str, Any], abstain_rows: list[dict[str, Any]]) -> dict[str, Any]:
    families = sorted({str(row.get("family") or "unknown") for row in abstain_rows})
    decision = policy_decision["decision"]
    fallback_required = decision == "ABSTAIN_NOT_ALLOWED_NEEDS_FALLBACK" and bool(abstain_rows)
    unsafe = families if fallback_required else []
    notes = ["fallback_not_implemented_in_this_task", "do_not_guess_abstain_rows"]
    if decision == "ABSTAIN_POLICY_UNKNOWN_BLOCK_SUBMISSION":
        notes.append("resolve_final_policy_before_designing_fallback")
    if fallback_required:
        notes.append("candidate_fallback_requires_separate_solver_or_router_design")
    return {
        "fallback_required": fallback_required,
        "fallback_allowed_now": False,
        "candidate_fallback_families": families if fallback_required else [],
        "unsafe_fallback_families": unsafe,
        "notes": notes,
    }


def _score_interpretation(final_report: dict[str, Any], policy_decision: dict[str, Any]) -> dict[str, Any]:
    overall = final_report.get("overall", {})
    raw = _optional_float(overall.get("raw_exact_match"))
    answerable = _optional_float(overall.get("answerable_accuracy"))
    abstain_rate = _optional_float(overall.get("correct_abstain_rate"))
    return {
        "raw_exact_match": raw,
        "answerable_accuracy": answerable,
        "correct_abstain_rate": abstain_rate,
        "if_abstain_allowed": "local solver answerable accuracy and correct-abstain rate are strong, but this is still not leaderboard evidence",
        "if_abstain_not_allowed": "raw exact-match is the conservative scoring view; fallback is required before package or submit",
        "resolved_policy_decision": policy_decision["decision"],
    }


def _next_action(policy_decision: dict[str, Any]) -> str:
    decision = policy_decision["decision"]
    if decision == "ABSTAIN_ALLOWED":
        return "continue_day2_evidence_review_without_packaging_or_submission"
    if decision == "ABSTAIN_NOT_ALLOWED_NEEDS_FALLBACK":
        return "design_verified_abstain_fallback_strategy_before_submission"
    return "resolve_final_competition_abstain_policy_before_submission"


def _expected(row: dict[str, Any]) -> str | None:
    for key in ("expected", "gold_answer", "answer"):
        if row.get(key) is not None:
            return str(row[key]).strip()
    return None


def _is_abstain(value: str | None) -> bool:
    return value is not None and value.strip().upper() == "ABSTAIN"


def _float(value: Any) -> float:
    return float(0.0 if value is None else value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Day2 ABSTAIN final-policy resolution report.")
    parser.add_argument("--final-solver-report", default=DEFAULT_FINAL_SOLVER_REPORT)
    parser.add_argument("--abstain-policy-gate", default=DEFAULT_ABSTAIN_POLICY_GATE)
    parser.add_argument("--core-predictions", default=DEFAULT_PREDICTIONS["core_eval"])
    parser.add_argument("--family-predictions", default=DEFAULT_PREDICTIONS["family_eval"])
    parser.add_argument("--rule-holdout-predictions", default=DEFAULT_PREDICTIONS["rule_holdout"])
    parser.add_argument("--anti-leak-predictions", default=DEFAULT_PREDICTIONS["anti_leak"])
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out", default="artifacts/sprint11/day2_abstain_resolution_report.json")
    args = parser.parse_args(argv)
    report = build_abstain_resolution_report(
        final_solver_report_path=args.final_solver_report,
        abstain_policy_gate_path=args.abstain_policy_gate,
        prediction_files={
            "core_eval": args.core_predictions,
            "family_eval": args.family_predictions,
            "rule_holdout": args.rule_holdout_predictions,
            "anti_leak": args.anti_leak_predictions,
        },
        repo_root=args.repo_root,
    )
    write_json_checked(args.out, report, field_name="day2_abstain_resolution_report")
    print(
        json.dumps(
            {
                "status": report["status"],
                "out": args.out,
                "policy_decision": report["policy_decision"],
                "fallback_required": report["fallback_need"]["fallback_required"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
