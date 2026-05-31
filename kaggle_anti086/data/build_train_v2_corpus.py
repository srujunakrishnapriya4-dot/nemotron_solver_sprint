from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.schema import RowValidationError, validate_row
from kaggle_anti086.data.v2_corpus_io import read_json, read_jsonl, resolve_existing_path, write_json_checked, write_jsonl_checked
from kaggle_anti086.data.v2_corpus_leakage import UNSUPPORTED_DIRECT_FAMILIES, build_leakage_report
from kaggle_anti086.data.v2_corpus_manifest import build_corpus_manifest
from kaggle_anti086.data.v2_corpus_mixture import build_mixture_report
from kaggle_anti086.data.v2_corpus_quality_gate import build_quality_gate
from kaggle_anti086.data.v2_prompt_templates import build_direct_answer_messages, normalized_answer_for_family, validate_training_messages
from kaggle_anti086.solvers.answer_normalizer import answers_match


SUPPORTED_DIRECT_FAMILIES = {
    "bit_manipulation",
    "numeric_formula",
    "gravity_numeric",
    "unit_conversion",
    "roman_numeral",
    "word_cipher",
    "char_cipher",
    "symbol_mapping",
    "format_only",
}


def build_train_v2_corpus(
    *,
    eligibility_report_path: str | Path,
    readiness_summary_path: str | Path,
    eval_paths: dict[str, str | Path],
    prediction_paths: dict[str, str | Path],
    out_dir: str | Path,
) -> dict[str, Any]:
    readiness = read_json(readiness_summary_path)
    if readiness.get("decision") != "ALLOW_DAY6_CORPUS_BUILD":
        raise SystemExit(f"Day 6 blocked: readiness decision is {readiness.get('decision')}")
    eligibility = read_json(eligibility_report_path)
    if eligibility.get("status") != "PASS" or eligibility.get("day6_decision") != "ALLOW_CORPUS_BUILD":
        raise SystemExit("Day 6 blocked: v2 eligibility report does not allow corpus build")

    resolved_evals = {name: _resolve_eval_path(path, name) for name, path in eval_paths.items()}
    resolved_predictions = {name: _resolve_prediction_path(path, name) for name, path in prediction_paths.items()}
    eval_rows = {name: read_jsonl(path) for name, path in resolved_evals.items()}
    predictions = {name: {str(row.get("id")): row for row in read_jsonl(path)} for name, path in resolved_predictions.items()}
    rows_by_id = {name: {str(row.get("id")): row for row in rows} for name, rows in eval_rows.items()}

    direct_rows: list[dict[str, Any]] = []
    solver_corrected_rows: list[dict[str, Any]] = []
    abstain_rows: list[dict[str, Any]] = []
    hard_negative_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    blocked_counts: Counter[str] = Counter()

    for item in eligibility.get("row_classifications", []):
        eval_name = str(item.get("eval", ""))
        source_id = str(item.get("id", ""))
        row = rows_by_id.get(eval_name, {}).get(source_id)
        pred = predictions.get(eval_name, {}).get(source_id, {})
        classification = str(item.get("classification", "blocked_missing_metadata"))
        if row is None:
            blocked_counts["missing_source_row"] += 1
            rejected_rows.append({"source_eval": eval_name, "source_row_id": source_id, "reason": "missing_source_row", "classification": classification})
            continue
        if classification == "eligible_verified_answer":
            built = _build_direct_row(len(direct_rows) + 1, eval_name, row, pred)
            if built is None:
                blocked_counts["direct_row_guard_rejected"] += 1
                rejected_rows.append(_rejected(eval_name, row, pred, classification, "direct_row_guard_rejected"))
                continue
            direct_rows.append(built)
            solver_corrected_rows.append(_as_solver_corrected(len(solver_corrected_rows) + 1, built))
        elif classification == "eligible_abstain_safety":
            abstain_rows.append(_build_abstain_row(len(abstain_rows) + 1, eval_name, row, pred, classification))
        elif classification == "eligible_hard_negative" or (classification == "blocked_solver_failure" and _expected_answer(row)):
            hard_negative_rows.append(_build_hard_negative_row(len(hard_negative_rows) + 1, eval_name, row, pred, classification))
        else:
            blocked_counts[classification] += 1
            rejected_rows.append(_rejected(eval_name, row, pred, classification, classification))

    leakage_report = build_leakage_report(direct_rows, eval_rows)
    mixture_report = build_mixture_report(direct_rows)
    quality_gate = build_quality_gate(direct_rows, abstain_rows, hard_negative_rows, leakage_report, mixture_report)
    output_dir = Path(out_dir)
    paths = {
        "verified_direct_answer": output_dir / "train_v2_verified_direct_answer.jsonl",
        "solver_corrected": output_dir / "train_v2_solver_corrected.jsonl",
        "abstain_safety": output_dir / "train_v2_abstain_safety.jsonl",
        "hard_negative": output_dir / "train_v2_hard_negative.jsonl",
        "rejected_rows": output_dir / "train_v2_rejected_rows.jsonl",
    }
    write_jsonl_checked(paths["verified_direct_answer"], direct_rows, field_name="train_v2_verified_direct_answer")
    write_jsonl_checked(paths["solver_corrected"], solver_corrected_rows, field_name="train_v2_solver_corrected")
    write_jsonl_checked(paths["abstain_safety"], abstain_rows, field_name="train_v2_abstain_safety")
    write_jsonl_checked(paths["hard_negative"], hard_negative_rows, field_name="train_v2_hard_negative")
    write_jsonl_checked(paths["rejected_rows"], rejected_rows, field_name="train_v2_rejected_rows")

    rule_coverage = _build_rule_coverage_report(direct_rows)
    blocked_reason_report = {"blocked_counts": dict(sorted(blocked_counts.items())), "rejected_row_count": len(rejected_rows)}
    write_json_checked(output_dir / "train_v2_rule_coverage_report.json", rule_coverage, field_name="train_v2_rule_coverage_report")
    write_json_checked(output_dir / "train_v2_leakage_report.json", leakage_report, field_name="train_v2_leakage_report")
    write_json_checked(output_dir / "train_v2_mixture_report.json", mixture_report, field_name="train_v2_mixture_report")
    write_json_checked(output_dir / "train_v2_quality_gate.json", quality_gate, field_name="train_v2_quality_gate")
    write_json_checked(output_dir / "train_v2_blocked_reason_report.json", blocked_reason_report, field_name="train_v2_blocked_reason_report")

    manifest_paths = {
        **paths,
        "rule_coverage_report": output_dir / "train_v2_rule_coverage_report.json",
        "leakage_report": output_dir / "train_v2_leakage_report.json",
        "mixture_report": output_dir / "train_v2_mixture_report.json",
        "quality_gate": output_dir / "train_v2_quality_gate.json",
        "blocked_reason_report": output_dir / "train_v2_blocked_reason_report.json",
    }
    manifest = build_corpus_manifest(
        files=manifest_paths,
        direct_rows=direct_rows,
        solver_corrected_rows=solver_corrected_rows,
        abstain_rows=abstain_rows,
        hard_negative_rows=hard_negative_rows,
        leakage_report=leakage_report,
        mixture_report=mixture_report,
        quality_gate=quality_gate,
        blocked_rows=dict(blocked_counts),
    )
    write_json_checked(output_dir / "train_v2_manifest.json", manifest, field_name="train_v2_manifest")
    summary = {
        "status": "PASS" if quality_gate["status"] == leakage_report["status"] == mixture_report["status"] == "PASS" else "FAIL",
        "decision": quality_gate["decision"],
        "training_allowed": False,
        "packaging_allowed": False,
        "submission_allowed": False,
        "verified_direct_answer_rows": len(direct_rows),
        "solver_corrected_rows": len(solver_corrected_rows),
        "abstain_safety_rows": len(abstain_rows),
        "hard_negative_rows": len(hard_negative_rows),
        "quality_gate_status": quality_gate["status"],
        "leakage_report_status": leakage_report["status"],
        "mixture_report_status": mixture_report["status"],
        "manifest_status": manifest["status"],
        "remaining_blockers": quality_gate.get("remaining_blockers", []),
        "no_training_performed": True,
        "no_adapter_packaged": True,
        "no_submission_created": True,
        "no_leaderboard_evidence": True,
        "no_0_95_evidence": True,
    }
    if summary["status"] != "PASS":
        summary["decision"] = "BLOCK_DAY7_TRAINING_CONFIG_PREP"
    write_json_checked(output_dir / "day6_corpus_build_summary.json", summary, field_name="day6_corpus_build_summary")
    return summary


def _build_direct_row(index: int, eval_name: str, row: dict[str, Any], pred: dict[str, Any]) -> dict[str, Any] | None:
    if row.get("family") not in SUPPORTED_DIRECT_FAMILIES or row.get("family") in UNSUPPORTED_DIRECT_FAMILIES:
        return None
    if row.get("verification_status") != "verified" or not _expected_answer(row):
        return None
    if pred.get("abstained") or not pred.get("correct"):
        return None
    if not answers_match(str(pred.get("prediction", "")), str(row.get("answer", "")), _answer_type_for_family(str(row.get("family", "")))):
        return None
    try:
        validate_row(row, context=str(row.get("id")))
    except RowValidationError:
        return None
    answer = str(row.get("answer", "")).strip()
    messages = build_direct_answer_messages(row, answer)
    if validate_training_messages(messages, answer=answer)["status"] != "PASS":
        return None
    source_rule_id = str(row.get("rule_id", ""))
    source_leakage_group = str(row.get("leakage_group", ""))
    return {
        "id": f"train_v2_direct_{index:06d}",
        "source_row_id": row.get("id"),
        "source_eval": eval_name,
        "family": row.get("family"),
        "subfamily": row.get("subfamily"),
        "rule_id": _mint_train_id("rule", eval_name, source_rule_id),
        "leakage_group": _mint_train_id("lg", eval_name, source_leakage_group),
        "prompt": row.get("prompt"),
        "messages": messages,
        "answer": answer,
        "normalized_answer": normalized_answer_for_family(answer, str(row.get("family", ""))),
        "target_style": "direct_answer",
        "loss_scope": "assistant_only",
        "verification_status": "verified",
        "solver_name": row.get("solver_name", "solver_ensemble"),
        "solver_source": pred.get("best_source", ""),
        "confidence": float(pred.get("confidence") or 0.0),
        "example_consistency": 1.0 if pred.get("verified") else 0.0,
        "training_allowed": True,
        "metadata": {
            "created_by": "SPRINT-11E",
            "corpus_kind": "verified_direct_answer",
            "no_full_prompt_loss": True,
            "source_prediction_correct": True,
            "source_rule_id": source_rule_id,
            "source_leakage_group": source_leakage_group,
            "source_prediction": pred.get("prediction", ""),
            "source_classification": "eligible_verified_answer",
            "original_metadata": row.get("metadata", {}),
        },
    }


def _as_solver_corrected(index: int, direct_row: dict[str, Any]) -> dict[str, Any]:
    row = dict(direct_row)
    row["id"] = f"train_v2_solver_corrected_{index:06d}"
    row["metadata"] = dict(direct_row.get("metadata", {}), corpus_kind="solver_corrected")
    return row


def _build_abstain_row(index: int, eval_name: str, row: dict[str, Any], pred: dict[str, Any], classification: str) -> dict[str, Any]:
    return {
        "id": f"train_v2_abstain_safety_{index:06d}",
        "source_row_id": row.get("id"),
        "source_eval": eval_name,
        "family": row.get("family"),
        "subfamily": row.get("subfamily"),
        "rule_id": _mint_train_id("abstain_rule", eval_name, str(row.get("rule_id", ""))),
        "leakage_group": _mint_train_id("abstain_lg", eval_name, str(row.get("leakage_group", ""))),
        "prompt": row.get("prompt"),
        "gold_answer": row.get("answer", ""),
        "target_style": "abstain_safety_metadata_only",
        "training_allowed": "false_for_direct_answer_sft",
        "usage": "safety_eval_or_negative_only",
        "metadata": {
            "created_by": "SPRINT-11E",
            "corpus_kind": "abstain_safety",
            "source_classification": classification,
            "solver_abstained": bool(pred.get("abstained", True)),
            "original_metadata": row.get("metadata", {}),
        },
    }


def _build_hard_negative_row(index: int, eval_name: str, row: dict[str, Any], pred: dict[str, Any], classification: str) -> dict[str, Any]:
    prediction = str(pred.get("prediction", ""))
    failure_type = classification if classification != "blocked_solver_failure" else ("solver_abstained_on_answerable" if pred.get("abstained", True) else "solver_wrong_answer")
    return {
        "id": f"train_v2_hard_negative_{index:06d}",
        "source_row_id": row.get("id"),
        "source_eval": eval_name,
        "family": row.get("family"),
        "subfamily": row.get("subfamily"),
        "rule_id": _mint_train_id("hardneg_rule", eval_name, str(row.get("rule_id", ""))),
        "leakage_group": _mint_train_id("hardneg_lg", eval_name, str(row.get("leakage_group", ""))),
        "prompt": row.get("prompt"),
        "wrong_prediction": prediction,
        "gold_answer": row.get("answer", ""),
        "failure_type": failure_type,
        "training_allowed": "false_for_direct_answer_sft",
        "usage": "contrastive_or_future_dpo_only",
        "do_not_use_as_sft": True,
        "metadata": {
            "created_by": "SPRINT-11E",
            "corpus_kind": "hard_negative",
            "source_classification": classification,
            "original_metadata": row.get("metadata", {}),
        },
    }


def _rejected(eval_name: str, row: dict[str, Any], pred: dict[str, Any], classification: str, reason: str) -> dict[str, Any]:
    return {
        "source_eval": eval_name,
        "source_row_id": row.get("id"),
        "family": row.get("family"),
        "subfamily": row.get("subfamily"),
        "classification": classification,
        "reason": reason,
        "prediction": pred.get("prediction", ""),
        "correct": bool(pred.get("correct", False)),
    }


def _build_rule_coverage_report(direct_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, Counter[str]] = defaultdict(Counter)
    for row in direct_rows:
        by_family[str(row.get("family", "unknown"))][str(row.get("subfamily", "unknown"))] += 1
    return {
        "status": "PASS",
        "row_count": len(direct_rows),
        "family_distribution": dict(sorted(Counter(row.get("family", "unknown") for row in direct_rows).items())),
        "subfamily_distribution": {family: dict(sorted(counter.items())) for family, counter in sorted(by_family.items())},
        "unique_rule_id_count": len({row.get("rule_id") for row in direct_rows}),
        "unique_leakage_group_count": len({row.get("leakage_group") for row in direct_rows}),
    }


def _expected_answer(row: dict[str, Any]) -> bool:
    return row.get("metadata", {}).get("expected_solver_behavior") == "answer"


def _mint_train_id(prefix: str, eval_name: str, source_value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", source_value)[:160].strip("_")
    return f"train_v2_{prefix}_{eval_name}_{slug}"


def _answer_type_for_family(family: str) -> str | None:
    if family in {"numeric_formula", "gravity_numeric", "unit_conversion"}:
        return "numeric"
    if family == "bit_manipulation":
        return "binary"
    if family == "roman_numeral":
        return "roman"
    if family in {"symbol_mapping", "format_only"}:
        return "symbol"
    if family in {"word_cipher", "char_cipher"}:
        return "text_phrase"
    return None


def _resolve_eval_path(path: str | Path, eval_name: str) -> Path:
    base = Path(path)
    candidates = [
        base,
        Path("artifacts/sprint11") / f"day5_{eval_name}_eval_512.jsonl",
        Path("artifacts/sprint11") / f"day5_{eval_name}_eval_256.jsonl",
    ]
    return resolve_existing_path(base, candidates)


def _resolve_prediction_path(path: str | Path, eval_name: str) -> Path:
    base = Path(path)
    candidates = [base, Path("artifacts/sprint11") / f"day5_{eval_name}_solver_predictions.jsonl"]
    return resolve_existing_path(base, candidates)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Sprint 11E Day 6 v2 corpus artifacts.")
    parser.add_argument("--eligibility-report", required=True)
    parser.add_argument("--readiness-summary", required=True)
    parser.add_argument("--private-like", required=True)
    parser.add_argument("--private-like-predictions", required=True)
    parser.add_argument("--rule-holdout", required=True)
    parser.add_argument("--rule-holdout-predictions", required=True)
    parser.add_argument("--family-hard", required=True)
    parser.add_argument("--family-hard-predictions", required=True)
    parser.add_argument("--anti-leak", required=True)
    parser.add_argument("--anti-leak-predictions", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    eval_paths = {
        "private_like": args.private_like,
        "rule_holdout": args.rule_holdout,
        "family_hard": args.family_hard,
        "anti_leak": args.anti_leak,
    }
    prediction_paths = {
        "private_like": args.private_like_predictions,
        "rule_holdout": args.rule_holdout_predictions,
        "family_hard": args.family_hard_predictions,
        "anti_leak": args.anti_leak_predictions,
    }
    summary = build_train_v2_corpus(
        eligibility_report_path=args.eligibility_report,
        readiness_summary_path=args.readiness_summary,
        eval_paths=eval_paths,
        prediction_paths=prediction_paths,
        out_dir=args.out_dir,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": summary["decision"],
                "verified_direct_answer_rows": summary["verified_direct_answer_rows"],
                "abstain_safety_rows": summary["abstain_safety_rows"],
                "hard_negative_rows": summary["hard_negative_rows"],
                "out_dir": args.out_dir,
            },
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
