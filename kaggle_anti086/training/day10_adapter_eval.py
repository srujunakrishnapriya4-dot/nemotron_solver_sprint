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

from kaggle_anti086.data.v2_corpus_io import read_jsonl, write_json_checked, write_jsonl_checked
from kaggle_anti086.solvers.answer_normalizer import answers_match, normalize_answer


def evaluate_prediction_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, Counter[str]] = defaultdict(Counter)
    correct = empty = verbose = invalid = extraction = prompt_copy = abstain_overuse = unsafe = 0
    for row in rows:
        family = str(row.get("family", "unknown"))
        pred = str(row.get("day10_pred", row.get("prediction", "")))
        expected = str(row.get("expected", row.get("answer", "")))
        expected_behavior = str(row.get("expected_behavior", row.get("metadata", {}).get("expected_solver_behavior", "answer")))
        ok = answers_match(pred, expected, answer_type=_answer_type(family)) if expected_behavior != "abstain" else pred.strip().upper() == "ABSTAIN"
        correct += int(ok)
        by_family[family]["count"] += 1
        by_family[family]["correct"] += int(ok)
        empty += int(not pred.strip())
        verbose += int(_is_verbose(pred))
        invalid += int(_invalid_format(pred))
        extraction += int(bool(row.get("extraction_failure", False)))
        prompt = str(row.get("prompt", ""))
        prompt_copy += int(bool(prompt) and prompt[:80] in pred)
        abstain_overuse += int(expected_behavior != "abstain" and pred.strip().upper() == "ABSTAIN")
        unsafe += int(expected_behavior == "abstain" and pred.strip().upper() not in {"", "ABSTAIN"})
    total = len(rows)
    return {
        "status": "PASS",
        "model_eval_completed": True,
        "row_count": total,
        "exact_match": correct / total if total else 0.0,
        "by_family": {
            fam: {"count": c["count"], "correct": c["correct"], "exact_match": c["correct"] / c["count"] if c["count"] else 0.0}
            for fam, c in sorted(by_family.items())
        },
        "empty_output_rate": empty / total if total else 0.0,
        "verbose_output_rate": verbose / total if total else 0.0,
        "invalid_format_rate": invalid / total if total else 0.0,
        "extraction_failure_rate": extraction / total if total else 0.0,
        "prompt_copy_rate": prompt_copy / total if total else 0.0,
        "abstain_overuse_rate": abstain_overuse / total if total else 0.0,
        "unsafe_answer_rate": unsafe / total if total else 0.0,
        "packaging_allowed": False,
        "submission_allowed": False,
        "no_leaderboard_evidence": True,
        "no_0_95_evidence": True,
        "failures": [],
        "warnings": [],
    }


def build_needs_kaggle_eval_report(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "status": "NEEDS_KAGGLE_MODEL_EVAL",
        "model_eval_completed": False,
        "base_model_path": args.base_model_path,
        "day10_adapter_dir": args.day10_adapter_dir,
        "v2a_adapter_dir": args.v2a_adapter_dir,
        "tinker_adapter_dir": args.tinker_adapter_dir,
        "tinker_usage": "baseline_only_if_locally_available",
        "sequential_model_loading_required": True,
        "packaging_allowed": False,
        "submission_allowed": False,
        "no_leaderboard_evidence": True,
        "no_0_95_evidence": True,
        "warnings": ["real_model_eval_requires_kaggle_mode"],
        "failures": [],
    }


def _answer_type(family: str) -> str | None:
    if family in {"numeric_formula", "gravity_numeric", "unit_conversion"}:
        return "numeric"
    if family == "roman_numeral":
        return "roman"
    if family == "bit_manipulation":
        return "binary"
    return None


def _is_verbose(text: str) -> bool:
    lowered = str(text).lower()
    return any(item in lowered for item in ("because", "therefore", "first,", "we need", "the answer is", "explanation"))


def _invalid_format(text: str) -> bool:
    stripped = str(text).strip()
    return not stripped or "\n" in stripped or stripped.lower().startswith("answer:")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Day 10 adapter-only eval entrypoint.")
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--day10-adapter-dir", required=True)
    parser.add_argument("--v2a-adapter-dir", default=None)
    parser.add_argument("--tinker-adapter-dir", default=None)
    parser.add_argument("--eval-path", required=True)
    parser.add_argument("--out-report", required=True)
    parser.add_argument("--out-predictions", required=True)
    parser.add_argument("--max-rows", type=int, default=128)
    parser.add_argument("--kaggle-mode", action="store_true")
    parser.add_argument("--predictions-input", default=None, help="Test-only precomputed predictions JSONL.")
    args = parser.parse_args(argv)
    if args.predictions_input:
        rows = read_jsonl(args.predictions_input)[: args.max_rows]
        report = evaluate_prediction_rows(rows)
        write_jsonl_checked(args.out_predictions, rows, field_name="day10_adapter_eval_predictions")
    elif not args.kaggle_mode:
        report = build_needs_kaggle_eval_report(args)
        write_jsonl_checked(args.out_predictions, [], field_name="day10_adapter_eval_predictions")
    else:
        report = {
            "status": "NEEDS_IMPLEMENTED_KAGGLE_MODEL_LOOP",
            "model_eval_completed": False,
            "sequential_model_loading_required": True,
            "failures": ["real_generation_backend_not_run_in_local_implementation_pass"],
            "packaging_allowed": False,
            "submission_allowed": False,
            "no_leaderboard_evidence": True,
            "no_0_95_evidence": True,
        }
        write_jsonl_checked(args.out_predictions, [], field_name="day10_adapter_eval_predictions")
    write_json_checked(args.out_report, report, field_name="day10_adapter_eval_report")
    print(json.dumps({"status": report["status"], "out": args.out_report}, sort_keys=True))
    return 0 if report["status"] in {"PASS", "NEEDS_KAGGLE_MODEL_EVAL"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
