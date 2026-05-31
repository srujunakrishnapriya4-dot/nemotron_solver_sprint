from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.v2_corpus_io import read_jsonl, write_json_checked


UNSUPPORTED = {"equation_operator", "sequence_pattern", "permutation_sorting", "custom_numeral", "unknown"}


def build_hard_negative_taxonomy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    classes = Counter()
    sft_violations = 0
    missing_do_not_use = 0
    missing_policy = 0
    for row in rows:
        classes[classify_hard_negative(row)] += 1
        if row.get("training_allowed") is True or row.get("allowed_for_sft") is True:
            sft_violations += 1
        if row.get("do_not_use_as_sft") is not True:
            missing_do_not_use += 1
        if not (row.get("allowed_for_dpo") or row.get("allowed_for_contrastive") or row.get("allowed_for_repair_eval")):
            missing_policy += 1
    warnings = []
    failures = []
    needs_expansion = len(rows) < 100
    insufficient = False
    if needs_expansion:
        warnings.append("hard_negative_rows_below_100_needs_day9_expansion")
    if len(rows) >= 50 and len(classes) < 5:
        insufficient = True
        warnings.append("insufficient_failure_diversity")
    if len(rows) < 50 and len(classes) < 3:
        insufficient = True
        warnings.append("insufficient_failure_diversity")
    if sft_violations:
        failures.append({"code": "sft_violation_count", "count": sft_violations})
    if missing_do_not_use:
        failures.append({"code": "do_not_use_as_sft_missing_count", "count": missing_do_not_use})
    if missing_policy:
        failures.append({"code": "contrastive_or_repair_policy_missing_count", "count": missing_policy})
    return {
        "status": "FAIL" if failures else ("WARN" if warnings else "PASS"),
        "hard_negative_rows": len(rows),
        "needs_day9_expansion": needs_expansion,
        "failure_class_counts": dict(sorted(classes.items())),
        "sft_violation_count": sft_violations,
        "do_not_use_as_sft_missing_count": missing_do_not_use,
        "contrastive_or_repair_policy_missing_count": missing_policy,
        "insufficient_failure_diversity": insufficient,
        "warnings": warnings,
        "failures": failures,
    }


def classify_hard_negative(row: dict[str, Any]) -> str:
    family = str(row.get("family", ""))
    failure = str(row.get("failure_type", ""))
    wrong = str(row.get("wrong_prediction", ""))
    gold = str(row.get("gold_answer", ""))
    if family in UNSUPPORTED:
        return "unsupported_family"
    if failure == "solver_abstained_on_answerable" or not wrong:
        return "solver_abstained_on_answerable"
    if family in {"numeric_formula", "gravity_numeric", "unit_conversion"} and _numeric_close(wrong, gold):
        return "wrong_precision"
    if family == "bit_manipulation" and re.fullmatch(r"[01]+", wrong) and re.fullmatch(r"[01]+", gold) and len(wrong) != len(gold):
        return "wrong_binary_width"
    if family in {"word_cipher", "char_cipher"} and any(token in wrong.lower() for token in ("?", "unknown", "_")):
        return "cipher_partial_mapping"
    if family == "symbol_mapping" and any(token in failure.lower() for token in ("unknown", "coverage")):
        return "symbol_unknown_mapping"
    if family == "bit_manipulation" and "ambiguous" in failure.lower():
        return "bit_ambiguous_transform"
    if any(token in wrong.lower() for token in ("because", "answer:", "therefore", "explanation")):
        return "format_wrong"
    if "router" in failure.lower():
        return "router_error"
    if "verifier" in failure.lower():
        return "verifier_reject"
    if failure == "solver_wrong_answer":
        return "solver_wrong_answer"
    return "unknown_failure"


def _numeric_close(wrong: str, gold: str) -> bool:
    try:
        return abs(float(wrong) - float(gold)) < 0.1
    except ValueError:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hard-negative", default="artifacts/sprint11/train_v2_hard_negative.jsonl")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = build_hard_negative_taxonomy(read_jsonl(args.hard_negative))
    write_json_checked(args.out, report, field_name="train_v2_hard_negative_taxonomy")
    print(json.dumps({"status": report["status"], "out": args.out}, sort_keys=True))
    return 0 if report["status"] != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
