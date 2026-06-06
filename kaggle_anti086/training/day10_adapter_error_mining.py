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

from kaggle_anti086.data.v2_corpus_io import read_jsonl, write_json_checked, write_jsonl_checked
from kaggle_anti086.solvers.answer_normalizer import answers_match


def mine_errors(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    counts = Counter()
    by_family: dict[str, Counter[str]] = defaultdict(Counter)
    mined_rows = []
    for row in rows:
        family = str(row.get("family", "unknown"))
        pred = str(row.get("day10_pred", row.get("prediction", "")))
        expected = str(row.get("expected", row.get("answer", "")))
        expected_behavior = str(row.get("expected_behavior", "answer"))
        ok = bool(row.get("correct", False))
        if not ok and expected:
            ok = answers_match(pred, expected)
        if ok:
            continue
        failure_class = classify_failure(family=family, pred=pred, expected=expected, prompt=str(row.get("prompt", "")), expected_behavior=expected_behavior)
        counts[failure_class] += 1
        by_family[family][failure_class] += 1
        mined = dict(row)
        mined["failure_class"] = failure_class
        mined_rows.append(mined)
    report = {
        "status": "PASS",
        "row_count": len(rows),
        "failure_count": len(mined_rows),
        "failure_class_counts": dict(sorted(counts.items())),
        "by_family_failure_counts": {family: dict(sorted(counter.items())) for family, counter in sorted(by_family.items())},
        "use_for_pass11_repair": True,
        "packaging_allowed": False,
        "submission_allowed": False,
        "no_leaderboard_evidence": True,
        "no_0_95_evidence": True,
        "failures": [],
    }
    return report, mined_rows


def classify_failure(*, family: str, pred: str, expected: str, prompt: str = "", expected_behavior: str = "answer") -> str:
    stripped = pred.strip()
    if not stripped:
        return "empty_output"
    lowered = stripped.lower()
    if expected_behavior != "abstain" and stripped.upper() == "ABSTAIN":
        return "abstain_overuse"
    if any(marker in lowered for marker in ("because", "therefore", "the answer is", "explanation")):
        return "verbose_output"
    if prompt and prompt[:80] in stripped:
        return "prompt_copy"
    if "\n" in stripped or lowered.startswith("answer:"):
        return "answer_format"
    if family in {"numeric_formula", "gravity_numeric", "unit_conversion"} and _numeric_close_wrong(stripped, expected):
        return "wrong_numeric_precision"
    if family == "bit_manipulation" and re.fullmatch(r"[01]+", stripped) and re.fullmatch(r"[01]+", expected) and len(stripped) != len(expected):
        return "wrong_binary_width"
    if family in {"char_cipher", "word_cipher"} and ("?" in stripped or "_" in stripped):
        return "cipher_partial"
    if family == "symbol_mapping" and any(ch in stripped for ch in ("?", "UNK", "unknown")):
        return "symbol_unknown"
    return "wrong_transformation_type"


def _numeric_close_wrong(pred: str, expected: str) -> bool:
    try:
        return abs(float(pred) - float(expected)) < 0.1 and str(pred).strip() != str(expected).strip()
    except ValueError:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mine Day 10 adapter eval failures for PASS 11 repair.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out-report", default="artifacts/sprint11/day10_adapter_error_mining_report.json")
    parser.add_argument("--out-rows", default="artifacts/sprint11/day10_adapter_error_mining_rows.jsonl")
    args = parser.parse_args(argv)
    report, rows = mine_errors(read_jsonl(args.predictions))
    write_json_checked(args.out_report, report, field_name="day10_adapter_error_mining_report")
    write_jsonl_checked(args.out_rows, rows, field_name="day10_adapter_error_mining_rows")
    print(json.dumps({"status": report["status"], "failure_count": report["failure_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
