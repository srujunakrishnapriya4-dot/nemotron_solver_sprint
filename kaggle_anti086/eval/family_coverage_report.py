from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.eval.day5_eval_factory import read_jsonl
from kaggle_anti086.kaggle_path_safety import safe_write_text


def build_family_coverage_report(rows_by_eval: dict[str, list[dict]]) -> dict:
    evals = {}
    warnings = []
    duplicate_prompts = _duplicate_prompt_count(rows_by_eval)
    for name, rows in rows_by_eval.items():
        by_family = Counter(row["family"] for row in rows)
        by_subfamily = Counter(f"{row['family']}::{row['subfamily']}" for row in rows)
        by_difficulty = Counter(str(row["difficulty"]) for row in rows)
        by_noise = Counter(str(row.get("metadata", {}).get("noise_profile", "missing")) for row in rows)
        behavior = Counter(str(row.get("metadata", {}).get("expected_solver_behavior", "answer")) for row in rows)
        unsupported_count = sum(1 for row in rows if row["family"] in {"equation_operator", "sequence_pattern", "permutation_sorting", "custom_numeral", "format_only"})
        evals[name] = {
            "row_count": len(rows),
            "by_family": dict(sorted(by_family.items())),
            "by_subfamily": dict(sorted(by_subfamily.items())),
            "difficulty_distribution": dict(sorted(by_difficulty.items())),
            "noise_profile_distribution": dict(sorted(by_noise.items())),
            "answerable_count": behavior.get("answer", 0),
            "expected_abstain_count": behavior.get("abstain", 0),
            "unsupported_family_count": unsupported_count,
        }
        if name == "private_like":
            for family in ("bit_manipulation", "symbol_mapping", "char_cipher", "word_cipher", "unit_conversion", "numeric_formula", "gravity_numeric", "roman_numeral"):
                if by_family.get(family, 0) < 16:
                    warnings.append(f"private_like_low_{family}_coverage")
            if behavior.get("abstain", 0) / max(1, len(rows)) > 0.35:
                warnings.append("private_like_expected_abstain_ratio_too_high")
        if name == "family_hard":
            thresholds = {"symbol_mapping": 48, "char_cipher": 48, "bit_manipulation": 48, "custom_numeral": 16}
            for family, threshold in thresholds.items():
                if by_family.get(family, 0) < threshold:
                    warnings.append(f"family_hard_low_{family}_coverage")
        for row in rows:
            if row["family"] == "equation_operator" and row.get("metadata", {}).get("expected_solver_behavior") != "abstain" and row.get("verification_status") != "verified":
                warnings.append("equation_operator_not_marked_abstain")
                break
    if duplicate_prompts:
        warnings.append("duplicate_prompt_count_gt_zero")
    return {
        "schema_version": 1,
        "created_by": "SPRINT-11D",
        "evals": evals,
        "duplicate_prompt_count": duplicate_prompts,
        "warnings": sorted(set(warnings)),
        "status": "PASS" if not warnings else "WARN",
    }


def write_family_coverage_report(report: dict, path: str | Path) -> None:
    safe_write_text(path, json.dumps(report, indent=2, sort_keys=True) + "\n", field_name="day5_family_coverage_report")


def _duplicate_prompt_count(rows_by_eval: dict[str, list[dict]]) -> int:
    seen = set()
    duplicate = 0
    for rows in rows_by_eval.values():
        for row in rows:
            prompt = row["prompt"]
            if prompt in seen:
                duplicate += 1
            else:
                seen.add(prompt)
    return duplicate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Day 5 family coverage report.")
    parser.add_argument("--private-like", required=True)
    parser.add_argument("--rule-holdout", required=True)
    parser.add_argument("--family-hard", required=True)
    parser.add_argument("--anti-leak", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = build_family_coverage_report(
        {
            "private_like": read_jsonl(args.private_like),
            "rule_holdout": read_jsonl(args.rule_holdout),
            "family_hard": read_jsonl(args.family_hard),
            "anti_leak": read_jsonl(args.anti_leak),
        }
    )
    write_family_coverage_report(report, args.out)
    print(json.dumps({"status": report["status"], "warning_count": len(report["warnings"]), "out": args.out}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
