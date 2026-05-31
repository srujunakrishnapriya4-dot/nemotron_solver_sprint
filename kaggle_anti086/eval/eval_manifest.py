from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.schema import validate_rows
from kaggle_anti086.eval.day5_eval_factory import read_jsonl
from kaggle_anti086.kaggle_path_safety import safe_write_text


REQUIRED_FAMILIES = {
    "private_like": {"bit_manipulation", "symbol_mapping", "char_cipher", "word_cipher", "unit_conversion", "numeric_formula", "gravity_numeric", "roman_numeral", "custom_numeral", "equation_operator"},
    "rule_holdout": {"bit_manipulation", "symbol_mapping", "char_cipher", "word_cipher", "unit_conversion", "numeric_formula", "gravity_numeric", "roman_numeral", "custom_numeral", "equation_operator"},
    "family_hard": {"bit_manipulation", "symbol_mapping", "char_cipher", "unit_conversion", "numeric_formula", "custom_numeral", "equation_operator"},
    "anti_leak": {"bit_manipulation", "symbol_mapping", "char_cipher", "word_cipher", "unit_conversion", "numeric_formula", "equation_operator"},
}

EVAL_PURPOSE = {
    "private_like": "mixed_behavior_eval",
    "private_like_answerable": "answer_accuracy_eval",
    "rule_holdout": "mixed_behavior_eval",
    "rule_holdout_answerable": "generalization_answer_eval",
    "family_hard": "mixed_behavior_eval",
    "family_hard_answerable": "hard_answer_accuracy_eval",
    "anti_leak": "leakage_safety_eval",
}


def collect_rule_ids(paths: list[Path]) -> set[str]:
    rule_ids: set[str] = set()
    for path in paths:
        if path.exists():
            rule_ids.update(str(row["rule_id"]) for row in read_jsonl(path))
    return rule_ids


def check_no_rule_overlap(groups: dict[str, list[dict]]) -> dict:
    seen_rules: dict[str, str] = {}
    seen_leakage: dict[str, str] = {}
    rule_overlaps = []
    leakage_overlaps = []
    for name, rows in groups.items():
        for row in rows:
            rule_id = row["rule_id"]
            leakage = row["leakage_group"]
            if rule_id in seen_rules:
                rule_overlaps.append({"rule_id": rule_id, "first": seen_rules[rule_id], "second": name})
            else:
                seen_rules[rule_id] = name
            if leakage in seen_leakage:
                leakage_overlaps.append({"leakage_group": leakage, "first": seen_leakage[leakage], "second": name})
            else:
                seen_leakage[leakage] = name
    return {
        "rule_id_overlap_count": len(rule_overlaps),
        "rule_id_overlap_examples": rule_overlaps[:20],
        "leakage_group_overlap_count": len(leakage_overlaps),
        "leakage_group_overlap_examples": leakage_overlaps[:20],
    }


def build_eval_manifest(paths: dict[str, Path]) -> dict:
    files = {}
    groups = {}
    failures = []
    prompt_seen: dict[str, str] = {}
    signature_seen: dict[str, str] = {}
    duplicate_prompt_count = 0
    duplicate_signature_count = 0
    for name, path in paths.items():
        rows = read_jsonl(path)
        groups[name] = rows
        data = path.read_bytes()
        validation = validate_rows(rows, context=name)
        family_distribution = dict(sorted(Counter(row["family"] for row in rows).items()))
        subfamily_distribution = dict(sorted(Counter(f"{row['family']}::{row['subfamily']}" for row in rows).items()))
        behaviors = Counter(row.get("metadata", {}).get("expected_solver_behavior", "answer") for row in rows)
        signatures = [str(row.get("metadata", {}).get("rule_signature", "")) for row in rows]
        duplicate_rules = len(rows) - len({row["rule_id"] for row in rows})
        duplicate_leakage = len(rows) - len({row["leakage_group"] for row in rows})
        duplicate_file_prompts = len(rows) - len({row["prompt"] for row in rows})
        duplicate_file_signatures = len(signatures) - len(set(signatures))
        missing = sorted(REQUIRED_FAMILIES.get(name, set()) - set(family_distribution))
        if validation["failure_count"]:
            failures.append({"file": name, "message": "schema_failures", "count": validation["failure_count"]})
        if duplicate_rules:
            failures.append({"file": name, "message": "duplicate_rule_id", "count": duplicate_rules})
        if duplicate_leakage:
            failures.append({"file": name, "message": "duplicate_leakage_group", "count": duplicate_leakage})
        if missing:
            failures.append({"file": name, "message": "missing_required_families", "families": missing})
        for row in rows:
            prompt = row["prompt"]
            if prompt in prompt_seen:
                duplicate_prompt_count += 1
            else:
                prompt_seen[prompt] = name
            signature = str(row.get("metadata", {}).get("rule_signature", ""))
            if signature:
                if signature in signature_seen:
                    duplicate_signature_count += 1
                else:
                    signature_seen[signature] = name
        files[name] = {
            "path": str(path),
            "eval_purpose": EVAL_PURPOSE.get(name, "unknown"),
            "row_count": len(rows),
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
            "family_distribution": family_distribution,
            "family_counts": family_distribution,
            "subfamily_counts": subfamily_distribution,
            "unique_rule_id_count": len({row["rule_id"] for row in rows}),
            "unique_leakage_group_count": len({row["leakage_group"] for row in rows}),
            "answerable_count": behaviors.get("answer", 0),
            "expected_abstain_count": behaviors.get("abstain", 0),
            "duplicate_prompt_count": duplicate_file_prompts,
            "duplicate_rule_signature_count": duplicate_file_signatures,
        }
    overlap = check_no_rule_overlap(groups)
    if overlap["rule_id_overlap_count"]:
        failures.append({"message": "cross_file_rule_id_overlap", "count": overlap["rule_id_overlap_count"]})
    if overlap["leakage_group_overlap_count"]:
        failures.append({"message": "cross_file_leakage_group_overlap", "count": overlap["leakage_group_overlap_count"]})
    if duplicate_prompt_count:
        failures.append({"message": "duplicate_prompt", "count": duplicate_prompt_count})
    if duplicate_signature_count:
        failures.append({"message": "cross_file_rule_signature_overlap", "count": duplicate_signature_count})
    return {
        "schema_version": 1,
        "created_by": "SPRINT-11D",
        "eval_purpose": EVAL_PURPOSE,
        "files": files,
        "cross_file_checks": overlap | {
            "duplicate_prompt_count": duplicate_prompt_count,
            "duplicate_rule_signature_count": duplicate_signature_count,
        },
        "failures": failures,
        "status": "FAIL" if failures else "PASS",
    }


def validate_eval_manifest(manifest: dict) -> dict:
    failures = list(manifest.get("failures", []))
    if manifest.get("status") != "PASS":
        failures.append({"message": "manifest_status_not_pass"})
    return {"status": "PASS" if not failures else "FAIL", "failure_count": len(failures), "failures": failures}


def write_eval_manifest(manifest: dict, path: str | Path) -> None:
    safe_write_text(path, json.dumps(manifest, indent=2, sort_keys=True) + "\n", field_name="day5_eval_manifest")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Day 5 eval manifest.")
    parser.add_argument("--private-like", required=True)
    parser.add_argument("--private-like-answerable")
    parser.add_argument("--rule-holdout", required=True)
    parser.add_argument("--rule-holdout-answerable")
    parser.add_argument("--family-hard", required=True)
    parser.add_argument("--family-hard-answerable")
    parser.add_argument("--anti-leak", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    paths = {
        "private_like": Path(args.private_like),
        "rule_holdout": Path(args.rule_holdout),
        "family_hard": Path(args.family_hard),
        "anti_leak": Path(args.anti_leak),
    }
    if args.private_like_answerable:
        paths["private_like_answerable"] = Path(args.private_like_answerable)
    if args.rule_holdout_answerable:
        paths["rule_holdout_answerable"] = Path(args.rule_holdout_answerable)
    if args.family_hard_answerable:
        paths["family_hard_answerable"] = Path(args.family_hard_answerable)
    manifest = build_eval_manifest(paths)
    write_eval_manifest(manifest, args.out)
    print(json.dumps({"status": manifest["status"], "out": args.out, "file_count": len(manifest["files"])}, sort_keys=True))
    return 0 if manifest["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
