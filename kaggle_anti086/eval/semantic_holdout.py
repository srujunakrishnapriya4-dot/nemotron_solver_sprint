from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.eval.day5_eval_factory import read_jsonl
from kaggle_anti086.kaggle_path_safety import safe_write_text


def build_semantic_holdout_report(rule_holdout: Path, comparison_paths: dict[str, Path]) -> dict:
    holdout_rows = read_jsonl(rule_holdout)
    other_rows = {name: read_jsonl(path) for name, path in comparison_paths.items() if path.exists()}
    failures = []
    holdout_signatures = {str(row.get("metadata", {}).get("rule_signature", "")) for row in holdout_rows}
    holdout_rules = {str(row.get("rule_id", "")) for row in holdout_rows}
    holdout_leakage = {str(row.get("leakage_group", "")) for row in holdout_rows}
    if any(not row.get("rule_id", "").startswith("holdout_") for row in holdout_rows):
        failures.append("holdout_rule_id_missing_prefix")
    if any(row.get("metadata", {}).get("holdout_only") is not True for row in holdout_rows):
        failures.append("holdout_only_metadata_missing")
    if any(row.get("metadata", {}).get("train_allowed") is not False for row in holdout_rows):
        failures.append("train_allowed_not_false")

    overlaps = []
    for name, rows in other_rows.items():
        other_signatures = {str(row.get("metadata", {}).get("rule_signature", "")) for row in rows}
        other_rules = {str(row.get("rule_id", "")) for row in rows}
        other_leakage = {str(row.get("leakage_group", "")) for row in rows}
        for signature in sorted(holdout_signatures & other_signatures):
            overlaps.append({"kind": "rule_signature", "value": signature, "other_eval": name})
        for rule_id in sorted(holdout_rules & other_rules):
            overlaps.append({"kind": "rule_id", "value": rule_id, "other_eval": name})
        for leakage in sorted(holdout_leakage & other_leakage):
            overlaps.append({"kind": "leakage_group", "value": leakage, "other_eval": name})
    if overlaps:
        failures.append("semantic_or_id_overlap")
    status = "PASS" if not failures else "FAIL"
    return {
        "schema_version": 1,
        "created_by": "SPRINT-11D.2",
        "status": status,
        "semantic_holdout_pass": status == "PASS",
        "rule_holdout_row_count": len(holdout_rows),
        "comparison_file_count": len(other_rows),
        "rule_signature_overlap_count": sum(1 for item in overlaps if item["kind"] == "rule_signature"),
        "rule_id_overlap_count": sum(1 for item in overlaps if item["kind"] == "rule_id"),
        "leakage_group_overlap_count": sum(1 for item in overlaps if item["kind"] == "leakage_group"),
        "overlap_examples": overlaps[:20],
        "failures": failures,
    }


def write_semantic_holdout_report(report: dict, path: str | Path) -> None:
    safe_write_text(path, json.dumps(report, indent=2, sort_keys=True) + "\n", field_name="semantic_holdout_report")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Day 5.2 semantic holdout report.")
    parser.add_argument("--rule-holdout", required=True)
    parser.add_argument("--private-like", required=True)
    parser.add_argument("--family-hard", required=True)
    parser.add_argument("--anti-leak", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = build_semantic_holdout_report(
        Path(args.rule_holdout),
        {
            "private_like": Path(args.private_like),
            "family_hard": Path(args.family_hard),
            "anti_leak": Path(args.anti_leak),
        },
    )
    write_semantic_holdout_report(report, args.out)
    print(json.dumps({"status": report["status"], "out": args.out}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
