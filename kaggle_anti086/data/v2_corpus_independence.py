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

from kaggle_anti086.data.v2_corpus_io import read_jsonl, write_json_checked
from kaggle_anti086.data.v2_corpus_leakage import prompt_hash


def build_independence_report(direct_rows: list[dict[str, Any]], solver_corrected_rows: list[dict[str, Any]]) -> dict[str, Any]:
    direct_keys = {_row_key(row) for row in direct_rows}
    direct_source_ids = {str(row.get("source_row_id", "")) for row in direct_rows}
    direct_prompt_answer = [(_prompt(row), str(row.get("normalized_answer", row.get("answer", "")))) for row in direct_rows]
    duplicate_direct_pairs = [
        key for key, count in Counter(direct_prompt_answer).items()
        if count > 1 and not any(row.get("metadata", {}).get("duplicate_for_format_variation") for row in direct_rows if (_prompt(row), str(row.get("normalized_answer", row.get("answer", "")))) == key)
    ]
    derived = 0
    independent = 0
    sampling_weight_violations = 0
    duplicate_source = 0
    policy_overlay = {}
    for row in solver_corrected_rows:
        mirror = _row_key(row) in direct_keys or str(row.get("source_row_id", "")) in direct_source_ids
        if mirror:
            derived += 1
            policy = {"derived_from_direct": True, "sampling_weight": 0.5, "counted_as_independent": False}
        elif row.get("metadata", {}).get("true_solver_correction") is True:
            independent += 1
            policy = {"derived_from_direct": False, "sampling_weight": 1.0, "counted_as_independent": True}
        else:
            derived += 1
            policy = {"derived_from_direct": True, "sampling_weight": 0.5, "counted_as_independent": False}
        if policy["derived_from_direct"] and policy["sampling_weight"] > 0.5:
            sampling_weight_violations += 1
        if str(row.get("source_row_id", "")) in direct_source_ids:
            duplicate_source += 1
        policy_overlay[str(row.get("id", ""))] = policy
    failures = []
    if sampling_weight_violations:
        failures.append({"code": "sampling_weight_violations", "count": sampling_weight_violations})
    if duplicate_direct_pairs:
        failures.append({"code": "duplicate_direct_prompt_answer_pairs", "count": len(duplicate_direct_pairs)})
    effective = len(direct_rows) + independent + derived * 0.5
    return {
        "status": "PASS" if not failures else "FAIL",
        "direct_rows": len(direct_rows),
        "solver_corrected_rows": len(solver_corrected_rows),
        "solver_corrected_derived_from_direct": derived,
        "independent_solver_corrected_rows": independent,
        "duplicate_prompt_answer_pairs": len(duplicate_direct_pairs),
        "duplicate_source_rows": duplicate_source,
        "effective_sft_example_count": effective,
        "sampling_weight_violations": sampling_weight_violations,
        "policy_overlay": policy_overlay,
        "failures": failures,
        "warnings": ["solver_corrected_mirrors_direct_rows"] if derived else [],
    }


def _prompt(row: dict[str, Any]) -> str:
    messages = row.get("messages", [])
    if messages and isinstance(messages, list):
        return prompt_hash(str(messages[0].get("content", "")))
    return prompt_hash(str(row.get("prompt", "")))


def _row_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
    return (
        _prompt(row),
        str(row.get("normalized_answer", row.get("answer", ""))),
        str(row.get("source_row_id", "")),
        str(metadata.get("source_rule_id", "")),
        str(metadata.get("source_prompt_hash", "")),
        str(row.get("family", "")) + "::" + str(row.get("subfamily", "")) + "::" + str(row.get("rule_id", "")),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct", default="artifacts/sprint11/train_v2_verified_direct_answer.jsonl")
    parser.add_argument("--solver-corrected", default="artifacts/sprint11/train_v2_solver_corrected.jsonl")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = build_independence_report(read_jsonl(args.direct), read_jsonl(args.solver_corrected))
    write_json_checked(args.out, report, field_name="train_v2_independence_report")
    print(json.dumps({"status": report["status"], "out": args.out}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
