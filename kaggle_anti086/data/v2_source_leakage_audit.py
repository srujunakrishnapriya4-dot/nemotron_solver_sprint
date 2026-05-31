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

from kaggle_anti086.data.v2_corpus_io import read_jsonl, resolve_existing_path, write_json_checked
from kaggle_anti086.data.v2_corpus_leakage import UNSUPPORTED_DIRECT_FAMILIES, prompt_hash


DEFAULT_CORPUS = {
    "direct": "artifacts/sprint11/train_v2_verified_direct_answer.jsonl",
    "solver_corrected": "artifacts/sprint11/train_v2_solver_corrected.jsonl",
    "abstain_safety": "artifacts/sprint11/train_v2_abstain_safety.jsonl",
    "hard_negative": "artifacts/sprint11/train_v2_hard_negative.jsonl",
}
DEFAULT_EVALS = {
    "rule_holdout": ("artifacts/sprint11/day5_rule_holdout_eval_512.jsonl",),
    "anti_leak": ("artifacts/sprint11/day5_anti_leak_eval_256.jsonl",),
    "private_like": ("artifacts/sprint11/day5_private_like_eval_512.jsonl",),
    "family_hard": ("artifacts/sprint11/day5_family_hard_eval_512.jsonl",),
}
REQUIRED_SOURCE_METADATA = ("source_row_id", "source_rule_id", "source_leakage_group", "source_prompt_hash", "expected_solver_behavior")


def build_source_leakage_audit(corpus_paths: dict[str, str | Path] | None = None, eval_paths: dict[str, str | Path] | None = None) -> dict[str, Any]:
    corpus_paths = corpus_paths or DEFAULT_CORPUS
    eval_paths = eval_paths or {name: paths[0] for name, paths in DEFAULT_EVALS.items()}
    corpus_rows = {name: read_jsonl(resolve_existing_path(path)) for name, path in corpus_paths.items()}
    eval_rows = {name: read_jsonl(resolve_existing_path(path)) for name, path in eval_paths.items()}
    all_eval_rows = [row for rows in eval_rows.values() for row in rows]
    eval_prompt_hashes = {prompt_hash(str(row.get("prompt", ""))) for row in all_eval_rows}
    eval_ids = {str(row.get("id", "")) for row in all_eval_rows}
    rule_holdout_rule_ids = {str(row.get("rule_id", "")) for row in eval_rows.get("rule_holdout", [])}
    anti_leak_groups = {str(row.get("leakage_group", "")) for row in eval_rows.get("anti_leak", [])}

    direct = corpus_rows.get("direct", [])
    direct_meta = [_metadata(row) for row in direct]
    source_rule_overlap = sorted({str(meta.get("source_rule_id", "")) for meta in direct_meta} & rule_holdout_rule_ids)
    source_lg_overlap = sorted({str(meta.get("source_leakage_group", "")) for meta in direct_meta} & anti_leak_groups)
    source_prompt_overlap = sorted({str(meta.get("source_prompt_hash", "")) for meta in direct_meta} & eval_prompt_hashes)
    corpus_prompt_overlap = sorted({prompt_hash(str(row.get("prompt", ""))) for row in direct} & eval_prompt_hashes)
    source_row_ids = [str(row.get("source_row_id") or _metadata(row).get("source_row_id", "")) for row in direct]
    duplicate_source_ids = [value for value, count in Counter(source_row_ids).items() if value and count > 1]
    source_ids_in_eval = sorted(set(source_row_ids) & eval_ids)
    direct_from_abstain = [row.get("id") for row in direct if _metadata(row).get("expected_solver_behavior") == "abstain"]
    unsupported_direct = [
        row.get("id")
        for row in direct
        if row.get("family") in UNSUPPORTED_DIRECT_FAMILIES
        and not (row.get("family") == "format_only" and row.get("verification_status") == "verified" and row.get("solver_name") == "format_only_solver")
    ]
    unverified_sft = [row.get("id") for row in direct if row.get("verification_status") != "verified"]
    missing_by_file: dict[str, int] = {}
    missing_examples = []
    for name, rows in corpus_rows.items():
        count = 0
        for row in rows:
            missing = [key for key in REQUIRED_SOURCE_METADATA if not _metadata(row).get(key)]
            if missing:
                count += 1
                if len(missing_examples) < 20:
                    missing_examples.append({"file": name, "id": row.get("id"), "missing": missing})
        missing_by_file[name] = count

    failures = []
    _fail_if(failures, "source_rule_id_overlap_with_rule_holdout", source_rule_overlap)
    _fail_if(failures, "source_leakage_group_overlap_with_anti_leak", source_lg_overlap)
    _fail_if(failures, "source_prompt_hash_overlap_with_eval", source_prompt_overlap)
    _fail_if(failures, "corpus_prompt_hash_overlap_with_eval", corpus_prompt_overlap)
    _fail_if(failures, "direct_source_row_id_duplicates", duplicate_source_ids)
    _fail_if(failures, "direct_source_row_id_in_eval_ids", source_ids_in_eval)
    _fail_if(failures, "direct_rows_from_expected_abstain", direct_from_abstain)
    _fail_if(failures, "unsupported_direct_family_rows", unsupported_direct)
    _fail_if(failures, "unverified_direct_sft_rows", unverified_sft)
    missing_count = sum(missing_by_file.values())
    warnings = []
    if missing_count:
        warnings.append("missing_source_metadata_in_non_sft_or_legacy_rows")
    return {
        "status": "FAIL" if failures else ("WARN" if warnings else "PASS"),
        "checked_files": [str(path) for path in corpus_paths.values()] + [str(path) for path in eval_paths.values()],
        "corpus_row_count": sum(len(rows) for rows in corpus_rows.values()),
        "eval_row_count": len(all_eval_rows),
        "source_rule_id_overlap_with_rule_holdout": len(source_rule_overlap),
        "source_leakage_group_overlap_with_anti_leak": len(source_lg_overlap),
        "source_prompt_hash_overlap_with_eval": len(source_prompt_overlap) + len(corpus_prompt_overlap),
        "direct_source_row_id_duplicates": len(duplicate_source_ids),
        "direct_source_row_id_in_eval_ids": len(source_ids_in_eval),
        "direct_rows_from_expected_abstain": len(direct_from_abstain),
        "unsupported_direct_family_rows": len(unsupported_direct),
        "unverified_direct_sft_rows": len(unverified_sft),
        "missing_source_metadata_count": missing_count,
        "missing_source_metadata_by_file": missing_by_file,
        "missing_source_metadata_examples": missing_examples,
        "warnings": warnings,
        "failures": failures,
    }


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _fail_if(failures: list[dict[str, Any]], code: str, examples: list[Any]) -> None:
    if examples:
        failures.append({"code": code, "count": len(examples), "examples": examples[:20]})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = build_source_leakage_audit()
    write_json_checked(args.out, report, field_name="train_v2_source_leakage_audit")
    print(json.dumps({"status": report["status"], "out": args.out}, sort_keys=True))
    return 0 if report["status"] != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
