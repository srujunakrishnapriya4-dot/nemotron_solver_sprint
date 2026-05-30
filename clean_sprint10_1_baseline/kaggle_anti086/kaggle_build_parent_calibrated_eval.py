from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from kaggle_prepare_anti086_tokens import load_simple_yaml, resolve_anti086_input_root, transform_direct_answer_row
from kaggle_path_safety import require_safe_config_output_paths, require_writable_output_path, safe_open_text_for_write, safe_write_text
from kaggle_runtime_patches import apply_runtime_patches, ensure_parent_adapter_in_config
from kaggle_eval_anti086_vllm import run_hf_fallback_eval, _prompt_from_row


FAMILIES = ("bit_manipulation", "gravity_numeric", "unit_conversion", "cipher_text", "roman_numeral", "equation_symbolic")


def select_candidate_rows(source_rows: list[dict], *, limit: int = 64) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in source_rows:
        fam = str(row.get("family", "unknown"))
        if fam in FAMILIES:
            if fam == "equation_symbolic" and row.get("corpus_type") not in {"short_rule_trace", "direct_answer_only", "direct_raw"}:
                continue
            buckets[fam].append(transform_direct_answer_row(row))
    selected: list[dict] = []
    while len(selected) < limit and any(buckets.values()):
        for fam in FAMILIES:
            if buckets[fam] and len(selected) < limit:
                selected.append(buckets[fam].pop(0))
    return selected


def classify_parent_records(records: list[dict]) -> dict:
    parent_correct = [row for row in records if row.get("exact_match")]
    parent_wrong = [row for row in records if not row.get("exact_match")]
    by_family = Counter()
    correct_by_family = Counter()
    for row in records:
        fam = str(row.get("family", "unknown"))
        by_family[fam] += 1
        if row.get("exact_match"):
            correct_by_family[fam] += 1
    parent_exact = len(parent_correct) / max(1, len(records))
    return {
        "valid": bool(records) and parent_exact > 0 and len(parent_wrong) > 0,
        "invalid_reason": None if bool(records) and parent_exact > 0 and len(parent_wrong) > 0 else ("parent_zero" if parent_exact == 0 else "no_parent_wrong_rows"),
        "parent_exact_match": parent_exact,
        "parent_by_family": {fam: {"count": by_family[fam], "correct": correct_by_family[fam], "exact_match": correct_by_family[fam] / max(1, by_family[fam])} for fam in sorted(by_family)},
        "parent_correct_ids": [row.get("id") for row in parent_correct],
        "parent_wrong_ids": [row.get("id") for row in parent_wrong],
        "prompt_hashes": [row.get("prompt_hash") for row in records],
        "exact_extraction_details": [row.get("extraction_detail", {}) for row in records[:10]],
    }


def load_source_rows(config: dict) -> list[dict]:
    root = resolve_anti086_input_root(config)
    source = root / str(config.get("corpus_file", "corpus_anti086_v1.jsonl"))
    if not source.exists() and str(config.get("corpus_file")) == "corpus_anti086_v1.jsonl":
        source = root / "win_v1.jsonl"
    if not source.exists():
        source = Path("artifacts/win_system/win_v1.jsonl")
    if not source.exists():
        raise SystemExit(f"missing source rows for parent calibration: {source}")
    return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path = require_writable_output_path(path, field_name="calibrated_eval_path")
    with safe_open_text_for_write(path, field_name="calibrated_eval_path", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="anti086_winmode_v1b.yaml")
    args = parser.parse_args()
    runtime_patch = apply_runtime_patches()
    config = load_simple_yaml(args.config)
    require_safe_config_output_paths(config, stage=str(config.get("stage", "parent_calibrated_eval")))
    parent_adapter = ensure_parent_adapter_in_config(args.config, config)
    rows = select_candidate_rows(load_source_rows(config), limit=int(config.get("max_eval_rows", 64)))
    eval_path = require_writable_output_path(str(config.get("calibrated_eval_path", "/kaggle/working/anti086_eval/v1b/parent_calibrated_eval.jsonl")), field_name="calibrated_eval_path")
    write_jsonl(eval_path, rows)
    prompts = [_prompt_from_row(row) for row in rows]
    parent_records = run_hf_fallback_eval(config, rows, prompts, adapter_path=parent_adapter) if parent_adapter else []
    report = classify_parent_records(parent_records)
    report.update({"runtime_patch": runtime_patch, "row_count": len(rows), "synthetic_only": all(row.get("corpus_type") == "synthetic_verified" for row in rows)})
    if report["synthetic_only"]:
        report["valid"] = False
        report["invalid_reason"] = "synthetic_only"
    report_path = require_writable_output_path(str(config.get("parent_calibration_report_path", "/kaggle/working/anti086_eval/v1b/parent_calibration_report.json")), field_name="parent_calibration_report_path")
    safe_write_text(report_path, json.dumps(report, sort_keys=True, indent=2), field_name="parent_calibration_report_path")
    print(json.dumps(report, sort_keys=True, indent=2))
    if not report["valid"]:
        raise SystemExit(f"invalid parent-calibrated eval: {report['invalid_reason']}")


if __name__ == "__main__":
    main()
