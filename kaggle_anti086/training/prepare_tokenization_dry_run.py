from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import statistics
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.v2_corpus_io import read_jsonl, write_json_checked
from kaggle_anti086.training.training_config_schema import load_training_config


UNSUPPORTED_SFT = {"equation_operator", "sequence_pattern", "permutation_sorting", "custom_numeral", "unknown"}


def fallback_tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", text)


def build_tokenization_dry_run(config: dict[str, Any], *, sample_size: int = 256) -> dict[str, Any]:
    direct = read_jsonl(config["train_direct_path"])
    corrected = read_jsonl(config["train_solver_corrected_path"])
    abstain = read_jsonl(config["train_abstain_safety_path"])
    hardneg = read_jsonl(config["train_hard_negative_path"])
    eligible = [("direct", row) for row in direct] + [("solver_corrected", row) for row in corrected]
    sample = _balanced_sample(eligible, sample_size)
    lengths = []
    family_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    empty_prompt = 0
    empty_answer = 0
    unsupported = 0
    answer_missing = 0
    max_seq_len = int(config.get("max_seq_len", 1024))
    for source, row in sample:
        prompt = _user_content(row)
        answer = _assistant_content(row)
        if not prompt.strip():
            empty_prompt += 1
        if not answer.strip():
            empty_answer += 1
        if row.get("family") in UNSUPPORTED_SFT:
            unsupported += 1
        prompt_tokens = fallback_tokenize(prompt)
        answer_tokens = fallback_tokenize(answer)
        if not answer_tokens:
            answer_missing += 1
        lengths.append(len(prompt_tokens) + len(answer_tokens))
        family_counts[str(row.get("family", "unknown"))] += 1
        source_counts[source] += 1
    truncation = sum(1 for length in lengths if length > max_seq_len)
    failures = []
    if len(sample) < sample_size:
        failures.append("sample_count_below_256")
    if lengths and truncation / len(lengths) > 0.02:
        failures.append("truncation_rate_above_0_02")
    if empty_prompt:
        failures.append("empty_prompt_count_nonzero")
    if empty_answer:
        failures.append("empty_answer_count_nonzero")
    if unsupported:
        failures.append("unsupported_sft_rows_nonzero")
    if answer_missing:
        failures.append("answer_tokens_missing")
    return {
        "status": "PASS" if not failures else "FAIL",
        "sample_count": len(sample),
        "max_seq_len": max_seq_len,
        "truncation_count": truncation,
        "truncation_rate": truncation / len(lengths) if lengths else 0.0,
        "empty_prompt_count": empty_prompt,
        "empty_answer_count": empty_answer,
        "unsupported_sft_rows": unsupported,
        "excluded_abstain_safety_rows": len(abstain),
        "excluded_hard_negative_rows": len(hardneg),
        "family_counts": dict(sorted(family_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "token_length_min": min(lengths) if lengths else 0,
        "token_length_p50": statistics.median(lengths) if lengths else 0,
        "token_length_p95": _percentile(lengths, 0.95),
        "token_length_max": max(lengths) if lengths else 0,
        "warnings": [],
        "failures": failures,
    }


def _balanced_sample(items: list[tuple[str, dict[str, Any]]], count: int) -> list[tuple[str, dict[str, Any]]]:
    direct = [item for item in items if item[0] == "direct"]
    corrected = [item for item in items if item[0] == "solver_corrected"]
    if direct and corrected:
        return _balanced_sample_one_source(direct, count // 2) + _balanced_sample_one_source(corrected, count - count // 2)
    return _balanced_sample_one_source(items, count)


def _balanced_sample_one_source(items: list[tuple[str, dict[str, Any]]], count: int) -> list[tuple[str, dict[str, Any]]]:
    buckets: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for item in items:
        buckets.setdefault(str(item[1].get("family", "unknown")), []).append(item)
    sample = []
    while len(sample) < count and any(buckets.values()):
        for family in sorted(buckets):
            if buckets[family] and len(sample) < count:
                sample.append(buckets[family].pop(0))
    return sample


def _user_content(row: dict[str, Any]) -> str:
    messages = row.get("messages", [])
    return str(messages[0].get("content", "")) if messages else str(row.get("prompt", ""))


def _assistant_content(row: dict[str, Any]) -> str:
    messages = row.get("messages", [])
    return str(messages[1].get("content", "")) if len(messages) > 1 else str(row.get("answer", ""))


def _percentile(values: list[int], q: float) -> float:
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((len(ordered) - 1) * q)))
    return ordered[idx]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = build_tokenization_dry_run(load_training_config(args.config))
    write_json_checked(args.out, report, field_name="day7_tokenization_dry_run_report")
    print(json.dumps({"status": report["status"], "sample_count": report["sample_count"], "out": args.out}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
