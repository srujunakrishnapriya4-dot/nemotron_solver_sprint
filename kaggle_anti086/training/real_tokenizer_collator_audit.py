from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Protocol

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.v2_corpus_io import read_jsonl, write_json_checked
from kaggle_anti086.training.prepare_tokenization_dry_run import UNSUPPORTED_SFT, fallback_tokenize
from kaggle_anti086.training.training_config_schema import load_training_config


SYSTEM_PROMPT = "You are a reasoning model. Respond with only the final answer."
COLLATOR_CONTRACT = "assistant_only_answer_loss"


class TokenizerLike(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...


class FallbackAuditTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [abs(hash(token)) % 50000 for token in fallback_tokenize(text)]


def load_real_tokenizer(base_model_path: str) -> tuple[Any | None, list[str]]:
    warnings: list[str] = []
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local optional deps
        return None, [f"transformers_unavailable:{type(exc).__name__}"]
    try:
        return AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True), warnings
    except Exception as exc:  # pragma: no cover - depends on local model files
        return None, [f"tokenizer_unavailable:{type(exc).__name__}"]


def build_labels_for_row(row: dict[str, Any], tokenizer: TokenizerLike, *, supervise: bool, max_seq_len: int) -> dict[str, Any]:
    messages = row.get("messages", [])
    user = str(messages[0].get("content", "")) if messages else str(row.get("prompt", ""))
    assistant = str(messages[1].get("content", "")) if len(messages) > 1 else str(row.get("answer", ""))
    system_ids = tokenizer.encode(SYSTEM_PROMPT + "\n", add_special_tokens=False)
    user_ids = tokenizer.encode("USER:\n" + user + "\n", add_special_tokens=False)
    assistant_prefix_ids = tokenizer.encode("ASSISTANT:\n", add_special_tokens=False)
    answer_ids = tokenizer.encode(assistant, add_special_tokens=False)
    input_ids = system_ids + user_ids + assistant_prefix_ids + answer_ids
    labels = [-100] * (len(system_ids) + len(user_ids) + len(assistant_prefix_ids))
    labels += answer_ids[:] if supervise else [-100] * len(answer_ids)
    if len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]
        labels = labels[:max_seq_len]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "system_span": [0, len(system_ids)],
        "user_span": [len(system_ids), len(system_ids) + len(user_ids)],
        "assistant_prefix_span": [len(system_ids) + len(user_ids), len(system_ids) + len(user_ids) + len(assistant_prefix_ids)],
        "answer_span": [len(system_ids) + len(user_ids) + len(assistant_prefix_ids), len(system_ids) + len(user_ids) + len(assistant_prefix_ids) + len(answer_ids)],
        "answer_token_count": len(answer_ids),
        "truncated": len(system_ids) + len(user_ids) + len(assistant_prefix_ids) + len(answer_ids) > max_seq_len,
    }


def select_audit_sample(config: dict[str, Any], *, sample_size: int) -> list[tuple[str, dict[str, Any]]]:
    direct = [("direct_answer", row) for row in read_jsonl(config["train_direct_path"])]
    corrected = [("solver_corrected", row) for row in read_jsonl(config["train_solver_corrected_path"])]
    candidates = direct + corrected
    buckets: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for item in candidates:
        row = item[1]
        if row.get("family") in UNSUPPORTED_SFT or row.get("verification_status") != "verified":
            continue
        buckets.setdefault(str(row.get("family", "unknown")), []).append(item)
    sample: list[tuple[str, dict[str, Any]]] = []
    while len(sample) < sample_size and any(buckets.values()):
        for family in sorted(buckets):
            if buckets[family] and len(sample) < sample_size:
                sample.append(buckets[family].pop(0))
    return sample


def build_real_collator_audit(config: dict[str, Any], *, sample_size: int = 128, kaggle_mode: bool = False, tokenizer: TokenizerLike | None = None) -> dict[str, Any]:
    warnings: list[str] = []
    failures: list[str] = []
    tokenizer_loaded = tokenizer is not None
    mode = "real_tokenizer" if tokenizer_loaded else "local_dev_warning"
    if tokenizer is None:
        tokenizer, load_warnings = load_real_tokenizer(str(config.get("base_model_path", "")))
        warnings.extend(load_warnings)
        tokenizer_loaded = tokenizer is not None
    if tokenizer is None:
        if kaggle_mode:
            failures.append("tokenizer_unavailable_in_kaggle_mode")
            mode = "local_fallback_blocked"
        else:
            mode = "local_dev_warning"
        tokenizer = FallbackAuditTokenizer()
    sample = select_audit_sample(config, sample_size=sample_size)
    max_seq_len = int(config.get("max_seq_len", 1024))
    counts = Counter()
    examples = []
    for source, row in sample:
        supervise = source in {"direct_answer", "solver_corrected"} and row.get("family") not in UNSUPPORTED_SFT and row.get("verification_status") == "verified"
        features = build_labels_for_row(row, tokenizer, supervise=supervise, max_seq_len=max_seq_len)
        labels = features["labels"]
        counts["truncation_count"] += int(bool(features["truncated"]))
        answer_start, answer_end = features["answer_span"]
        for idx, label in enumerate(labels):
            if label == -100:
                continue
            if idx < features["system_span"][1]:
                counts["system_tokens_supervised"] += 1
            elif idx < features["user_span"][1]:
                counts["user_tokens_supervised"] += 1
            elif idx < answer_start:
                counts["prompt_tokens_supervised"] += 1
            elif idx < answer_end:
                counts["assistant_answer_tokens_supervised"] += 1
        supervised = sum(1 for label in labels if label != -100)
        if supervise and supervised == 0:
            counts["zero_supervised_sft_rows"] += 1
        if row.get("family") in UNSUPPORTED_SFT and supervised:
            counts["unsupported_supervised_rows"] += 1
        if len(examples) < 5:
            examples.append({"id": row.get("id"), "source": source, "family": row.get("family"), "supervised_tokens": supervised})
    abstain = read_jsonl(config["train_abstain_safety_path"])[:20]
    hardneg = read_jsonl(config["train_hard_negative_path"])[:20]
    for source, rows in (("abstain_safety", abstain), ("hard_negative", hardneg)):
        for row in rows:
            features = build_labels_for_row(row, tokenizer, supervise=False, max_seq_len=max_seq_len)
            supervised = sum(1 for label in features["labels"] if label != -100)
            if source == "abstain_safety" and supervised:
                counts["abstain_safety_supervised_rows"] += 1
            if source == "hard_negative" and supervised:
                counts["hard_negative_supervised_rows"] += 1
    hard_zero_fields = [
        "prompt_tokens_supervised",
        "system_tokens_supervised",
        "user_tokens_supervised",
        "zero_supervised_sft_rows",
        "abstain_safety_supervised_rows",
        "hard_negative_supervised_rows",
        "unsupported_supervised_rows",
    ]
    for field in hard_zero_fields:
        if counts[field]:
            failures.append(f"{field}_nonzero")
    truncation_rate = counts["truncation_count"] / len(sample) if sample else 0.0
    if truncation_rate > 0.02:
        failures.append("truncation_rate_above_0_02")
    if len(sample) < sample_size:
        failures.append("sample_size_below_requested")
    status = "PASS" if not failures and tokenizer_loaded else ("WARN_LOCAL_TOKENIZER_UNAVAILABLE" if not failures else "FAIL")
    return {
        "status": status,
        "mode": "real_tokenizer" if tokenizer_loaded else mode,
        "sample_size": len(sample),
        "tokenizer_loaded": tokenizer_loaded,
        "collator_contract": COLLATOR_CONTRACT,
        "prompt_tokens_supervised": counts["prompt_tokens_supervised"],
        "system_tokens_supervised": counts["system_tokens_supervised"],
        "user_tokens_supervised": counts["user_tokens_supervised"],
        "assistant_answer_tokens_supervised": counts["assistant_answer_tokens_supervised"],
        "zero_supervised_sft_rows": counts["zero_supervised_sft_rows"],
        "abstain_safety_supervised_rows": counts["abstain_safety_supervised_rows"],
        "hard_negative_supervised_rows": counts["hard_negative_supervised_rows"],
        "unsupported_supervised_rows": counts["unsupported_supervised_rows"],
        "max_seq_len": max_seq_len,
        "truncation_count": counts["truncation_count"],
        "truncation_rate": truncation_rate,
        "examples": examples,
        "warnings": warnings,
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sample-size", type=int, default=128)
    parser.add_argument("--kaggle-mode", action="store_true")
    args = parser.parse_args(argv)
    report = build_real_collator_audit(load_training_config(args.config), sample_size=args.sample_size, kaggle_mode=args.kaggle_mode)
    write_json_checked(args.out, report, field_name="day8_real_collator_audit_report")
    print(json.dumps({"status": report["status"], "mode": report["mode"], "out": args.out}, sort_keys=True))
    return 0 if report["status"] in {"PASS", "WARN_LOCAL_TOKENIZER_UNAVAILABLE"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
