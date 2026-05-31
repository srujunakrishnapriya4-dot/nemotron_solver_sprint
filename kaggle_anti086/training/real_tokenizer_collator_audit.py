from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
from typing import Any, Protocol

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.v2_corpus_io import read_jsonl, write_json_checked
from kaggle_anti086.training.prepare_tokenization_dry_run import UNSUPPORTED_SFT, fallback_tokenize
from kaggle_anti086.training.sft_dataset import build_assistant_only_features
from kaggle_anti086.training.training_config_schema import load_training_config


SYSTEM_PROMPT = "You are a reasoning model. Respond with only the final answer."
COLLATOR_CONTRACT = "assistant_only_answer_loss"


class TokenizerLike(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...


class FallbackAuditTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __init__(self):
        self._vocab: dict[str, int] = {}
        self._inverse: dict[int, str] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        ids = []
        for token in fallback_tokenize(text):
            if token not in self._vocab:
                idx = len(self._vocab) + 1
                self._vocab[token] = idx
                self._inverse[idx] = token
            ids.append(self._vocab[token])
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        text = " ".join(self._inverse.get(int(idx), "") for idx in ids).strip()
        text = re.sub(r"\s+([^\w\s])", r"\1", text)
        text = re.sub(r"([^\w\s])\s+", r"\1", text)
        return text.strip()


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
    features = build_assistant_only_features(row, tokenizer, max_seq_len)
    if not supervise:
        features = dict(features)
        features["labels"] = [-100] * len(features["labels"])
        features["supervised_token_count"] = 0
    return features


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
        "shared_label_builder": True,
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
