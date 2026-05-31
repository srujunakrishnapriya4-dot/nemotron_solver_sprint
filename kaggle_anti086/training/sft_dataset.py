from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Protocol

from kaggle_anti086.data.v2_corpus_io import read_jsonl
from kaggle_anti086.training.prepare_tokenization_dry_run import UNSUPPORTED_SFT


SYSTEM_PROMPT = "You are a reasoning model. Respond with only the final answer."
ALLOWED_SFT_SOURCES = {"direct_answer", "solver_corrected"}


class TokenizerLike(Protocol):
    pad_token_id: int | None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...


@dataclass(frozen=True)
class SFTItem:
    row: dict[str, Any]
    source: str
    sampling_weight: float


def render_sft_example(row: dict[str, Any]) -> dict[str, str]:
    messages = row.get("messages", [])
    user = str(messages[0].get("content", "")) if messages else str(row.get("prompt", ""))
    assistant = str(messages[1].get("content", "")) if len(messages) > 1 else str(row.get("answer", ""))
    return {
        "system": SYSTEM_PROMPT,
        "user": user,
        "assistant": assistant,
        "text": f"SYSTEM:\n{SYSTEM_PROMPT}\n\nUSER:\n{user}\n\nASSISTANT:\n{assistant}",
    }


def build_assistant_only_features(row: dict[str, Any], tokenizer: TokenizerLike, max_seq_len: int) -> dict[str, Any]:
    rendered = render_sft_example(row)
    system_ids = tokenizer.encode(f"SYSTEM:\n{rendered['system']}\n\n", add_special_tokens=False)
    user_ids = tokenizer.encode(f"USER:\n{rendered['user']}\n\n", add_special_tokens=False)
    assistant_prefix_ids = tokenizer.encode("ASSISTANT:\n", add_special_tokens=False)
    answer_ids = tokenizer.encode(rendered["assistant"], add_special_tokens=False)
    input_ids = system_ids + user_ids + assistant_prefix_ids + answer_ids
    labels = [-100] * (len(system_ids) + len(user_ids) + len(assistant_prefix_ids)) + answer_ids[:]
    attention_mask = [1] * len(input_ids)
    original_len = len(input_ids)
    if original_len > max_seq_len:
        input_ids = input_ids[:max_seq_len]
        labels = labels[:max_seq_len]
        attention_mask = attention_mask[:max_seq_len]
    answer_start = len(system_ids) + len(user_ids) + len(assistant_prefix_ids)
    supervised = sum(1 for label in labels if label != -100)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "system_span": [0, len(system_ids)],
        "user_span": [len(system_ids), len(system_ids) + len(user_ids)],
        "assistant_prefix_span": [len(system_ids) + len(user_ids), answer_start],
        "answer_span": [answer_start, answer_start + len(answer_ids)],
        "answer_token_count": len(answer_ids),
        "supervised_token_count": supervised,
        "truncated": original_len > max_seq_len,
        "metadata": {
            "id": row.get("id"),
            "family": row.get("family"),
            "subfamily": row.get("subfamily"),
            "rule_id": row.get("rule_id"),
            "source": row.get("source"),
            "verification_status": row.get("verification_status"),
        },
    }


def load_weighted_sft_rows(config: dict[str, Any]) -> list[SFTItem]:
    items: list[SFTItem] = []
    for row in read_jsonl(config["train_direct_path"]):
        if _row_allowed(row):
            items.append(SFTItem(row=row, source="direct_answer", sampling_weight=1.0))
    for row in read_jsonl(config["train_solver_corrected_path"]):
        if _row_allowed(row):
            items.append(SFTItem(row=row, source="solver_corrected", sampling_weight=0.5))
    return items


def build_sft_dataset_report(config: dict[str, Any], rows: list[SFTItem] | None = None) -> dict[str, Any]:
    rows = rows if rows is not None else load_weighted_sft_rows(config)
    family_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    unsupported = 0
    unverified = 0
    zero_answer = 0
    missing_required = 0
    effective = 0.0
    for item in rows:
        row = item.row
        family_counts[str(row.get("family", "unknown"))] += 1
        source_counts[item.source] += 1
        effective += item.sampling_weight
        if row.get("family") in UNSUPPORTED_SFT:
            unsupported += 1
        if row.get("verification_status") != "verified":
            unverified += 1
        if not str(row.get("answer", "")).strip():
            zero_answer += 1
        for key in ("id", "rule_id", "leakage_group"):
            if not row.get(key):
                missing_required += 1
    failures = []
    if unsupported:
        failures.append("unsupported_rows_nonzero")
    if unverified:
        failures.append("unverified_rows_nonzero")
    if zero_answer:
        failures.append("zero_answer_rows_nonzero")
    if missing_required:
        failures.append("missing_required_metadata")
    return {
        "status": "PASS" if not failures else "FAIL",
        "sft_row_count": len(rows),
        "direct_rows": source_counts.get("direct_answer", 0),
        "solver_corrected_rows": source_counts.get("solver_corrected", 0),
        "effective_sft_rows": effective,
        "unsupported_rows": unsupported,
        "unverified_rows": unverified,
        "zero_answer_rows": zero_answer,
        "missing_required_metadata": missing_required,
        "family_counts": dict(sorted(family_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "max_seq_len": int(config.get("max_seq_len", 1024)),
        "failures": failures,
    }


class Sprint11SFTDataset:
    def __init__(self, rows: list[SFTItem], tokenizer: TokenizerLike, max_seq_len: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.rows[index]
        features = build_assistant_only_features(item.row, self.tokenizer, self.max_seq_len)
        features["sampling_weight"] = item.sampling_weight
        features["source"] = item.source
        return features


def _row_allowed(row: dict[str, Any]) -> bool:
    return (
        row.get("verification_status") == "verified"
        and row.get("family") not in UNSUPPORTED_SFT
        and bool(str(row.get("answer", "")).strip())
        and bool(row.get("rule_id"))
        and bool(row.get("leakage_group"))
    )
