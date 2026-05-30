from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kaggle_sprint4.kaggle_train_curated_adapter import (  # noqa: E402
    IGNORE_INDEX,
    build_tokenized_dataset,
    messages_for_answer_format,
    tokenize_assistant_only,
    validate_supervision_diagnostics,
)


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = f"USER: {messages[0]['content']}\nASSISTANT: "
        if len(messages) > 1:
            text += messages[1]["content"]
        if add_generation_prompt:
            return text
        return text

    def __call__(self, text, truncation=True, max_length=128, padding=False):
        ids = [ord(ch) + 1 for ch in text]
        if truncation:
            ids = ids[:max_length]
        if padding == "max_length":
            ids = ids + [self.pad_token_id] * (max_length - len(ids))
        return {"input_ids": ids}

    def decode(self, ids):
        return "".join(chr(token - 1) for token in ids if token)


def row(answer: str = "\\boxed{42}") -> dict:
    return {
        "problem_id": "p1",
        "family": "bit_manipulation",
        "messages": [
            {"role": "user", "content": "SECRET_PROMPT Return only the final answer."},
            {"role": "assistant", "content": answer},
        ],
    }


def supervised_text(payload: dict, tokenizer: TinyTokenizer) -> str:
    return tokenizer.decode([token for token, label in zip(payload["input_ids"], payload["labels"]) if label != IGNORE_INDEX])


def test_assistant_only_labels_mask_prompt_tokens_raw_answer() -> None:
    tokenizer = TinyTokenizer()
    payload = tokenize_assistant_only(row(), tokenizer, max_seq_len=128, answer_format="raw")

    assert supervised_text(payload, tokenizer) == "42"
    assert "SECRET_PROMPT" not in supervised_text(payload, tokenizer)


def test_assistant_only_labels_preserve_boxed_answer_only() -> None:
    tokenizer = TinyTokenizer()
    payload = tokenize_assistant_only(row(), tokenizer, max_seq_len=128, answer_format="boxed")

    assert supervised_text(payload, tokenizer) == "\\boxed{42}"


def test_messages_for_raw_and_boxed_answer_formats() -> None:
    assert messages_for_answer_format(row("\\boxed{00110100}"), answer_format="raw")[1]["content"] == "00110100"
    assert messages_for_answer_format(row("\\boxed{00110100}"), answer_format="boxed")[1]["content"] == "\\boxed{00110100}"


def test_zero_supervised_tokens_rejected() -> None:
    with pytest.raises(ValueError):
        validate_supervision_diagnostics({"zero_supervised_rows": 1, "supervised_ratio_mean": 0.0})


def test_too_high_supervised_ratio_rejected() -> None:
    with pytest.raises(ValueError):
        validate_supervision_diagnostics({"zero_supervised_rows": 0, "supervised_ratio_mean": 0.75})
