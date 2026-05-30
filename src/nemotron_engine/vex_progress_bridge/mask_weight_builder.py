from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class MaskWeightBuilderError(ValueError):
    pass


@dataclass(frozen=True)
class MaskDiagnostics:
    supervised_token_count_min: int
    supervised_token_count_mean: float
    supervised_token_count_max: int
    zero_supervised_rows: int
    prompt_supervised_leak_count: int
    total_unmasked_tokens: int
    answer_token_ratio_mean: float
    decoded_supervised_spans: tuple[str, ...]


def build_char_level_mask(text: str, answer: str) -> dict[str, Any]:
    if not answer:
        raise MaskWeightBuilderError("answer must be non-empty")
    assistant_index = text.rfind("Assistant:")
    if assistant_index < 0:
        raise MaskWeightBuilderError("Assistant span missing")
    answer_index = text.rfind(answer)
    if answer_index < assistant_index:
        raise MaskWeightBuilderError("answer absent from assistant supervised span")
    token_ids = [ord(ch) for ch in text]
    target_ids = list(token_ids)
    loss_weights = [0.0 for _ in token_ids]
    for index in range(answer_index, answer_index + len(answer)):
        loss_weights[index] = 1.0
    return {
        "token_ids": token_ids,
        "target_ids": target_ids,
        "loss_weights": loss_weights,
        "prompt_token_count": answer_index,
        "supervised_token_count": len(answer),
        "supervised_text": text[answer_index : answer_index + len(answer)],
    }


def validate_masked_rows(rows: list[dict[str, Any]], *, decoded_spans: int = 2) -> MaskDiagnostics:
    if not rows:
        raise MaskWeightBuilderError("no masked rows")
    counts = [int(row.get("supervised_token_count", 0)) for row in rows]
    zero = sum(1 for count in counts if count <= 0)
    prompt_leaks = 0
    total_unmasked = 0
    ratios = []
    spans = []
    for row in rows:
        token_ids = row.get("token_ids")
        target_ids = row.get("target_ids")
        weights = row.get("loss_weights")
        if not isinstance(token_ids, list) or not isinstance(target_ids, list) or not isinstance(weights, list):
            raise MaskWeightBuilderError("token_ids, target_ids, and loss_weights are required lists")
        if len(token_ids) != len(target_ids):
            raise MaskWeightBuilderError("target_ids length mismatch")
        if len(token_ids) != len(weights):
            raise MaskWeightBuilderError("loss_weights length mismatch")
        supervised_indices = [idx for idx, weight in enumerate(weights) if float(weight) > 0]
        total_unmasked += len(supervised_indices)
        ratios.append(len(supervised_indices) / len(token_ids) if token_ids else 0.0)
        prompt_count = int(row.get("prompt_token_count", 0))
        if any(idx < prompt_count for idx in supervised_indices):
            prompt_leaks += 1
        if len(spans) < decoded_spans:
            spans.append(str(row.get("supervised_text", "")))
    if zero:
        raise MaskWeightBuilderError("zero supervised rows")
    if prompt_leaks:
        raise MaskWeightBuilderError("prompt fragments supervised")
    return MaskDiagnostics(
        supervised_token_count_min=min(counts),
        supervised_token_count_mean=sum(counts) / len(counts),
        supervised_token_count_max=max(counts),
        zero_supervised_rows=zero,
        prompt_supervised_leak_count=prompt_leaks,
        total_unmasked_tokens=total_unmasked,
        answer_token_ratio_mean=sum(ratios) / len(ratios),
        decoded_supervised_spans=tuple(spans),
    )
