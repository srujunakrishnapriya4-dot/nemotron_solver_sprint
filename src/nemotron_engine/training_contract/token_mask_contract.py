from __future__ import annotations

from typing import Any


class TokenMaskContractError(ValueError):
    pass


def validate_token_mask_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    if not rows:
        raise TokenMaskContractError("no token rows")
    row_count = total_tokens = supervised_tokens = zero = leakage = truncation = 0
    for row in rows:
        row_count += 1
        input_ids = row.get("input_ids", [])
        target_ids = row.get("target_ids", [])
        weights = row.get("loss_weights", [])
        if len(input_ids) != len(target_ids) or len(input_ids) != len(weights):
            raise TokenMaskContractError("target_ids/loss_weights length mismatch")
        total_tokens += len(input_ids)
        sup_positions = [i for i, value in enumerate(weights) if float(value) > 0]
        supervised_tokens += len(sup_positions)
        if not sup_positions:
            zero += 1
        prompt_token_count = int(row.get("prompt_token_count", row.get("metadata", {}).get("prompt_token_count", min(sup_positions) if sup_positions else len(weights))))
        if any(i < prompt_token_count for i in sup_positions):
            leakage += 1
        if row.get("truncated"):
            truncation += 1
    if zero:
        raise TokenMaskContractError(f"zero supervised rows: {zero}")
    if leakage:
        raise TokenMaskContractError(f"prompt leakage rows: {leakage}")
    return {"row_count": row_count, "total_tokens": total_tokens, "supervised_tokens": supervised_tokens, "zero_supervised_rows": zero, "prompt_leakage_count": leakage, "truncation_count": truncation}
