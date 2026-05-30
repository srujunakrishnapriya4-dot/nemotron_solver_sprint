from __future__ import annotations

REQUIRED = ("row_count", "total_tokens", "supervised_tokens", "prompt_leakage_count", "zero_supervised_rows", "max_seq_len", "truncation_count", "corpus_hash")


def validate_train_manifest(manifest: dict) -> dict:
    missing = [key for key in REQUIRED if key not in manifest]
    if missing:
        raise ValueError(f"missing manifest keys: {missing}")
    if manifest["zero_supervised_rows"] != 0 or manifest["prompt_leakage_count"] != 0:
        raise ValueError("unsafe token mask manifest")
    if manifest["supervised_tokens"] <= 0:
        raise ValueError("no supervised tokens")
    return manifest
