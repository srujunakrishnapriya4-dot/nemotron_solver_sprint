from __future__ import annotations


CATEGORY_WEIGHTS = {
    "binary": 169 / 950,
    "cipher": 162 / 950,
    "cipher_digit": 85 / 950,
    "gravity": 159 / 950,
    "numeral": 149 / 950,
    "symbol_digit": 55 / 950,
    "unit_conv": 171 / 950,
}


def vllm_eval_settings(mode: str = "smoke_16") -> dict:
    rows = {"smoke_3": 3, "smoke_16": 16, "family_60": 60, "validation_300": 300, "vex_held_v2": None, "vex_held_v3": None}.get(mode)
    serious = mode in {"vex_held_v2", "vex_held_v3"}
    if rows is None:
        rows = 0 if serious else None
    if rows is None:
        raise ValueError(f"unknown eval mode: {mode}")
    return {
        "mode": mode,
        "max_rows": rows,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_model_len": 8192,
        "max_tokens": 7680 if serious else (16 if mode in {"smoke_3", "smoke_16"} else 32),
        "max_num_seqs": 64,
        "gpu_memory_utilization": 0.85,
        "requires_vllm": True,
        "uses_lora_request": True,
        "enable_lora": True,
        "max_lora_rank": 32,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
        "category_weights": CATEGORY_WEIGHTS,
        "held_variant_primary": "v2",
        "held_variant_secondary": "v3",
        "outputs": ("eval_per_puzzle.jsonl", "eval_summary.json"),
    }
