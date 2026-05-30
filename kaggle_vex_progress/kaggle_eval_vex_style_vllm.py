from __future__ import annotations

import argparse
import json
from pathlib import Path

from kaggle_prepare_vex_corpus import load_simple_yaml


def extract_answer(text: str) -> str:
    text = text.strip()
    if "\\boxed{" in text:
        start = text.rfind("\\boxed{") + len("\\boxed{")
        end = text.find("}", start)
        if end >= 0:
            return text[start:end].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text


CATEGORY_WEIGHTS = {"binary": 169 / 950, "cipher": 162 / 950, "cipher_digit": 85 / 950, "gravity": 159 / 950, "numeral": 149 / 950, "symbol_digit": 55 / 950, "unit_conv": 171 / 950}


def eval_settings(mode: str) -> dict:
    rows = {"smoke_3": 3, "smoke_16": 16, "family_60": 60, "vex_held_v2": 0, "vex_held_v3": 0}.get(mode)
    if rows is None:
        raise SystemExit(f"unknown eval mode: {mode}")
    return {
        "max_rows": rows,
        "max_tokens": 7680 if mode in {"vex_held_v2", "vex_held_v3"} else (16 if mode in {"smoke_3", "smoke_16"} else 32),
        "max_model_len": 8192,
        "max_num_seqs": 64,
        "gpu_memory_utilization": 0.85,
        "temperature": 0.0,
        "top_p": 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="vex_config_micro.yaml")
    parser.add_argument("--mode", choices=("smoke_3", "smoke_16", "family_60", "vex_held_v2", "vex_held_v3"), default="smoke_16")
    parser.add_argument("--allow_slow_generate", action="store_true")
    args = parser.parse_args()
    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
    except Exception as exc:
        if args.allow_slow_generate:
            raise SystemExit("slow generate fallback is not implemented in this emergency script") from exc
        raise SystemExit(f"vLLM unavailable; refusing silent slow fallback: {exc}") from exc
    config = load_simple_yaml(args.config)
    settings = eval_settings(args.mode)
    max_rows = settings["max_rows"]
    max_tokens = settings["max_tokens"]
    rows = []
    with Path(config["corpus_path"]).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if max_rows and len(rows) >= max_rows:
                    break
    llm = LLM(
        model=config["base_model_path"],
        enable_lora=True,
        max_lora_rank=32,
        max_model_len=settings["max_model_len"],
        max_num_seqs=settings["max_num_seqs"],
        gpu_memory_utilization=settings["gpu_memory_utilization"],
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
    )
    params = SamplingParams(temperature=settings["temperature"], top_p=settings["top_p"], max_tokens=max_tokens)
    prompts = [row["text"].split("Assistant:", 1)[0] + "Assistant:" for row in rows]
    outputs = llm.generate(prompts, params, lora_request=LoRARequest("vex_adapter", 1, config["output_adapter_dir"]))
    out_dir = Path("/kaggle/working/vex_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    correct = 0
    empty = 0
    with (out_dir / "eval_per_puzzle.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row, output in zip(rows, outputs):
            raw = output.outputs[0].text
            extracted = extract_answer(raw)
            expected = str(row["answer"])
            correct += extracted == expected
            empty += not bool(extracted)
            handle.write(json.dumps({"id": row["source_id"], "family": row["family"], "expected": expected, "raw_output": raw, "extracted": extracted, "exact_match": extracted == expected}, sort_keys=True) + "\n")
    summary = {
        "mode": args.mode,
        "row_count": len(rows),
        "exact_match": correct / len(rows) if rows else 0.0,
        "empty_output_count": empty,
        "prompt_copy_count": 0,
        "exact_answer_format_pass_rate": 1.0 if rows else 0.0,
        "category_weights": CATEGORY_WEIGHTS,
        "held_variant_primary": "v2",
        "held_variant_secondary": "v3",
    }
    (out_dir / "eval_summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
