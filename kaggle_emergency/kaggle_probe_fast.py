from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


CUTLASS_PATH = Path("/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/nvidia_cutlass_dsl/python_packages")


def setup_cutlass_path() -> None:
    import sys

    if CUTLASS_PATH.exists():
        sys.path.insert(0, str(CUTLASS_PATH))


def extract_answer(text: str) -> str:
    text = text.strip()
    if "\\boxed{" in text and "}" in text:
        start = text.rfind("\\boxed{") + len("\\boxed{")
        end = text.find("}", start)
        if end >= 0:
            return text[start:end].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text


def load_probe_rows(path: str | Path, max_rows: int) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
                if len(rows) >= max_rows:
                    break
    return rows


def gold_from_row(row: dict) -> str:
    messages = row.get("messages") or []
    if len(messages) < 2:
        return ""
    return extract_answer(str(messages[1].get("content", "")))


def prompt_from_row(row: dict) -> str:
    messages = row.get("messages") or []
    if not messages:
        return ""
    return str(messages[0].get("content", ""))


def run_probe(adapter_dir: str, base_model_path: str, validation_file: str, output_json: str, *, max_probe_rows: int = 3, max_new_tokens: int = 16, timeout_seconds: float = 120.0) -> dict:
    setup_cutlass_path()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = load_probe_rows(validation_file, max_probe_rows)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
    model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    model.eval()
    results = []
    start_all = time.time()
    for row in rows:
        start = time.time()
        prompt = prompt_from_row(row)
        rendered = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True) if hasattr(tokenizer, "apply_chat_template") else f"USER: {prompt}\nASSISTANT: "
        encoded = tokenizer(rendered, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        raw = tokenizer.decode(output[0][encoded["input_ids"].shape[-1] :], skip_special_tokens=True).strip()
        extracted = extract_answer(raw)
        expected = gold_from_row(row)
        elapsed = time.time() - start
        print(str(row.get("problem_id")), row.get("family"), expected, raw, extracted, extracted == expected, elapsed)
        results.append({"id": str(row.get("problem_id")), "family": row.get("family"), "expected": expected, "raw_output": raw, "extracted": extracted, "exact_match": extracted == expected, "seconds": elapsed})
        if time.time() - start_all > timeout_seconds:
            raise SystemExit("PROBE_TOO_SLOW")
    report = {"results": results, "sane_outputs": all(bool(item["extracted"]) for item in results), "exact_match_count": sum(1 for item in results if item["exact_match"])}
    Path(output_json).write_text(json.dumps(report, sort_keys=True, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--validation-file", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--max-probe-rows", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()
    report = run_probe(args.adapter_dir, args.base_model_path, args.validation_file, args.output_json, max_probe_rows=args.max_probe_rows, max_new_tokens=args.max_new_tokens)
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
