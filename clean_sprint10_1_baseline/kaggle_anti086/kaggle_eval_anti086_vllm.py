from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from decimal import Decimal, InvalidOperation

from kaggle_prepare_anti086_tokens import load_simple_yaml
from kaggle_runtime_patches import apply_runtime_patches, require_real_optional_path

MICRO_LABEL = "INFRASTRUCTURE STACK TEST ONLY - NOT A 0.95 CANDIDATE - NOT MAIN TRAINING - NOT SUBMISSION READY"

def extract_answer(text: str) -> str:
    if "\\boxed{" in text:
        start = text.rfind("\\boxed{") + len("\\boxed{")
        end = text.find("}", start)
        if end >= 0:
            return text[start:end].strip()
    for pattern in (
        r"(?:answer|result|output)\s+is\s*[:\-]?\s*([A-Za-z0-9 ._\-]+)",
        r"decrypted text is\s*[:\-]?\s*([A-Za-z0-9 ._\-]+)",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _clean_answer(match.group(1))
    binary = re.search(r"\b[01]{4,64}\b", text)
    if binary:
        return binary.group(0)
    numeric = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text)
    if numeric:
        return numeric.group(0)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return _clean_answer(lines[-1]) if lines else ""


def _clean_answer(value: str) -> str:
    return value.strip().strip("`'\".,;: ")


def normalize_answer(value: str, family: str | None = None) -> str:
    text = _clean_answer(str(value))
    if family == "roman_numeral":
        return text.upper()
    if re.fullmatch(r"[01]{4,64}", text):
        return text
    if family in {"gravity_numeric", "unit_conversion"}:
        try:
            return str(Decimal(text).normalize())
        except (InvalidOperation, ValueError):
            return text
    return " ".join(text.split()).lower()


def answers_match(prediction: str, expected: str, family: str | None = None) -> bool:
    if family in {"gravity_numeric", "unit_conversion"}:
        try:
            return abs(Decimal(str(prediction)) - Decimal(str(expected))) <= Decimal("0.000001")
        except Exception:
            pass
    return normalize_answer(prediction, family) == normalize_answer(expected, family)


def answer_format_pass(answer: str) -> bool:
    if not answer:
        return False
    if any(marker in answer.lower() for marker in ("expected_answer", "gold_answer", "correct answer", "target_answer")):
        return False
    return "\n" not in answer.strip()


def compute_eval_metrics(records: list[dict], *, mode: str, adapter_path: str, decoding_params: dict, parent_records: list[dict] | None = None) -> dict:
    output_count = len(records)
    empty = sum(1 for row in records if not row.get("prediction"))
    prompt_copy = sum(1 for row in records if row.get("prediction") and str(row.get("prediction")) in str(row.get("prompt", "")) and len(str(row.get("prediction"))) > 12)
    format_pass = sum(1 for row in records if answer_format_pass(str(row.get("prediction", ""))))
    correct = sum(1 for row in records if row.get("exact_match"))
    by_family: dict[str, Counter] = defaultdict(Counter)
    for row in records:
        fam = str(row.get("family", "unknown"))
        by_family[fam]["total"] += 1
        by_family[fam]["correct"] += bool(row.get("exact_match"))
    by_family_metrics = {fam: {"count": vals["total"], "exact_match": vals["correct"] / max(1, vals["total"])} for fam, vals in sorted(by_family.items())}
    hard_fams = {"bit_manipulation", "equation_symbolic", "gravity_numeric", "unit_conversion", "cipher_text"}
    hard_rows = [row for row in records if row.get("family") in hard_fams]
    hard_correct = sum(1 for row in hard_rows if row.get("exact_match"))
    regressions = []
    improvements = []
    if parent_records:
        parent_by_id = {row.get("id"): row for row in parent_records}
        for row in records:
            parent = parent_by_id.get(row.get("id"))
            if parent and parent.get("exact_match") and not row.get("exact_match"):
                regressions.append(row.get("id"))
            if parent and not parent.get("exact_match") and row.get("exact_match"):
                improvements.append(row.get("id"))
    parent_exact = None
    if parent_records is not None:
        parent_exact = sum(1 for row in parent_records if row.get("exact_match")) / max(1, len(parent_records))
    prompt_blob = "|".join(str(row.get("prompt_hash", "")) for row in records)
    return {
        "micro_stack_test_label": MICRO_LABEL,
        "not_submission_ready": True,
        "not_095_candidate": True,
        "mode": mode,
        "row_count": output_count,
        "output_count": output_count,
        "exact_match": correct / max(1, output_count),
        "empty_output_count": empty,
        "empty_output_rate": empty / max(1, output_count),
        "prompt_copy_count": prompt_copy,
        "prompt_copy_rate": prompt_copy / max(1, output_count),
        "answer_format_pass_count": format_pass,
        "answer_format_pass_rate": format_pass / max(1, output_count),
        "exact_answer_format_pass_rate": format_pass / max(1, output_count),
        "by_family_exact_match": by_family_metrics,
        "public_like_score": correct / max(1, output_count) if "public" in mode or "smoke" in mode else 0.0,
        "family_hard_score": hard_correct / max(1, len(hard_rows)) if hard_rows else 0.0,
        "rule_holdout_score": correct / max(1, output_count) if "rule" in mode else 0.0,
        "synthetic_holdout_score": correct / max(1, output_count) if "synthetic" in mode or "adversarial" in mode else 0.0,
        "synthetic_only_gain": False,
        "regression_ids": regressions,
        "improvement_ids": improvements,
        "parent_exact_match": parent_exact,
        "child_exact_match": correct / max(1, output_count),
        "child_minus_parent_delta": None if parent_exact is None else (correct / max(1, output_count)) - parent_exact,
        "child_regressions": regressions,
        "child_improvements": improvements,
        "sample_outputs": records[:5],
        "adapter_hash": hashlib.sha256(str(adapter_path).encode()).hexdigest(),
        "prompt_hash": hashlib.sha256(prompt_blob.encode()).hexdigest(),
        "decoding_params": decoding_params,
    }


def _load_eval_rows(config: dict, mode: str) -> list[dict]:
    if mode == "parent_calibrated_64":
        calibrated = Path(str(config.get("calibrated_eval_path", "")))
        if calibrated.exists():
            rows = [json.loads(line) for line in calibrated.read_text(encoding="utf-8").splitlines() if line.strip()]
            return rows[: int(config.get("max_eval_rows", len(rows)))]
    file_by_mode = {
        "smoke_16": "corpus_anti086_micro.jsonl" if str(config.get("stage")) == "micro" else str(config.get("corpus_file", "corpus_anti086_v1.jsonl")),
        "parent_calibrated_64": Path(str(config.get("calibrated_eval_path", ""))).name or "parent_calibrated_eval.jsonl",
        "private_like_rule_holdout": "private_like_val_rule_holdout.jsonl",
        "private_like_family_hard": "private_like_val_family_hard.jsonl",
    }
    input_root = Path(str(config.get("anti086_input_root", "auto")))
    if str(input_root) == "auto":
        input_root = Path("/kaggle/working/anti086_input")
        if not input_root.exists():
            input_root = Path("artifacts/anti086")
    source = input_root / file_by_mode[mode]
    if not source.exists() and mode == "smoke_16":
        fallback_names = [str(config.get("corpus_file", "")), "winmode_micro.jsonl", "win_micro.jsonl", "corpus_anti086_micro.jsonl", "corpus_anti086_v1.jsonl"]
        for name in fallback_names:
            if name and (input_root / name).exists():
                source = input_root / name
                break
    if not source.exists():
        raise SystemExit(f"missing eval source: {source}")
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    max_rows = int(config.get("max_eval_rows", 16 if mode == "smoke_16" else len(rows)))
    return rows[:max_rows]


def _prompt_from_row(row: dict) -> str:
    if "text" in row:
        return row["text"].split("Assistant:", 1)[0] + "Assistant:"
    return f"User:\n{row.get('prompt', '')}\nAssistant:"


def _records_from_outputs(rows: list[dict], prompts: list[str], raw_outputs: list[str]) -> list[dict]:
    records = []
    for row, prompt, raw in zip(rows, prompts, raw_outputs):
        pred = extract_answer(raw)
        expected = str(row.get("answer", ""))
        family = row.get("family")
        records.append(
            {
                "id": row.get("id"),
                "family": row.get("family"),
                "prompt": prompt,
                "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
                "expected": expected,
                "raw_output": raw,
                "prediction": pred,
                "extraction_detail": {"raw": raw, "extracted": pred, "expected": expected},
                "exact_match": answers_match(pred, expected, str(family) if family else None),
            }
        )
    return records


def run_hf_fallback_eval(config: dict, rows: list[dict], prompts: list[str], *, adapter_path: str | None) -> list[dict]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["base_model_path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config["base_model_path"], torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.eval()
    raw_outputs = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, do_sample=False, max_new_tokens=int(config.get("vllm_max_tokens", 128)), pad_token_id=tokenizer.eos_token_id)
        raw_outputs.append(tokenizer.decode(output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True))
    return _records_from_outputs(rows, prompts, raw_outputs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="anti086_config_micro.yaml")
    parser.add_argument("--mode", choices=("smoke_16", "parent_calibrated_64", "private_like_rule_holdout", "private_like_family_hard"), default="smoke_16")
    args = parser.parse_args()
    runtime_patch = apply_runtime_patches()
    config = load_simple_yaml(args.config)
    if str(config.get("stage")) not in {"micro", "v1", "v1b"}:
        raise SystemExit("SPRINT-10 eval path allows micro, v1, or v1b only")
    rows = _load_eval_rows(config, args.mode)
    prompts = [_prompt_from_row(row) for row in rows]
    parent_adapter = require_real_optional_path(config.get("parent_adapter_path"), field_name="parent_adapter_path")
    child_adapter = str(config["output_adapter_dir"])
    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
    except Exception as exc:
        if not bool(config.get("hf_fallback_eval", False)):
            raise SystemExit(f"vLLM unavailable and HF fallback disabled: {exc}") from exc
        child_records = run_hf_fallback_eval(config, rows, prompts, adapter_path=child_adapter)
        parent_records = run_hf_fallback_eval(config, rows, prompts, adapter_path=parent_adapter) if parent_adapter else None
        backend = "hf_fallback"
    else:
        llm = LLM(
            model=config["base_model_path"],
            enable_lora=True,
            max_lora_rank=int(config.get("rank", 32)),
            max_model_len=int(config.get("vllm_max_model_len", 8192)),
            max_num_seqs=int(config.get("vllm_max_num_seqs", 64)),
            gpu_memory_utilization=float(config.get("vllm_gpu_memory_utilization", 0.85)),
        )
        params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=int(config.get("vllm_max_tokens", 32)))
        child_outputs = llm.generate(prompts, params, lora_request=LoRARequest("child", 1, child_adapter))
        child_records = _records_from_outputs(rows, prompts, [output.outputs[0].text for output in child_outputs])
        parent_records = None
        if parent_adapter:
            parent_outputs = llm.generate(prompts, params, lora_request=LoRARequest("parent", 2, parent_adapter))
            parent_records = _records_from_outputs(rows, prompts, [output.outputs[0].text for output in parent_outputs])
        backend = "vllm"
    out_dir = Path(str(config.get("eval_output_dir", "/kaggle/working/anti086_eval")))
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "eval_per_puzzle.jsonl").open("w", encoding="utf-8") as handle:
        for rec in child_records:
            handle.write(json.dumps(rec, sort_keys=True) + "\n")
    if parent_records is not None:
        with (out_dir / "parent_eval_per_puzzle.jsonl").open("w", encoding="utf-8") as handle:
            for rec in parent_records:
                handle.write(json.dumps(rec, sort_keys=True) + "\n")
    summary = compute_eval_metrics(child_records, mode=args.mode, adapter_path=child_adapter, decoding_params={"temperature": 0.0, "top_p": 1.0, "max_tokens": int(config.get("vllm_max_tokens", 32)), "backend": backend}, parent_records=parent_records)
    summary["runtime_patch"] = runtime_patch
    summary["parent_eval_status"] = "available" if parent_records is not None else "missing_parent_adapter"
    if str(config.get("stage")) in {"v1", "v1b"} and parent_records is None:
        summary["v1_gate_status"] = "INCONCLUSIVE_PARENT_MISSING"
    elif str(config.get("stage")) in {"v1", "v1b"} and (summary.get("parent_exact_match") or 0.0) == 0:
        summary["v1_gate_status"] = "INVALID_EVAL_PARENT_ZERO"
    elif str(config.get("stage")) in {"v1", "v1b"} and summary.get("child_exact_match", 0.0) == 0:
        summary["v1_gate_status"] = "FAIL_CHILD_ZERO"
    elif summary.get("child_minus_parent_delta") is not None and summary["child_minus_parent_delta"] < 0:
        summary["v1_gate_status"] = "FAIL_CHILD_REGRESSED"
    else:
        summary["v1_gate_status"] = "V1_EVAL_COMPLETE_NOT_PACKAGEABLE"
    (out_dir / "eval_summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
