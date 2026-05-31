from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from kaggle_anti086.data.v2_corpus_io import read_jsonl
from kaggle_anti086.solvers.answer_normalizer import answers_match, normalize_gold


def load_eval_rows(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def generate_answer(model, tokenizer, prompt: str, generation_config: dict[str, Any]) -> str:
    import torch  # type: ignore

    inputs = tokenizer(prompt, return_tensors="pt")
    if hasattr(model, "device"):
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.no_grad():
        output = model.generate(**inputs, **generation_config)
    decoded = tokenizer.decode(output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
    return decoded.strip()


def run_model_eval(model, tokenizer, eval_path: str | Path, *, generator: Callable[[str, str], str] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = load_eval_rows(eval_path)
    by_family: dict[str, Counter[str]] = defaultdict(Counter)
    predictions = []
    wrong_examples = []
    verbose = 0
    empty = 0
    correct = 0
    for row in rows:
        prompt = str(row.get("prompt", ""))
        family = str(row.get("family", "unknown"))
        answer_type = _answer_type(family)
        if generator is not None:
            pred = generator(prompt, family)
        else:
            pred = generate_answer(model, tokenizer, _render_eval_prompt(prompt), _generation_config(family))
        if not pred.strip():
            empty += 1
        if _is_verbose(pred):
            verbose += 1
        expected = str(row.get("answer", ""))
        ok = answers_match(pred, expected, answer_type)
        correct += int(ok)
        by_family[family]["count"] += 1
        by_family[family]["correct"] += int(ok)
        pred_row = {
            "row_id": row.get("id"),
            "family": family,
            "expected": expected,
            "prediction": pred,
            "normalized_expected": normalize_gold(expected, answer_type),
            "normalized_prediction": normalize_gold(pred, answer_type),
            "correct": ok,
        }
        predictions.append(pred_row)
        if not ok and len(wrong_examples) < 20:
            wrong_examples.append(pred_row)
    report = {
        "row_count": len(rows),
        "exact_match": correct / len(rows) if rows else 0.0,
        "by_family": {
            family: {"count": counts["count"], "correct": counts["correct"], "exact_match": counts["correct"] / counts["count"] if counts["count"] else 0.0}
            for family, counts in sorted(by_family.items())
        },
        "wrong_examples": wrong_examples,
        "verbose_output_count": verbose,
        "empty_output_count": empty,
        "failures": [],
    }
    if rows and empty / len(rows) > 0.05:
        report["failures"].append("empty_output_rate_gt_0_05")
    if rows and verbose / len(rows) > 0.10:
        report["failures"].append("verbose_output_rate_gt_0_10")
    return report, predictions


def compare_base_vs_adapter(base_report: dict[str, Any], adapter_report: dict[str, Any]) -> dict[str, Any]:
    delta = float(adapter_report.get("exact_match", 0.0)) - float(base_report.get("exact_match", 0.0))
    family_deltas = {}
    regressions = []
    for family in sorted(set(base_report.get("by_family", {})) | set(adapter_report.get("by_family", {}))):
        base_score = float(base_report.get("by_family", {}).get(family, {}).get("exact_match", 0.0))
        adapter_score = float(adapter_report.get("by_family", {}).get(family, {}).get("exact_match", 0.0))
        family_deltas[family] = adapter_score - base_score
        if adapter_score - base_score < -0.05:
            regressions.append(family)
    return {"base_exact": base_report.get("exact_match", 0.0), "v2a_exact": adapter_report.get("exact_match", 0.0), "delta": delta, "family_deltas": family_deltas, "regressions": regressions}


def _render_eval_prompt(prompt: str) -> str:
    return f"You are given a task. Respond with only the final answer.\n\n{prompt}\n\nAnswer:"


def _generation_config(family: str) -> dict[str, Any]:
    max_tokens = 64 if family in {"word_cipher", "char_cipher", "text_phrase"} else 32
    return {"do_sample": False, "temperature": 0.0, "top_p": 1.0, "max_new_tokens": max_tokens}


def _answer_type(family: str) -> str | None:
    if family in {"numeric_formula", "gravity_numeric", "unit_conversion"}:
        return "numeric"
    if family == "bit_manipulation":
        return "binary"
    if family == "roman_numeral":
        return "roman"
    if family in {"symbol_mapping", "format_only"}:
        return "symbol"
    return None


def _is_verbose(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in ("because", "therefore", "explanation", "the answer is"))
