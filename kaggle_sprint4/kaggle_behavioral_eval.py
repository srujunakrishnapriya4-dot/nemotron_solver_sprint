from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Iterable


BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


class BehavioralEvalError(ValueError):
    pass


@dataclass(frozen=True)
class BehavioralGateConfig:
    min_child_exact_match_to_package: float = 0.0
    max_allowed_regression_vs_parent: float = 0.0
    max_family_regression: float = 0.10


def extract_final_answer(text: str) -> str:
    matches = BOXED_RE.findall(text)
    if matches:
        return matches[-1].strip()
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()


def gold_answer_from_row(row: dict) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise BehavioralEvalError("validation row missing assistant message")
    return extract_final_answer(str(messages[1].get("content", "")))


def load_validation_rows(path: str | Path, *, max_rows: int = 0) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise BehavioralEvalError("validation JSONL row is not an object")
                rows.append(payload)
                if max_rows and len(rows) >= max_rows:
                    break
    return rows


def score_predictions(rows: Iterable[dict], predictions: dict[str, str]) -> dict:
    total = 0
    correct = 0
    by_family_total: dict[str, int] = defaultdict(int)
    by_family_correct: dict[str, int] = defaultdict(int)
    wrong: list[dict] = []
    for row in rows:
        problem_id = str(row.get("problem_id"))
        family = str(row.get("family", "unknown"))
        gold = gold_answer_from_row(row)
        predicted = extract_final_answer(str(predictions.get(problem_id, "")))
        total += 1
        by_family_total[family] += 1
        if predicted == gold:
            correct += 1
            by_family_correct[family] += 1
        else:
            wrong.append({"problem_id": problem_id, "family": family, "gold": gold, "prediction": predicted})
    return {
        "exact_match": correct / total if total else 0.0,
        "by_family": {family: by_family_correct[family] / count for family, count in sorted(by_family_total.items())},
        "wrong": wrong,
    }


def compare_parent_child(rows: list[dict], parent_predictions: dict[str, str], child_predictions: dict[str, str]) -> dict:
    parent = score_predictions(rows, parent_predictions)
    child = score_predictions(rows, child_predictions)
    families = sorted(set(parent["by_family"]) | set(child["by_family"]))
    by_family = {
        family: {
            "parent": parent["by_family"].get(family, 0.0),
            "child": child["by_family"].get(family, 0.0),
            "delta": child["by_family"].get(family, 0.0) - parent["by_family"].get(family, 0.0),
        }
        for family in families
    }
    parent_wrong = {item["problem_id"]: item for item in parent["wrong"]}
    child_wrong = {item["problem_id"]: item for item in child["wrong"]}
    return {
        "parent_exact_match": parent["exact_match"],
        "child_exact_match": child["exact_match"],
        "delta": child["exact_match"] - parent["exact_match"],
        "by_family": by_family,
        "child_regressed": [child_wrong[key] for key in sorted(child_wrong) if key not in parent_wrong],
        "child_improved": [parent_wrong[key] for key in sorted(parent_wrong) if key not in child_wrong],
    }


def validate_behavioral_report(report: dict, *, gate: BehavioralGateConfig | None = None) -> bool:
    gate = gate or BehavioralGateConfig()
    required = ("parent_exact_match", "child_exact_match", "delta", "by_family")
    for key in required:
        if key not in report:
            raise BehavioralEvalError(f"behavioral report missing {key}")
    parent = float(report["parent_exact_match"])
    child = float(report["child_exact_match"])
    if child < gate.min_child_exact_match_to_package:
        raise BehavioralEvalError(f"child exact match below packaging minimum: child={child}, minimum={gate.min_child_exact_match_to_package}")
    if child < parent - gate.max_allowed_regression_vs_parent:
        raise BehavioralEvalError(f"child regressed vs parent: parent={parent}, child={child}")
    by_family = report.get("by_family")
    if not isinstance(by_family, dict):
        raise BehavioralEvalError("by_family must be an object")
    for family, payload in by_family.items():
        if not isinstance(payload, dict):
            raise BehavioralEvalError(f"family payload must be object: {family}")
        delta = float(payload.get("delta", 0.0))
        if delta < -gate.max_family_regression:
            raise BehavioralEvalError(f"catastrophic family regression for {family}: {delta}")
    return True


def load_report(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise BehavioralEvalError("behavioral report must be a JSON object")
    return payload


def generate_adapter_predictions(
    rows: list[dict],
    *,
    base_model_path: str,
    adapter_path: str,
    max_new_tokens: int = 64,
) -> dict[str, str]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.eval()
    predictions: dict[str, str] = {}
    for row in rows:
        prompt = str(row.get("messages", [{}])[0].get("content", ""))
        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, "apply_chat_template"):
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            rendered = f"USER: {prompt}\nASSISTANT: "
        encoded = tokenizer(rendered, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated_ids = output[0][encoded["input_ids"].shape[-1] :]
        predictions[str(row.get("problem_id"))] = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-file", required=True)
    parser.add_argument("--base-model-path")
    parser.add_argument("--parent-adapter-path")
    parser.add_argument("--child-adapter-path")
    parser.add_argument("--parent-predictions-json", help="Lightweight mode: problem_id -> response text")
    parser.add_argument("--child-predictions-json", help="Lightweight mode: problem_id -> response text")
    parser.add_argument("--predictions-output-json", help="For --mode parent/child, where to write generated predictions")
    parser.add_argument("--output-report-path", default="/kaggle/working/sprint4_behavioral_eval.json")
    parser.add_argument("--max-eval-rows", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-allowed-regression-vs-parent", type=float, default=0.0)
    parser.add_argument("--min-child-exact-match-to-package", type=float, default=0.0)
    parser.add_argument("--max-family-regression", type=float, default=0.10)
    parser.add_argument("--mode", choices=("compare", "parent", "child"), default="compare")
    args = parser.parse_args()
    rows = load_validation_rows(args.validation_file, max_rows=args.max_eval_rows)
    if args.mode in {"parent", "child"}:
        adapter_path = args.parent_adapter_path if args.mode == "parent" else args.child_adapter_path
        if not args.base_model_path or not adapter_path or not args.predictions_output_json:
            raise SystemExit(f"--mode {args.mode} requires --base-model-path, --{args.mode}-adapter-path, and --predictions-output-json")
        predictions = generate_adapter_predictions(
            rows,
            base_model_path=args.base_model_path,
            adapter_path=adapter_path,
            max_new_tokens=args.max_new_tokens,
        )
        Path(args.predictions_output_json).write_text(json.dumps(predictions, sort_keys=True, indent=2), encoding="utf-8")
        print(f"{args.mode}_predictions_json={args.predictions_output_json}")
        return
    if not args.parent_predictions_json or not args.child_predictions_json:
        raise SystemExit("compare mode requires --parent-predictions-json and --child-predictions-json. Use --mode parent/child to generate them.")
    parent_predictions = json.loads(Path(args.parent_predictions_json).read_text(encoding="utf-8"))
    child_predictions = json.loads(Path(args.child_predictions_json).read_text(encoding="utf-8"))
    report = compare_parent_child(rows, parent_predictions, child_predictions)
    validate_behavioral_report(
        report,
        gate=BehavioralGateConfig(args.min_child_exact_match_to_package, args.max_allowed_regression_vs_parent, args.max_family_regression),
    )
    Path(args.output_report_path).write_text(json.dumps(report, sort_keys=True, indent=2), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
