from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.v2_corpus_io import read_json, write_json_checked, write_jsonl_checked
from kaggle_anti086.training.inference_eval_backend import compare_base_vs_adapter, run_model_eval
from kaggle_anti086.training.model_loader import load_adapter_model, load_base_model, load_tokenizer


def compare_reports(base: dict[str, Any], v2a: dict[str, Any]) -> dict[str, Any]:
    evals = {}
    regressions = []
    for name in sorted(set(base) | set(v2a)):
        b = float(base.get(name, {}).get("exact_match", 0.0))
        v = float(v2a.get(name, {}).get("exact_match", 0.0))
        delta = v - b
        evals[name] = {"base_exact": b, "v2a_exact": v, "delta": delta}
        if delta < -0.02:
            regressions.append(name)
    return {"evals": evals, "regressions": regressions}


DEFAULT_EVALS = {
    "private_like_answerable_512": "artifacts/sprint11/day5_private_like_answerable_512.jsonl",
    "rule_holdout_answerable_512": "artifacts/sprint11/day5_rule_holdout_answerable_512.jsonl",
    "family_hard_answerable_512": "artifacts/sprint11/day5_family_hard_answerable_512.jsonl",
}


def build_post_train_eval(adapter_dir: str | Path, *, base_report_path: str | Path | None = None, v2a_report_path: str | Path | None = None) -> dict[str, Any]:
    adapter = Path(adapter_dir)
    if base_report_path and v2a_report_path and Path(base_report_path).exists() and Path(v2a_report_path).exists():
        comparison = compare_reports(read_json(base_report_path), read_json(v2a_report_path))
        status = "PASS" if not comparison["regressions"] else "FAIL"
        return {
            "status": status,
            "adapter_dir": str(adapter),
            "evals": comparison["evals"],
            "by_family": {},
            "regressions": comparison["regressions"],
            "catastrophic_families": comparison["regressions"],
            "model_eval_completed": True,
            "warnings": [],
            "failures": [] if status == "PASS" else ["candidate_regression"],
        }
    return {
        "status": "NEEDS_KAGGLE_MODEL_EVAL",
        "adapter_dir": str(adapter),
        "evals": {},
        "by_family": {},
        "regressions": [],
        "catastrophic_families": [],
        "model_eval_completed": False,
        "warnings": ["local_model_inference_unavailable"],
        "failures": [],
        "required_kaggle_command": "python kaggle_anti086/training/post_train_eval.py --adapter-dir /kaggle/working/anti086_adapters/<v2a_run> --out-report artifacts/sprint11/day8_v2a_eval_report.json --out-predictions artifacts/sprint11/day8_v2a_eval_predictions.jsonl",
    }


def run_real_post_train_eval(base_model_path: str, adapter_dir: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    adapter = Path(adapter_dir)
    failures = []
    if not (adapter / "adapter_config.json").exists() or not (adapter / "adapter_model.safetensors").exists():
        failures.append("adapter_files_missing")
    tokenizer = load_tokenizer(base_model_path)
    base_model = load_base_model(base_model_path, load_in_4bit=False, bf16=True)
    adapter_model = load_adapter_model(base_model_path, str(adapter), load_in_4bit=False, bf16=True)
    evals: dict[str, Any] = {}
    all_predictions = []
    regressions = []
    catastrophic = []
    by_family: dict[str, Any] = {}
    for name, path in DEFAULT_EVALS.items():
        if not Path(path).exists():
            failures.append(f"missing_eval:{name}")
            continue
        base_report, base_predictions = run_model_eval(base_model, tokenizer, path)
        adapter_report, adapter_predictions = run_model_eval(adapter_model, tokenizer, path)
        delta = compare_base_vs_adapter(base_report, adapter_report)
        evals[name] = delta
        if delta["delta"] < -0.02:
            regressions.append(name)
        for family, family_delta in delta["family_deltas"].items():
            by_family.setdefault(family, {})[name] = family_delta
            if family_delta < -0.05:
                catastrophic.append(family)
        for base_pred, v2a_pred in zip(base_predictions, adapter_predictions):
            all_predictions.append(
                {
                    "row_id": base_pred["row_id"],
                    "family": base_pred["family"],
                    "expected": base_pred["expected"],
                    "base_pred": base_pred["prediction"],
                    "v2a_pred": v2a_pred["prediction"],
                    "base_correct": base_pred["correct"],
                    "v2a_correct": v2a_pred["correct"],
                }
            )
    if regressions:
        failures.append("major_eval_regression")
    if catastrophic:
        failures.append("family_regression")
    return (
        {
            "status": "PASS" if not failures else "FAIL",
            "model_eval_completed": not failures,
            "adapter_dir": str(adapter),
            "evals": evals,
            "by_family": by_family,
            "regressions": regressions,
            "catastrophic_families": sorted(set(catastrophic)),
            "warnings": [],
            "failures": failures,
        },
        all_predictions,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--base-model-path")
    parser.add_argument("--out-report", required=True)
    parser.add_argument("--out-predictions", required=True)
    parser.add_argument("--base-report")
    parser.add_argument("--v2a-report")
    parser.add_argument("--kaggle-mode", action="store_true")
    args = parser.parse_args(argv)
    if args.kaggle_mode:
        if not args.base_model_path:
            raise SystemExit("--base-model-path is required in --kaggle-mode")
        report, predictions = run_real_post_train_eval(args.base_model_path, args.adapter_dir)
    else:
        report = build_post_train_eval(args.adapter_dir, base_report_path=args.base_report, v2a_report_path=args.v2a_report)
        predictions = []
    write_json_checked(args.out_report, report, field_name="day8_v2a_eval_report")
    write_jsonl_checked(args.out_predictions, predictions, field_name="day8_v2a_eval_predictions")
    print(json.dumps({"status": report["status"], "out": args.out_report}, sort_keys=True))
    return 0 if report["status"] in {"PASS", "NEEDS_KAGGLE_MODEL_EVAL"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
