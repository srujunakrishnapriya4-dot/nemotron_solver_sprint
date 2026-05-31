from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.eval.day5_eval_factory import read_jsonl
from kaggle_anti086.eval.run_solver_eval import run_solver_eval
from kaggle_anti086.kaggle_path_safety import safe_write_text


QUALITY_GATES = {
    "private_like": {"min_behavior_accuracy": 0.65, "max_unsafe_answer_rate": 0.05, "min_correct_abstain_rate": 0.80},
    "rule_holdout": {"min_behavior_accuracy": 0.55, "max_unsafe_answer_rate": 0.05, "min_correct_abstain_rate": 0.80},
    "family_hard": {"min_behavior_accuracy": 0.50, "max_unsafe_answer_rate": 0.05, "min_correct_abstain_rate": 0.80},
    "anti_leak": {"min_behavior_accuracy": 0.55, "max_unsafe_answer_rate": 0.03, "min_correct_abstain_rate": 0.85},
}


def run_stack_benchmark(eval_path: str | Path, mode: str, out_report: str | Path, out_predictions: str | Path) -> dict:
    eval_type = _infer_eval_type(str(eval_path))
    if mode == "solver":
        gates = QUALITY_GATES.get(eval_type, QUALITY_GATES["private_like"])
        report = run_solver_eval(
            eval_path,
            out_report,
            out_predictions,
            max_unsafe_answer_rate=gates["max_unsafe_answer_rate"],
            min_correct_abstain_rate=gates["min_correct_abstain_rate"],
            min_behavior_accuracy=gates["min_behavior_accuracy"],
        )
        rows = read_jsonl(eval_path)
        report = report | {
            "benchmark_mode": "solver",
            "eval_type": eval_type,
            "by_subfamily": _by_subfamily(rows),
            "abstention_examples": [item for item in report.get("failure_examples", []) if item.get("abstained")][:20],
            "weak_family_warnings": _weak_family_warnings(report),
        }
        if report["answerable_wrong_answer_rate"] > _wrong_answer_limit(eval_type):
            report["quality_status"] = "FAIL"
            report.setdefault("weak_family_warnings", []).append("answerable_wrong_answer_rate_exceeds_day5_gate")
        safe_write_text(out_report, json.dumps(report, indent=2, sort_keys=True) + "\n", field_name="stack_benchmark_report")
        return report
    if mode == "model_plan":
        plan = _model_plan(eval_path)
        safe_write_text(out_report, json.dumps(plan, indent=2, sort_keys=True) + "\n", field_name="stack_benchmark_plan")
        safe_write_text(out_predictions, "", field_name="model_plan_predictions_placeholder")
        print(json.dumps({"execution_status": "PASS", "mode": "model_plan", "out_report": str(out_report)}, sort_keys=True))
        return plan
    if mode in {"base", "parent", "child"}:
        raise SystemExit(f"{mode} model execution is not available locally in Day 5; use --mode model_plan for Kaggle commands")
    raise SystemExit(f"unsupported benchmark mode: {mode}")


def _model_plan(eval_path: str | Path) -> dict:
    return {
        "execution_status": "PASS",
        "mode": "model_plan",
        "model_results_faked": False,
        "eval_path": str(eval_path),
        "required_next_commands": [
            "python kaggle_anti086/kaggle_eval_stage.py --config anti086_winmode_v1b.yaml --stage parent_eval",
            "python kaggle_anti086/eval/compare_solver_base_parent.py --solver-report artifacts/sprint11/day5_private_like_solver_report.json --parent-report <parent_report.json> --out artifacts/sprint11/day5_solver_base_parent_comparison.json",
        ],
        "note": "Day 5 does not run base/parent/child model inference locally.",
    }


def _infer_eval_type(path_text: str) -> str:
    for name in ("private_like", "rule_holdout", "family_hard", "anti_leak"):
        if name in path_text:
            return name
    return "private_like"


def _wrong_answer_limit(eval_type: str) -> float:
    return {"private_like": 0.10, "rule_holdout": 0.12, "family_hard": 0.15, "anti_leak": 0.15}.get(eval_type, 0.10)


def _by_subfamily(rows: list[dict]) -> dict:
    return dict(sorted(Counter(f"{row['family']}::{row['subfamily']}" for row in rows).items()))


def _weak_family_warnings(report: dict) -> list[str]:
    warnings = []
    for family, bucket in report.get("by_family", {}).items():
        if bucket.get("count", 0) >= 16 and bucket.get("exact_match", 0.0) < 0.35:
            warnings.append(f"weak_solver_family_{family}")
    return warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Day 5 stack benchmark harness.")
    parser.add_argument("--eval", required=True)
    parser.add_argument("--mode", required=True, choices=["solver", "model_plan", "base", "parent", "child"])
    parser.add_argument("--out-report", required=True)
    parser.add_argument("--out-predictions", required=True)
    args = parser.parse_args(argv)
    report = run_stack_benchmark(args.eval, args.mode, args.out_report, args.out_predictions)
    print(json.dumps({"execution_status": report.get("execution_status"), "quality_status": report.get("quality_status", "NA"), "mode": args.mode, "row_count": report.get("row_count", 0)}, sort_keys=True))
    return 0 if report.get("quality_status", "PASS") != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
