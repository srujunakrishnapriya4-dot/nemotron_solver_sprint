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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--out-report", required=True)
    parser.add_argument("--out-predictions", required=True)
    parser.add_argument("--base-report")
    parser.add_argument("--v2a-report")
    args = parser.parse_args(argv)
    report = build_post_train_eval(args.adapter_dir, base_report_path=args.base_report, v2a_report_path=args.v2a_report)
    write_json_checked(args.out_report, report, field_name="day8_v2a_eval_report")
    write_jsonl_checked(args.out_predictions, [], field_name="day8_v2a_eval_predictions")
    print(json.dumps({"status": report["status"], "out": args.out_report}, sort_keys=True))
    return 0 if report["status"] in {"PASS", "NEEDS_KAGGLE_MODEL_EVAL"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
