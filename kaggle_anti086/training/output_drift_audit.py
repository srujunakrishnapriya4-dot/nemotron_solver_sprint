from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.v2_corpus_io import read_jsonl, write_json_checked


def audit_output_drift(predictions: list[dict], *, exact_match_delta: float = 0.0) -> dict:
    total = len(predictions)
    if not total:
        return {"status": "FAIL", "output_changed_rate": 0.0, "failures": ["no_predictions"]}
    changed = sum(1 for row in predictions if row.get("base_pred") != row.get("v2a_pred"))
    verbose = sum(1 for row in predictions if _verbose(str(row.get("v2a_pred", ""))))
    empty = sum(1 for row in predictions if not str(row.get("v2a_pred", "")).strip())
    copied = sum(1 for row in predictions if str(row.get("prompt", "")) and str(row.get("prompt", "")) in str(row.get("v2a_pred", "")))
    prefix = sum(1 for row in predictions if str(row.get("v2a_pred", "")).lower().startswith("answer:"))
    failures, warnings = [], []
    if changed / total < 0.01:
        warnings.append("output_changed_rate_below_0_01")
    if verbose / total > 0.10:
        failures.append("verbose_output_rate_gt_0_10")
    if empty / total > 0.05:
        failures.append("empty_output_rate_gt_0_05")
    if copied / total > 0.01:
        failures.append("copied_prompt_rate_gt_0_01")
    if exact_match_delta < -0.02:
        failures.append("exact_match_delta_negative")
    return {
        "status": "PASS" if not failures else "FAIL",
        "output_changed_rate": changed / total,
        "verbose_output_rate": verbose / total,
        "empty_output_rate": empty / total,
        "answer_prefix_rate": prefix / total,
        "copied_prompt_rate": copied / total,
        "contains_explanation_rate": verbose / total,
        "exact_match_delta": exact_match_delta,
        "warnings": warnings,
        "failures": failures,
    }


def _verbose(text: str) -> bool:
    lowered = text.lower()
    return any(x in lowered for x in ("because", "therefore", "explanation"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="artifacts/sprint11/day8_2_v2a_smoke_eval_predictions.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--exact-match-delta", type=float, default=0.0)
    args = parser.parse_args(argv)
    predictions = read_jsonl(args.predictions) if Path(args.predictions).exists() else []
    report = audit_output_drift(predictions, exact_match_delta=args.exact_match_delta)
    write_json_checked(args.out, report, field_name="day8_2_output_drift_report")
    print(json.dumps({"status": report["status"], "out": args.out}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
