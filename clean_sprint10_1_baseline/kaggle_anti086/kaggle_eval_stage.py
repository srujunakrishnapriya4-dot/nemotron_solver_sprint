from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from kaggle_prepare_anti086_tokens import load_simple_yaml
from kaggle_path_safety import require_safe_config_output_paths, require_writable_output_dir, safe_write_text


MODE_BY_STAGE = {
    "eval_micro": "smoke_16",
    "eval_v1": "smoke_16",
    "eval_v1b": "parent_calibrated_64",
}


def validate_eval_summary(summary_path: str | Path, *, min_outputs: int = 1) -> dict:
    path = Path(summary_path)
    if not path.exists():
        raise SystemExit(f"missing real eval report: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    output_count = int(summary.get("output_count", summary.get("row_count", 0)))
    empty_rate = float(summary.get("empty_output_rate", summary.get("empty_output_count", 1) / max(1, output_count)))
    prompt_copy_rate = float(summary.get("prompt_copy_rate", 1.0))
    format_rate = float(summary.get("answer_format_pass_rate", summary.get("exact_answer_format_pass_rate", 0.0)))
    if output_count < min_outputs:
        raise SystemExit("eval produced no outputs")
    if empty_rate > 0.05:
        raise SystemExit("eval empty output rate too high")
    if prompt_copy_rate > 0.01:
        raise SystemExit("eval prompt-copy rate too high")
    if format_rate < 0.95:
        raise SystemExit("eval answer format pass rate too low")
    if summary.get("status") == "EVAL_SCRIPT_PLACEHOLDER":
        raise SystemExit("placeholder eval reports are forbidden")
    if "by_family_exact_match" not in summary:
        raise SystemExit("eval report missing by-family metrics")
    if not summary.get("sample_outputs"):
        raise SystemExit("eval report missing sample raw outputs")
    per_puzzle = path.parent / "eval_per_puzzle.jsonl"
    if not per_puzzle.exists() or per_puzzle.stat().st_size == 0:
        raise SystemExit("eval report missing raw per-puzzle outputs")
    if "eval_v1" in str(path) or "eval_v1b" in str(path) or summary.get("mode") == "parent_calibrated_64":
        if summary.get("parent_eval_status") == "missing_parent_adapter":
            raise SystemExit("parent-vs-child eval is inconclusive because parent adapter is missing")
        if summary.get("parent_exact_match") == 0:
            raise SystemExit("invalid eval: parent_exact_match is zero")
        if summary.get("child_exact_match") == 0:
            raise SystemExit("child exact_match is zero")
    return summary


def write_stage_gate(stage: str, payload: dict) -> None:
    gate_dir = require_writable_output_dir("/kaggle/working/anti086_stage_logs", field_name="stage_log_dir")
    gate_dir.mkdir(parents=True, exist_ok=True)
    gate = {"stage": stage, "decision": "PASS", "reason": "eval stage completed", **payload}
    safe_write_text(gate_dir / f"{stage}.gate.json", json.dumps(gate, sort_keys=True, indent=2), field_name="stage_log_dir")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", default="eval_micro", choices=sorted(MODE_BY_STAGE))
    args = parser.parse_args()
    config = load_simple_yaml(args.config)
    require_safe_config_output_paths(config, stage=str(config.get("stage", "eval_stage")))
    if str(config.get("stage")) not in {"micro", "v1", "v1b"}:
        raise SystemExit("SPRINT-10 permits eval_micro/eval_v1/eval_v1b only")
    mode = str(config.get("eval_mode", MODE_BY_STAGE[args.stage]))
    subprocess.check_call([sys.executable, "kaggle_eval_anti086_vllm.py", "--config", args.config, "--mode", mode])
    summary_path = Path(str(config.get("eval_output_dir", "/kaggle/working/anti086_eval"))) / "eval_summary.json"
    summary = validate_eval_summary(summary_path)
    write_stage_gate(args.stage, summary)
    print(json.dumps(summary, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
