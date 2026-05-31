from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.v2_corpus_io import read_json, write_json_checked


def build_training_log_history(trainer_state_or_summary: dict[str, Any]) -> dict[str, Any]:
    history = list(trainer_state_or_summary.get("log_history", []))
    if not history and "loss_start" in trainer_state_or_summary:
        history = [
            {"step": 0, "loss": trainer_state_or_summary.get("loss_start")},
            {"step": trainer_state_or_summary.get("steps_completed", 0), "loss": trainer_state_or_summary.get("loss_end")},
        ]
    losses = [float(row["loss"]) for row in history if row.get("loss") is not None]
    lrs = [row.get("learning_rate") for row in history if row.get("learning_rate") is not None]
    grad_norms = [row.get("grad_norm") for row in history if row.get("grad_norm") is not None]
    nan = any(math.isnan(loss) for loss in losses)
    inf = any(math.isinf(loss) for loss in losses)
    return {
        "status": "FAIL" if nan or inf else "PASS",
        "log_history": history,
        "losses": losses,
        "learning_rates": lrs,
        "grad_norms": grad_norms,
        "runtime_summary": trainer_state_or_summary.get("runtime_summary", {}),
        "loss_start": losses[0] if losses else None,
        "loss_end": losses[-1] if losses else None,
        "loss_min": min(losses) if losses else None,
        "loss_max": max(losses) if losses else None,
        "loss_nan_detected": nan,
        "loss_inf_detected": inf,
        "steps_completed": int(trainer_state_or_summary.get("steps_completed", len(losses)) or 0),
        "failures": [x for x, flag in (("nan_loss", nan), ("inf_loss", inf)) if flag],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    source = read_json(args.summary) if args.summary and Path(args.summary).exists() else {"log_history": []}
    report = build_training_log_history(source)
    write_json_checked(args.out, report, field_name="training_log_history")
    print(json.dumps({"status": report["status"], "out": args.out}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
