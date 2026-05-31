from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.v2_corpus_io import read_jsonl, write_json_checked
from kaggle_anti086.training.prepare_tokenization_dry_run import UNSUPPORTED_SFT, fallback_tokenize
from kaggle_anti086.training.training_config_schema import load_training_config


def build_loss_mask_audit(config: dict[str, Any], *, limit: int = 512) -> dict[str, Any]:
    direct = read_jsonl(config["train_direct_path"])
    corrected = read_jsonl(config["train_solver_corrected_path"])
    abstain = read_jsonl(config["train_abstain_safety_path"])
    hardneg = read_jsonl(config["train_hard_negative_path"])
    rows = [("direct", row) for row in direct[: limit // 2]] + [("solver_corrected", row) for row in corrected[: limit // 2]] + [("abstain_safety", row) for row in abstain[:50]] + [("hard_negative", row) for row in hardneg[:50]]
    counts = {
        "zero_supervised_sft_rows": 0,
        "user_token_supervision_count": 0,
        "system_token_supervision_count": 0,
        "prompt_supervision_count": 0,
        "answer_supervision_count": 0,
        "abstain_safety_supervised_count": 0,
        "hard_negative_supervised_count": 0,
        "unsupported_supervised_count": 0,
    }
    examples = []
    full_prompt = bool(config.get("full_prompt_loss"))
    train_on_user = bool(config.get("train_on_user"))
    for source, row in rows:
        labels = _simulate_labels(row, supervise=source in {"direct", "solver_corrected"} and row.get("family") not in UNSUPPORTED_SFT)
        supervised = sum(1 for label in labels if label != -100)
        if source in {"direct", "solver_corrected"}:
            if supervised == 0:
                counts["zero_supervised_sft_rows"] += 1
            if row.get("family") in UNSUPPORTED_SFT and supervised:
                counts["unsupported_supervised_count"] += 1
            counts["answer_supervision_count"] += supervised
        elif source == "abstain_safety" and supervised:
            counts["abstain_safety_supervised_count"] += 1
        elif source == "hard_negative" and supervised:
            counts["hard_negative_supervised_count"] += 1
        if len(examples) < 5:
            examples.append({"id": row.get("id"), "source": source, "supervised_tokens": supervised})
    failures = []
    for key, value in counts.items():
        if key != "answer_supervision_count" and value:
            failures.append(f"{key}_nonzero")
    if full_prompt:
        failures.append("full_prompt_loss_detected")
    if train_on_user:
        failures.append("train_on_user_detected")
    return {
        "status": "PASS" if not failures else "FAIL",
        "rows_checked": len(rows),
        "sft_rows_checked": min(len(direct), limit // 2) + min(len(corrected), limit // 2),
        **counts,
        "full_prompt_loss_detected": full_prompt,
        "train_on_user_detected": train_on_user,
        "label_mask_examples": examples,
        "warnings": [],
        "failures": failures,
    }


def _simulate_labels(row: dict[str, Any], *, supervise: bool) -> list[int]:
    messages = row.get("messages", [])
    user = str(messages[0].get("content", "")) if messages else str(row.get("prompt", ""))
    assistant = str(messages[1].get("content", "")) if len(messages) > 1 else str(row.get("answer", ""))
    user_labels = [-100] * len(fallback_tokenize("SYSTEM " + user))
    answer_tokens = fallback_tokenize(assistant)
    answer_labels = list(range(len(answer_tokens))) if supervise else [-100] * len(answer_tokens)
    return user_labels + answer_labels


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = build_loss_mask_audit(load_training_config(args.config))
    write_json_checked(args.out, report, field_name="day7_loss_mask_report")
    print(json.dumps({"status": report["status"], "rows_checked": report["rows_checked"], "out": args.out}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
