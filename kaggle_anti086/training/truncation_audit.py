from __future__ import annotations

import argparse, json, statistics, sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.v2_corpus_io import write_json_checked
from kaggle_anti086.training.real_tokenizer_collator_audit import FallbackAuditTokenizer
from kaggle_anti086.training.sft_dataset import build_assistant_only_features, load_weighted_sft_rows
from kaggle_anti086.training.training_config_schema import load_training_config


def build_truncation_audit(config: dict[str, Any], tokenizer: Any | None = None) -> dict[str, Any]:
    tokenizer = tokenizer or FallbackAuditTokenizer()
    rows = load_weighted_sft_rows(config)
    max_seq_len = int(config.get("max_seq_len", 1024))
    lengths, rejected = [], []
    counts = {"truncated_rows": 0, "answer_truncated_count": 0, "zero_supervised_rows": 0, "missing_answer_span_count": 0}
    for item in rows:
        features = build_assistant_only_features(item.row, tokenizer, max_seq_len)
        lengths.append(len(features["input_ids"]))
        counts["truncated_rows"] += int(bool(features["truncated"]))
        counts["answer_truncated_count"] += int(bool(features["answer_truncated"]))
        counts["zero_supervised_rows"] += int(features["supervised_token_count"] == 0)
        counts["missing_answer_span_count"] += int(bool(features["missing_answer_span"]))
        if features["rejected_reason"] and len(rejected) < 20:
            rejected.append({"id": item.row.get("id"), "reason": features["rejected_reason"]})
    failures = [key for key in ("answer_truncated_count", "zero_supervised_rows", "missing_answer_span_count") if counts[key]]
    return {
        "status": "PASS" if not failures else "FAIL",
        "rows_checked": len(rows),
        **counts,
        "max_seq_len": max_seq_len,
        "token_length_p95": _pctl(lengths, 0.95),
        "token_length_max": max(lengths) if lengths else 0,
        "rejected_rows": rejected,
        "failures": failures,
    }


def _pctl(vals: list[int], q: float) -> float:
    if not vals:
        return 0
    return sorted(vals)[min(len(vals) - 1, int(round((len(vals) - 1) * q)))]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="kaggle_anti086/training/configs/v2a_base_lora.yaml")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    report = build_truncation_audit(load_training_config(args.config))
    write_json_checked(args.out, report, field_name="day8_2_truncation_audit_report")
    print(json.dumps({"status": report["status"], "out": args.out}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
