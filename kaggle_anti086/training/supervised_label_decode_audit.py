from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.v2_corpus_io import write_json_checked
from kaggle_anti086.solvers.answer_normalizer import answers_match
from kaggle_anti086.training.real_tokenizer_collator_audit import FallbackAuditTokenizer
from kaggle_anti086.training.sft_dataset import build_assistant_only_features, load_weighted_sft_rows
from kaggle_anti086.training.training_config_schema import load_training_config


def audit_supervised_label_decode(config: dict[str, Any], tokenizer: Any | None = None, *, limit: int = 128) -> dict[str, Any]:
    tokenizer = tokenizer or FallbackAuditTokenizer()
    rows = load_weighted_sft_rows(config)[:limit]
    mismatches = contains_prompt = contains_assistant = contains_explanation = empty = 0
    examples = []
    for item in rows:
        features = build_assistant_only_features(item.row, tokenizer, int(config.get("max_seq_len", 1024)))
        ids = [label for label in features["labels"] if label != -100]
        decoded = _decode(tokenizer, ids)
        expected = str(item.row.get("answer", ""))
        if not decoded.strip():
            empty += 1
        if not answers_match(decoded, expected, None):
            mismatches += 1
            if len(examples) < 10:
                examples.append({"id": item.row.get("id"), "expected": expected, "decoded": decoded})
        lowered = decoded.lower()
        if "assistant:" in lowered:
            contains_assistant += 1
        if "because" in lowered or "explanation" in lowered:
            contains_explanation += 1
        if str(item.row.get("prompt", ""))[:20] and str(item.row.get("prompt", ""))[:20] in decoded:
            contains_prompt += 1
    failures = []
    if mismatches:
        failures.append("decoded_supervision_mismatch")
    if contains_prompt:
        failures.append("supervised_contains_prompt")
    if contains_assistant:
        failures.append("supervised_contains_assistant_prefix")
    if contains_explanation:
        failures.append("supervised_contains_explanation")
    if empty:
        failures.append("supervised_empty")
    return {
        "status": "PASS" if not failures else "FAIL",
        "rows_checked": len(rows),
        "decoded_match_count": len(rows) - mismatches,
        "decoded_mismatch_count": mismatches,
        "contains_prompt_count": contains_prompt,
        "contains_assistant_prefix_count": contains_assistant,
        "contains_explanation_count": contains_explanation,
        "empty_supervised_count": empty,
        "examples": examples,
        "failures": failures,
    }


def _decode(tokenizer: Any, ids: list[int]) -> str:
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(ids, skip_special_tokens=True)
    # Fallback tokenizer ids are not reversible; tests use a reversible tokenizer.
    return " ".join(str(i) for i in ids)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="kaggle_anti086/training/configs/v2a_base_lora.yaml")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    report = audit_supervised_label_decode(load_training_config(args.config))
    write_json_checked(args.out, report, field_name="day8_2_supervised_label_decode_report")
    print(json.dumps({"status": report["status"], "out": args.out}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
