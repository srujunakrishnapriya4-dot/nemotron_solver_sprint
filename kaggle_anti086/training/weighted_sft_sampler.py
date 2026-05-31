from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import random
from typing import Any
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.v2_corpus_io import write_json_checked
from kaggle_anti086.training.sft_dataset import SFTItem, load_weighted_sft_rows
from kaggle_anti086.training.training_config_schema import load_training_config


def build_weighted_sample(items: list[SFTItem], *, seed: int = 42) -> list[SFTItem]:
    direct = [item for item in items if item.source == "direct_answer"]
    corrected = [item for item in items if item.source == "solver_corrected"]
    rng = random.Random(seed)
    corrected = corrected[:]
    rng.shuffle(corrected)
    selected = direct[:] + corrected[: len(direct) // 2]
    rng.shuffle(selected)
    return _deblock(selected)


def build_weighted_sampling_report(items: list[SFTItem], selected: list[SFTItem], *, seed: int = 42) -> dict[str, Any]:
    direct_available = sum(1 for item in items if item.source == "direct_answer")
    corrected_available = sum(1 for item in items if item.source == "solver_corrected")
    direct_selected = sum(1 for item in selected if item.source == "direct_answer")
    corrected_selected = sum(1 for item in selected if item.source == "solver_corrected")
    family_counts = Counter(str(item.row.get("family", "unknown")) for item in selected)
    failures = []
    if corrected_selected > int(direct_selected * 0.60):
        failures.append("solver_corrected_selected_gt_60pct_direct")
    if any(item.source not in {"direct_answer", "solver_corrected"} for item in selected):
        failures.append("non_sft_source_selected")
    if any(item.row.get("verification_status") != "verified" for item in selected):
        failures.append("unverified_selected")
    if any(item.row.get("family") in {"equation_operator", "sequence_pattern", "permutation_sorting", "custom_numeral", "unknown"} for item in selected):
        failures.append("unsupported_selected")
    max_family = _max_run([str(item.row.get("family", "")) for item in selected])
    max_rule = _max_run([str(item.row.get("rule_id", "")) for item in selected])
    direct_hashes = [
        hashlib.sha256(str(item.row.get("prompt", "")).encode()).hexdigest()
        for item in selected
        if item.source == "direct_answer"
    ]
    all_hashes = [hashlib.sha256(str(item.row.get("prompt", "")).encode()).hexdigest() for item in selected]
    duplicate_prompt_hash_count = len(direct_hashes) - len(set(direct_hashes))
    solver_corrected_mirror_count = max(0, len(all_hashes) - len(set(all_hashes)) - duplicate_prompt_hash_count)
    if max_family > 50:
        failures.append("same_family_block_gt_50")
    if max_rule > 5:
        failures.append("same_rule_id_block_gt_5")
    return {
        "status": "PASS" if not failures else "FAIL",
        "seed": seed,
        "direct_rows_available": direct_available,
        "solver_corrected_rows_available": corrected_available,
        "direct_rows_selected": direct_selected,
        "solver_corrected_rows_selected": corrected_selected,
        "effective_solver_corrected_weight": corrected_selected / direct_selected if direct_selected else 0.0,
        "abstain_safety_selected": 0,
        "hard_negative_selected": 0,
        "unsupported_selected": sum(1 for item in selected if item.row.get("family") in {"equation_operator", "sequence_pattern", "permutation_sorting", "custom_numeral", "unknown"}),
        "total_selected": len(selected),
        "family_counts": dict(sorted(family_counts.items())),
        "first_200_family_distribution": dict(sorted(Counter(str(item.row.get("family", "unknown")) for item in selected[:200]).items())),
        "max_consecutive_same_family": max_family,
        "max_consecutive_same_rule_id": max_rule,
        "duplicate_prompt_hash_count": duplicate_prompt_hash_count,
        "solver_corrected_mirror_prompt_count": solver_corrected_mirror_count,
        "warnings": [],
        "failures": failures,
    }


def _deblock(items: list[SFTItem]) -> list[SFTItem]:
    # Greedy round-robin by family to avoid pathological family blocks.
    buckets: dict[str, list[SFTItem]] = {}
    for item in items:
        buckets.setdefault(str(item.row.get("family", "unknown")), []).append(item)
    out: list[SFTItem] = []
    while any(buckets.values()):
        for family in sorted(buckets):
            if buckets[family]:
                out.append(buckets[family].pop(0))
    return out


def _max_run(values: list[str]) -> int:
    best = cur = 0
    prev = None
    for value in values:
        cur = cur + 1 if value == prev else 1
        prev = value
        best = max(best, cur)
    return best


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="kaggle_anti086/training/configs/v2a_base_lora.yaml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    config = load_training_config(args.config)
    items = load_weighted_sft_rows(config)
    selected = build_weighted_sample(items, seed=args.seed)
    report = build_weighted_sampling_report(items, selected, seed=args.seed)
    write_json_checked(args.out, report, field_name="day8_2_weighted_sampling_report")
    print(json.dumps({"status": report["status"], "out": args.out}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
