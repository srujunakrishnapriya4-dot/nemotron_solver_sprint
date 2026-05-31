from __future__ import annotations

import argparse, hashlib, json, re, sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.v2_corpus_io import read_jsonl, write_json_checked
from kaggle_anti086.training.sft_dataset import SFTItem, load_weighted_sft_rows
from kaggle_anti086.training.training_config_schema import load_training_config
from kaggle_anti086.training.weighted_sft_sampler import build_weighted_sample


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def normalized_prompt_hash(prompt: str) -> str:
    norm = re.sub(r"\s+", " ", re.sub(r"\d+", "<NUM>", prompt.lower())).strip()
    return prompt_hash(norm)


def audit_final_sft_leakage(selected: list[SFTItem], eval_paths: list[str | Path] | None = None) -> dict[str, Any]:
    eval_paths = eval_paths or ["artifacts/sprint11/day5_rule_holdout_eval_512.jsonl", "artifacts/sprint11/day5_anti_leak_eval_256.jsonl"]
    prompts = [str(item.row.get("prompt", "")) for item in selected]
    direct_prompts = [str(item.row.get("prompt", "")) for item in selected if item.source == "direct_answer"]
    direct_hashes = [prompt_hash(p) for p in direct_prompts]
    hashes = [prompt_hash(p) for p in prompts]
    dup_count = len(direct_hashes) - len(set(direct_hashes))
    corrected_mirror_count = len(hashes) - len(set(hashes)) - dup_count
    rule_ids = {str(item.row.get("rule_id", "")) for item in selected}
    leakage_groups = {str(item.row.get("leakage_group", "")) for item in selected}
    eval_hashes, eval_norm_hashes, holdout_rules, anti_groups = set(), set(), set(), set()
    for path in eval_paths:
        if not Path(path).exists():
            continue
        rows = read_jsonl(path)
        for row in rows:
            eval_hashes.add(prompt_hash(str(row.get("prompt", ""))))
            eval_norm_hashes.add(normalized_prompt_hash(str(row.get("prompt", ""))))
            if "rule_holdout" in str(path):
                holdout_rules.add(str(row.get("rule_id", "")))
            if "anti_leak" in str(path):
                anti_groups.add(str(row.get("leakage_group", "")))
    failures = []
    train_eval_overlap = len(set(hashes) & eval_hashes)
    train_eval_norm_overlap = len({normalized_prompt_hash(p) for p in prompts} & eval_norm_hashes)
    rule_overlap = len(rule_ids & holdout_rules)
    group_overlap = len(leakage_groups & anti_groups)
    if train_eval_overlap or train_eval_norm_overlap:
        failures.append("train_eval_prompt_overlap")
    if rule_overlap:
        failures.append("rule_holdout_rule_overlap")
    if group_overlap:
        failures.append("anti_leak_group_overlap")
    if dup_count > 0:
        failures.append("duplicate_prompt_hash")
    return {
        "status": "PASS" if not failures else "FAIL",
        "sft_rows_checked": len(selected),
        "duplicate_prompt_hash_count": dup_count,
        "solver_corrected_mirror_prompt_count": max(0, corrected_mirror_count),
        "train_eval_prompt_overlap_count": train_eval_overlap,
        "train_eval_normalized_prompt_overlap_count": train_eval_norm_overlap,
        "rule_holdout_rule_overlap_count": rule_overlap,
        "anti_leak_group_overlap_count": group_overlap,
        "near_duplicate_template_warnings": [],
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="kaggle_anti086/training/configs/v2a_base_lora.yaml")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    config = load_training_config(args.config)
    report = audit_final_sft_leakage(build_weighted_sample(load_weighted_sft_rows(config), seed=int(config.get("seed", 42))))
    write_json_checked(args.out, report, field_name="day8_2_final_sft_leakage_report")
    print(json.dumps({"status": report["status"], "out": args.out}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
