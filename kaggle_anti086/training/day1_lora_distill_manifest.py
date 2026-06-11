from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kaggle_anti086.training.day1_verified_data_schema import read_jsonl


def build_lora_distill_manifest(
    artifacts_dir: Path,
    *,
    teacher_quality_path: Path | None = None,
    out: Path | None = None,
) -> dict[str, Any]:
    teacher_quality_path = teacher_quality_path or artifacts_dir / "day1_teacher_quality_report.json"
    input_files = {
        "sft_train": str(artifacts_dir / "phase4_verified_sft_train.jsonl"),
        "sft_eval": str(artifacts_dir / "phase4_verified_sft_eval.jsonl"),
        "sft_probe": str(artifacts_dir / "phase4_verified_eval_probe.jsonl"),
        "dpo_train": str(artifacts_dir / "phase5_dpo_train_pairs.jsonl"),
        "dpo_eval": str(artifacts_dir / "phase5_dpo_eval_pairs.jsonl"),
        "phase4_manifest": str(artifacts_dir / "phase4_verified_data_manifest.json"),
        "phase5_manifest": str(artifacts_dir / "phase5_dpo_pair_manifest.json"),
        "teacher_quality": str(teacher_quality_path),
    }
    missing = [path for path in input_files.values() if not Path(path).exists()]
    if missing:
        manifest = _base_manifest(input_files)
        manifest["status"] = "FAIL"
        manifest["missing_artifacts"] = missing
        _maybe_write(out, manifest)
        return manifest

    train = read_jsonl(Path(input_files["sft_train"]))
    eval_rows = read_jsonl(Path(input_files["sft_eval"]))
    probe = read_jsonl(Path(input_files["sft_probe"]))
    dpo_train = read_jsonl(Path(input_files["dpo_train"]))
    dpo_eval = read_jsonl(Path(input_files["dpo_eval"]))
    phase4 = _read_json(Path(input_files["phase4_manifest"]))
    phase5 = _read_json(Path(input_files["phase5_manifest"]))
    teacher_quality = _read_json(Path(input_files["teacher_quality"]))

    all_rows = [*train, *eval_rows, *probe]
    duplicate_prompt_count, split_leakage_count = _prompt_issues(train, eval_rows, probe)
    rows_by_family = Counter(str(row.get("family")) for row in all_rows)
    rows_by_difficulty = Counter(str(row.get("difficulty")) for row in all_rows)
    rows_by_prompt_style = Counter(str(row.get("prompt_style")) for row in all_rows)
    format_error_count = int(teacher_quality.get("format_error_count", 0))
    verification_fail_count = int(teacher_quality.get("verification_fail_count", 0))
    ambiguity_accepted_count = int(teacher_quality.get("ambiguity_accepted", 0))
    abstain_accepted_count = int(teacher_quality.get("abstain_placeholders_accepted", 0))
    gates = {
        "teacher_quality_pass": teacher_quality.get("status") in {"PASS", "WARN"} and teacher_quality.get("teacher_ready_for_lora") is True,
        "phase4_manifest_pass": phase4.get("status") in {"PASS", "WARN"} and phase4.get("gates", {}).get("phase4_ready_for_phase5") is True,
        "phase5_manifest_pass": phase5.get("status") in {"PASS", "WARN"} and phase5.get("gates", {}).get("dpo_pairs_ready_for_phase6") is True,
        "zero_format_errors": format_error_count == 0,
        "zero_verification_failures": verification_fail_count == 0,
        "zero_ambiguity_accepted": ambiguity_accepted_count == 0,
        "zero_abstain_accepted": abstain_accepted_count == 0,
        "zero_duplicate_prompts": duplicate_prompt_count == 0,
        "zero_split_leakage": split_leakage_count == 0,
        "enough_families": len(rows_by_family) >= 6,
        "enough_sft_rows": len(train) >= 10000 and len(eval_rows) >= 1000,
        "enough_dpo_pairs": len(dpo_train) >= 3000,
        "training_authorized": False,
        "package_authorized": False,
        "submission_authorized": False,
    }
    hard_gates = [value for key, value in gates.items() if key not in {"training_authorized", "package_authorized", "submission_authorized"}]
    training_authorized = all(hard_gates)
    gates["training_authorized"] = training_authorized
    manifest = _base_manifest(input_files)
    manifest.update(
        {
            "status": "PASS" if training_authorized else "FAIL",
            "total_rows": len(all_rows),
            "train_rows": len(train),
            "eval_rows": len(eval_rows),
            "probe_rows": len(probe),
            "rows_by_family": dict(sorted(rows_by_family.items())),
            "rows_by_difficulty": dict(sorted(rows_by_difficulty.items())),
            "rows_by_prompt_style": dict(sorted(rows_by_prompt_style.items())),
            "dpo_train_pair_count": len(dpo_train),
            "dpo_eval_pair_count": len(dpo_eval),
            "format_error_count": format_error_count,
            "verification_fail_count": verification_fail_count,
            "ambiguity_reject_count": 0,
            "ambiguity_accepted_count": ambiguity_accepted_count,
            "abstain_placeholder_reject_count": 0,
            "abstain_placeholder_accepted_count": abstain_accepted_count,
            "duplicate_prompt_count": duplicate_prompt_count,
            "split_leakage_count": split_leakage_count,
            "family_balance_status": phase4.get("audits", {}).get("family_balance", "FAIL"),
            "prompt_diversity_status": phase4.get("audits", {}).get("prompt_diversity", "FAIL"),
            "dedup_leakage_status": phase4.get("audits", {}).get("dedup_leakage", "FAIL"),
            "format_audit_status": phase4.get("audits", {}).get("format", "FAIL"),
            "teacher_verification_status": phase4.get("audits", {}).get("teacher_verification", "FAIL"),
            "dpo_pair_audit_status": phase5.get("status", "FAIL"),
            "gates": gates,
            "training_authorized": training_authorized,
            "package_authorized": False,
            "submission_authorized": False,
            "leaderboard_claim": False,
            "no_0_93_evidence": True,
            "no_0_95_evidence": True,
        }
    )
    _maybe_write(out, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Day1 LoRA distillation manifest.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/sprint11"))
    args = parser.parse_args(argv)
    manifest = build_lora_distill_manifest(args.artifacts_dir, out=args.out)
    print(json.dumps({"status": manifest["status"], "training_authorized": manifest["training_authorized"]}, sort_keys=True))
    return 0 if manifest["status"] in {"PASS", "WARN"} else 1


def _base_manifest(input_files: dict[str, str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_by": "DAY1_LORA_DISTILL_MANIFEST",
        "status": "FAIL",
        "input_files": input_files,
        "output_role": "training_manifest_only_no_training",
        "total_rows": 0,
        "train_rows": 0,
        "eval_rows": 0,
        "probe_rows": 0,
        "rows_by_family": {},
        "rows_by_difficulty": {},
        "rows_by_prompt_style": {},
        "dpo_train_pair_count": 0,
        "dpo_eval_pair_count": 0,
        "format_error_count": 0,
        "verification_fail_count": 0,
        "ambiguity_reject_count": 0,
        "ambiguity_accepted_count": 0,
        "abstain_placeholder_reject_count": 0,
        "abstain_placeholder_accepted_count": 0,
        "duplicate_prompt_count": 0,
        "split_leakage_count": 0,
        "family_balance_status": "FAIL",
        "prompt_diversity_status": "FAIL",
        "dedup_leakage_status": "FAIL",
        "format_audit_status": "FAIL",
        "teacher_verification_status": "FAIL",
        "dpo_pair_audit_status": "FAIL",
        "recommended_training_plan": {
            "stage_1": "SFT",
            "stage_2": "optional_DPO_or_rejection_tuning",
            "sft_train_file": input_files.get("sft_train"),
            "sft_eval_file": input_files.get("sft_eval"),
            "probe_file": input_files.get("sft_probe"),
            "dpo_train_file": input_files.get("dpo_train"),
            "dpo_eval_file": input_files.get("dpo_eval"),
            "base_model_hint": "Nemotron-3-Nano-30B",
            "lora_rank_max": 32,
            "target_modules_hint": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
            "max_seq_len_hint": 2048,
            "train_mode_hint": "start_sft_only_then_ablate_dpo",
            "do_not_package_without_adapter_eval": True,
        },
        "gates": {},
        "training_authorized": False,
        "package_authorized": False,
        "submission_authorized": False,
        "leaderboard_claim": False,
        "no_0_93_evidence": True,
        "no_0_95_evidence": True,
    }


def _prompt_issues(train: list[dict[str, Any]], eval_rows: list[dict[str, Any]], probe: list[dict[str, Any]]) -> tuple[int, int]:
    splits = {"train": train, "eval": eval_rows, "probe": probe}
    duplicate_count = 0
    prompt_sets: dict[str, set[str]] = {}
    for split, rows in splits.items():
        prompts = [" ".join(str(row.get("prompt", "")).split()) for row in rows]
        duplicate_count += len(prompts) - len(set(prompts))
        prompt_sets[split] = set(prompts)
    leakage = len(prompt_sets["train"] & prompt_sets["eval"]) + len(prompt_sets["train"] & prompt_sets["probe"]) + len(prompt_sets["eval"] & prompt_sets["probe"])
    return duplicate_count, leakage


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_write(out: Path | None, manifest: dict[str, Any]) -> None:
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
