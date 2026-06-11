from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import random
import re
import sys
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kaggle_anti086.training.day1_teacher_families import SyntheticExample, available_teachers
from kaggle_anti086.training.day1_teacher_trainability_gate import verify_trainable_row
from kaggle_anti086.training.day1_verified_data_audit import (
    audit_dedup_leakage,
    audit_family_balance,
    audit_format,
    audit_prompt_diversity,
    audit_teacher_verification,
)
from kaggle_anti086.training.day1_verified_data_manifest import build_manifest, manifest_status
from kaggle_anti086.training.day1_verified_data_schema import (
    HardNegativePair,
    RejectedRow,
    VerifiedSFTRecord,
    stable_pair_id,
    stable_record_id,
    stable_rejected_id,
    to_json_dict,
    write_jsonl,
)


TEACHER_VERSION = "day1_phase4_verified_synthetic_factory_v1"
DIFFICULTIES = ("easy", "medium", "hard")
PROMPT_STYLES = ("direct", "minimal", "examples_query", "table", "story")


def build_verified_data_factory(
    out_dir: Path,
    *,
    train_per_family: int = 1000,
    eval_per_family: int = 100,
    probe_per_family: int = 100,
    hard_negatives_per_family: int = 500,
    seed: int = 123,
    require_preconditions: bool = True,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path.cwd()
    input_reports = {
        "phase2b_teacher_bank": str(root / "artifacts/sprint11/day1_teacher_bank_report_phase2b_stress_1000.json"),
        "phase3_verifier": str(root / "artifacts/sprint11/day1_teacher_verifier_report.json"),
    }
    if require_preconditions:
        _check_preconditions(input_reports)

    teachers = available_teachers()
    rejected: list[RejectedRow] = []
    seen_positive_prompts: set[str] = set()
    seen_record_ids: set[str] = set()
    train_records = _generate_split(
        teachers,
        "train",
        train_per_family,
        seed,
        seen_positive_prompts,
        seen_record_ids,
        rejected,
    )
    eval_records = _generate_split(
        teachers,
        "eval",
        eval_per_family,
        seed + 100000,
        seen_positive_prompts,
        seen_record_ids,
        rejected,
    )
    probe_records = _generate_split(
        teachers,
        "probe",
        probe_per_family,
        seed + 200000,
        seen_positive_prompts,
        seen_record_ids,
        rejected,
    )
    pairs = _generate_pairs(
        teachers,
        hard_negatives_per_family,
        seed + 300000,
        seen_positive_prompts,
        rejected,
    )

    train_rows = [to_json_dict(row) for row in train_records]
    eval_rows = [to_json_dict(row) for row in eval_records]
    probe_rows = [to_json_dict(row) for row in probe_records]
    pair_rows = [to_json_dict(row) for row in pairs]
    rejected_rows = [to_json_dict(row) for row in rejected]

    output_files = {
        "train": str(out_dir / "phase4_verified_sft_train.jsonl"),
        "eval": str(out_dir / "phase4_verified_sft_eval.jsonl"),
        "probe": str(out_dir / "phase4_verified_eval_probe.jsonl"),
        "hard_negatives": str(out_dir / "phase4_hard_negative_pairs.jsonl"),
        "rejected": str(out_dir / "phase4_rejected_rows.jsonl"),
        "manifest": str(out_dir / "phase4_verified_data_manifest.json"),
        "family_balance": str(out_dir / "phase4_family_balance_report.json"),
        "prompt_diversity": str(out_dir / "phase4_prompt_diversity_report.json"),
        "dedup_leakage": str(out_dir / "phase4_dedup_leakage_report.json"),
        "format_audit": str(out_dir / "phase4_format_audit_report.json"),
        "teacher_verification": str(out_dir / "phase4_teacher_verification_audit.json"),
        "decision": str(out_dir / "phase4_decision_report.json"),
    }

    all_positive = [*train_rows, *eval_rows, *probe_rows]
    audits = {
        "format": audit_format(all_positive),
        "dedup_leakage": audit_dedup_leakage(train_rows, eval_rows, probe_rows, pair_rows),
        "family_balance": audit_family_balance(
            train_rows,
            {family: train_per_family for family in teachers},
        ),
        "prompt_diversity": audit_prompt_diversity(train_rows),
        "teacher_verification": audit_teacher_verification(all_positive),
    }
    counts = {
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "probe_rows": len(probe_rows),
        "hard_negative_pairs": len(pair_rows),
        "rejected_rows": len(rejected_rows),
        "total_positive_rows": len(all_positive),
    }
    manifest, decision = build_manifest(
        seed=seed,
        input_reports=input_reports,
        output_files=output_files,
        counts=counts,
        train_rows=train_rows,
        eval_rows=eval_rows,
        probe_rows=probe_rows,
        rejected_rows=rejected_rows,
        audits=audits,
    )
    status = manifest_status(manifest, audits)

    write_jsonl(Path(output_files["train"]), train_records)
    write_jsonl(Path(output_files["eval"]), eval_records)
    write_jsonl(Path(output_files["probe"]), probe_records)
    write_jsonl(Path(output_files["hard_negatives"]), pairs)
    write_jsonl(Path(output_files["rejected"]), rejected)
    _write_json(Path(output_files["family_balance"]), audits["family_balance"])
    _write_json(Path(output_files["prompt_diversity"]), audits["prompt_diversity"])
    _write_json(Path(output_files["dedup_leakage"]), audits["dedup_leakage"])
    _write_json(Path(output_files["format_audit"]), audits["format"])
    _write_json(Path(output_files["teacher_verification"]), audits["teacher_verification"])
    manifest_dict = to_json_dict(manifest)
    manifest_dict["status"] = status
    manifest_dict["families"] = manifest_dict["family_counts"]
    manifest_dict["audits"] = {name: audit["status"] for name, audit in audits.items()}
    _write_json(Path(output_files["manifest"]), manifest_dict)
    _write_json(Path(output_files["decision"]), decision)
    return {
        "status": status,
        "manifest": manifest_dict,
        "decision": decision,
        "audits": audits,
        "output_files": output_files,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Day1 Phase4 verified synthetic SFT/eval/pair artifacts.")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/sprint11"))
    parser.add_argument("--train-per-family", type=int, default=1000)
    parser.add_argument("--eval-per-family", type=int, default=100)
    parser.add_argument("--probe-per-family", type=int, default=100)
    parser.add_argument("--hard-negatives-per-family", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args(argv)
    result = build_verified_data_factory(
        args.out_dir,
        train_per_family=args.train_per_family,
        eval_per_family=args.eval_per_family,
        probe_per_family=args.probe_per_family,
        hard_negatives_per_family=args.hard_negatives_per_family,
        seed=args.seed,
    )
    manifest = result["manifest"]
    print(
        json.dumps(
            {
                "status": result["status"],
                "train_rows": manifest["counts"]["train_rows"],
                "eval_rows": manifest["counts"]["eval_rows"],
                "probe_rows": manifest["counts"]["probe_rows"],
                "hard_negative_pairs": manifest["counts"]["hard_negative_pairs"],
                "phase4_ready_for_phase5": manifest["gates"]["phase4_ready_for_phase5"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] in {"PASS", "WARN"} else 1


def _generate_split(
    teachers: dict[str, Any],
    split: str,
    per_family: int,
    seed: int,
    seen_prompts: set[str],
    seen_record_ids: set[str],
    rejected: list[RejectedRow],
) -> list[VerifiedSFTRecord]:
    records: list[VerifiedSFTRecord] = []
    for family, teacher in teachers.items():
        rng = random.Random(seed + _family_offset(family))
        family_records = 0
        attempts = 0
        max_attempts = max(2000, per_family * 250)
        while family_records < per_family and attempts < max_attempts:
            difficulty = DIFFICULTIES[attempts % len(DIFFICULTIES)]
            prompt_style = PROMPT_STYLES[attempts % len(PROMPT_STYLES)]
            attempts += 1
            try:
                example = teacher.generate_synthetic(rng, difficulty, prompt_style)
            except Exception as exc:
                _reject(rejected, family, None, "teacher_generation_error", split, seed + attempts, {"error": str(exc)})
                continue
            prompt_key = _prompt_key(example.prompt)
            if prompt_key in seen_prompts:
                _reject(rejected, family, example.prompt, "duplicate_prompt_resampled", split, seed + attempts, {})
                continue
            record = _example_to_record(example, split, seed + attempts)
            if record.record_id in seen_record_ids:
                _reject(rejected, family, example.prompt, "duplicate_record_id_resampled", split, seed + attempts, {})
                continue
            missing = _missing_required_fields(record)
            if missing:
                _reject(rejected, family, example.prompt, f"missing_{missing}", split, seed + attempts, {})
                continue
            check = verify_trainable_row(to_json_dict(record))
            if not check.trainable:
                _reject(rejected, family, example.prompt, check.rejection_reason or "trainability_gate_failed", split, seed + attempts, check.metadata)
                continue
            seen_prompts.add(prompt_key)
            seen_record_ids.add(record.record_id)
            records.append(record)
            family_records += 1
        if family_records != per_family:
            raise RuntimeError(f"{split}:{family} generated {family_records}/{per_family} accepted rows")
    return records


def _generate_pairs(
    teachers: dict[str, Any],
    per_family: int,
    seed: int,
    positive_prompts: set[str],
    rejected_rows: list[RejectedRow],
) -> list[HardNegativePair]:
    pairs: list[HardNegativePair] = []
    seen_pair_prompts: set[str] = set()
    seen_pair_ids: set[str] = set()
    for family, teacher in teachers.items():
        rng = random.Random(seed + _family_offset(family))
        accepted = 0
        attempts = 0
        max_attempts = max(2000, per_family * 250)
        while accepted < per_family and attempts < max_attempts:
            difficulty = DIFFICULTIES[attempts % len(DIFFICULTIES)]
            prompt_style = PROMPT_STYLES[attempts % len(PROMPT_STYLES)]
            attempts += 1
            try:
                example = teacher.generate_synthetic(rng, difficulty, prompt_style)
            except Exception as exc:
                _reject(rejected_rows, family, None, "pair_teacher_generation_error", "hard_negative", seed + attempts, {"error": str(exc)})
                continue
            prompt_key = _prompt_key(example.prompt)
            if prompt_key in positive_prompts or prompt_key in seen_pair_prompts:
                _reject(rejected_rows, family, example.prompt, "pair_duplicate_prompt_resampled", "hard_negative", seed + attempts, {})
                continue
            check = verify_trainable_row(to_json_dict(_example_to_record(example, "train", seed + attempts)))
            if not check.trainable:
                _reject(rejected_rows, family, example.prompt, check.rejection_reason or "pair_chosen_not_trainable", "hard_negative", seed + attempts, check.metadata)
                continue
            negative = teacher.generate_hard_negative(example, rng)
            if negative.chosen == negative.rejected or not negative.reason_rejected:
                _reject(rejected_rows, family, example.prompt, "hard_negative_malformed", "hard_negative", seed + attempts, {})
                continue
            pair_id = stable_pair_id(family, example.prompt, negative.chosen, negative.rejected, seed + attempts)
            if pair_id in seen_pair_ids:
                _reject(rejected_rows, family, example.prompt, "duplicate_pair_id_resampled", "hard_negative", seed + attempts, {})
                continue
            pairs.append(
                HardNegativePair(
                    pair_id=pair_id,
                    family=family,
                    rule_id=example.rule_id,
                    difficulty=difficulty,
                    prompt_style=prompt_style,
                    prompt=example.prompt,
                    chosen=negative.chosen,
                    rejected=negative.rejected,
                    correct_answer=negative.correct_answer,
                    rejected_answer=negative.rejected_answer,
                    reason_rejected=negative.reason_rejected,
                    verifier_status=negative.verifier_status,
                    split="hard_negative",
                    seed=seed + attempts,
                    metadata=dict(negative.metadata),
                )
            )
            seen_pair_prompts.add(prompt_key)
            seen_pair_ids.add(pair_id)
            accepted += 1
        if accepted != per_family:
            raise RuntimeError(f"hard_negative:{family} generated {accepted}/{per_family} accepted pairs")
    return pairs


def _example_to_record(example: SyntheticExample, split: str, seed: int) -> VerifiedSFTRecord:
    return VerifiedSFTRecord(
        record_id=stable_record_id(example.family, example.prompt, example.answer, example.rule_id, seed),
        family=example.family,
        rule_id=example.rule_id,
        difficulty=example.difficulty,
        prompt_style=example.prompt_style,
        prompt=example.prompt,
        answer=example.answer,
        trace=example.trace,
        target_text=example.target_text,
        source="deterministic_teacher",
        verification_status=example.verification_status,
        ambiguity_count=example.ambiguity_count,
        trainable=True,
        split=split,
        seed=seed,
        teacher_version=TEACHER_VERSION,
        metadata=dict(example.metadata),
    )


def _check_preconditions(input_reports: dict[str, str]) -> None:
    bank = _read_json(Path(input_reports["phase2b_teacher_bank"]))
    verifier = _read_json(Path(input_reports["phase3_verifier"]))
    if bank.get("teacher_bank_ready_for_verified_data_factory") is not True:
        raise ValueError("teacher_bank_ready_for_verified_data_factory is not true")
    if verifier.get("trainability_gate_ready_for_phase4") is not True:
        raise ValueError("trainability_gate_ready_for_phase4 is not true")
    if bank.get("leaderboard_claim") or verifier.get("leaderboard_claim"):
        raise ValueError("leaderboard claim present in input reports")
    if bank.get("no_0_93_evidence") is not True or verifier.get("no_0_93_evidence") is not True:
        raise ValueError("0.93 evidence flag is not blocked")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"missing required precondition report: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _missing_required_fields(record: VerifiedSFTRecord) -> str | None:
    for field in ("family", "rule_id", "difficulty", "prompt_style", "prompt", "answer", "target_text"):
        if not str(getattr(record, field)).strip():
            return field
    return None


def _reject(
    rejected: list[RejectedRow],
    family: str | None,
    prompt: str | None,
    reason: str,
    source: str,
    seed: int,
    metadata: dict[str, Any],
) -> None:
    rejected.append(
        RejectedRow(
            rejected_id=stable_rejected_id(family, prompt, reason, seed),
            family=family,
            prompt=prompt,
            reason=reason,
            source=source,
            metadata=metadata,
        )
    )


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _prompt_key(prompt: str) -> str:
    return re.sub(r"\s+", " ", str(prompt).strip())


def _family_offset(family: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(family))


if __name__ == "__main__":
    raise SystemExit(main())
