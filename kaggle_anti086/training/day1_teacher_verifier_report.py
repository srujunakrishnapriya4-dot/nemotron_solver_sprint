from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import re
import sys
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kaggle_anti086.training.day1_teacher_families import available_teachers
from kaggle_anti086.training.day1_teacher_trainability_gate import verify_trainable_batch


def build_teacher_verifier_report(
    root: Path,
    *,
    per_family: int = 100,
    seed: int = 123,
    out: Path | None = None,
) -> dict[str, Any]:
    rng = random.Random(seed)
    rows = []
    generation_warnings: list[str] = []
    seen_prompts: set[str] = set()
    for family, teacher in available_teachers().items():
        family_rows = []
        attempts = 0
        max_attempts = max(1000, per_family * 100)
        while len(family_rows) < per_family and attempts < max_attempts:
            example = teacher.generate_synthetic(
                rng,
                ("easy", "medium", "hard")[attempts % 3],
                ("direct", "examples_query", "table", "story", "minimal")[attempts % 5],
            )
            prompt_key = _normalize_prompt(example.prompt)
            attempts += 1
            if prompt_key in seen_prompts:
                continue
            seen_prompts.add(prompt_key)
            family_rows.append(example)
        if len(family_rows) < per_family:
            generation_warnings.append(
                f"{family}: generated {len(family_rows)} unique prompts after {attempts} attempts"
            )
        rows.extend(family_rows)
    generated_checks, generated_report = verify_trainable_batch(rows)

    bad_rows = _bad_rows(rows[0])
    combined_checks, combined_report = verify_trainable_batch([*rows, *bad_rows])
    bad_checks = combined_checks[len(rows) :]
    bad_passed = sum(1 for check in bad_checks if check.trainable)
    bad_rejected = len(bad_checks) - bad_passed

    status = "PASS" if generated_report.fail_rows == 0 and bad_passed == 0 and not generation_warnings else "FAIL"
    family_pass = generated_report.family_pass_counts
    report = {
        "schema_version": 1,
        "created_by": "DAY1_TEACHER_VERIFIER_REPORT",
        "status": status,
        "generated_rows_checked": generated_report.total_rows,
        "generated_rows_passed": generated_report.pass_rows,
        "generated_rows_failed": generated_report.fail_rows,
        "bad_rows_injected": len(bad_rows),
        "bad_rows_rejected": bad_rejected,
        "bad_rows_accidentally_passed": bad_passed,
        "duplicate_prompt_count": combined_report.duplicate_prompt_count,
        "abstain_placeholder_count": combined_report.abstain_placeholder_count,
        "multiple_box_count": combined_report.multiple_box_count,
        "text_after_box_count": combined_report.text_after_box_count,
        "answer_mismatch_count": combined_report.answer_mismatch_count,
        "ambiguity_nonzero_count": combined_report.ambiguity_nonzero_count,
        "unknown_family_count": combined_report.unknown_family_count,
        "unsafe_metadata_count": combined_report.unsafe_metadata_count,
        "family_pass_counts": family_pass,
        "family_fail_counts": generated_report.family_fail_counts,
        "rejection_reasons": combined_report.rejection_reasons,
        "generation_warnings": generation_warnings,
        "trainability_gate_ready_for_phase4": status == "PASS" and len(family_pass) >= 11,
        "safe_to_train_lora": False,
        "safe_to_package": False,
        "safe_to_submit": False,
        "leaderboard_claim": False,
        "no_0_93_evidence": True,
        "no_0_95_evidence": True,
        "model_results_faked": False,
    }
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Day1 Phase3 teacher trainability verifier report.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, default=Path("artifacts/sprint11/day1_teacher_verifier_report.json"))
    parser.add_argument("--per-family", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args(argv)
    report = build_teacher_verifier_report(args.root, per_family=args.per_family, seed=args.seed, out=args.out)
    print(json.dumps({"status": report["status"], "out": str(args.out)}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


def _bad_rows(example: Any) -> list[dict[str, Any]]:
    row = asdict(example)
    duplicate = dict(row)
    return [
        {**row, "answer": "ABSTAIN", "target_text": "\\boxed{ABSTAIN}", "trace": "\\boxed{ABSTAIN}"},
        {**row, "target_text": "\\boxed{1}\\boxed{1}", "trace": "\\boxed{1}\\boxed{1}"},
        {**row, "target_text": f"{row['trace']} trailing", "trace": f"{row['trace']} trailing"},
        {**row, "answer": "definitely_wrong"},
        duplicate,
        duplicate,
        {**row, "ambiguity_count": 2},
        {**row, "family": "unknown_family"},
        {**row, "metadata": {"source": "model_guess_unverified"}},
    ]


def _normalize_prompt(prompt: str) -> str:
    return re.sub(r"\s+", " ", str(prompt).strip())


if __name__ == "__main__":
    raise SystemExit(main())
