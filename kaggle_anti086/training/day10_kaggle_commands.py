from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.kaggle_path_safety import safe_write_text


COMMAND_PLAN = """# SPRINT-11 Day 10 PASS 10A/10B command plan
# No package. No submit. No submission.zip. No safetensors in Git. Tinker is baseline-only if locally available.

## Stage A: Day 9 regression
python -m pytest tests/test_day10_day9_lock_regression.py -q -p no:cacheprovider

## Stage B: teacher corpus
python kaggle_anti086/training/day10_build_solver_teacher_corpus.py --rows 768 --seed 1110 --out-direct artifacts/sprint11/day10_solver_teacher_direct.jsonl --out-manifest artifacts/sprint11/day10_solver_teacher_manifest.json --out-audit artifacts/sprint11/day10_solver_teacher_audit.json --out-overlap artifacts/sprint11/day10_teacher_overlap_audit.json --out-learnability artifacts/sprint11/day10_teacher_learnability_audit.json

## Stage C: audits
python -m pytest tests/test_day10_solver_teacher_corpus.py tests/test_day10_teacher_leakage.py tests/test_day10_teacher_answer_format.py -q -p no:cacheprovider

## Stage D: capacity audit + dry-run train
python kaggle_anti086/training/day10_lora_capacity_audit.py --configs kaggle_anti086/training/configs/v4_solver_teacher_lora_small.yaml kaggle_anti086/training/configs/v4_solver_teacher_lora_wide.yaml --out artifacts/sprint11/day10_lora_capacity_audit.json
python kaggle_anti086/training/day10_train_solver_teacher_lora.py --config kaggle_anti086/training/configs/v4_solver_teacher_lora_small.yaml --teacher-audit artifacts/sprint11/day10_solver_teacher_audit.json --overlap-audit artifacts/sprint11/day10_teacher_overlap_audit.json --learnability-audit artifacts/sprint11/day10_teacher_learnability_audit.json --capacity-audit artifacts/sprint11/day10_lora_capacity_audit.json --out-summary artifacts/sprint11/day10_train_solver_teacher_dry_run_summary.json --out-manifest artifacts/sprint11/day10_train_solver_teacher_manifest.json --dry-run

## Stage E: smoke train, Kaggle only
# RUN ONLY AFTER Stage A-D PASS and after reviewing full_training_blockers.
# python kaggle_anti086/training/day10_train_solver_teacher_lora.py --config kaggle_anti086/training/configs/v4_solver_teacher_lora_small.yaml --teacher-audit artifacts/sprint11/day10_solver_teacher_audit.json --overlap-audit artifacts/sprint11/day10_teacher_overlap_audit.json --learnability-audit artifacts/sprint11/day10_teacher_learnability_audit.json --capacity-audit artifacts/sprint11/day10_lora_capacity_audit.json --out-summary artifacts/sprint11/day10_train_solver_teacher_smoke_summary.json --out-manifest artifacts/sprint11/day10_train_solver_teacher_manifest.json --smoke-steps 5 --kaggle-mode

## Stage F: full train, gated
# RUN ONLY IF smoke report is PASS and corpus adequacy blockers are resolved.
# python kaggle_anti086/training/day10_train_solver_teacher_lora.py --config kaggle_anti086/training/configs/v4_solver_teacher_lora_small.yaml --teacher-audit artifacts/sprint11/day10_solver_teacher_audit.json --overlap-audit artifacts/sprint11/day10_teacher_overlap_audit.json --learnability-audit artifacts/sprint11/day10_teacher_learnability_audit.json --capacity-audit artifacts/sprint11/day10_lora_capacity_audit.json --out-summary artifacts/sprint11/day10_train_solver_teacher_full_summary.json --out-manifest artifacts/sprint11/day10_train_solver_teacher_manifest.json --smoke-report artifacts/sprint11/day10_train_solver_teacher_smoke_summary.json --kaggle-mode

## Stage G: adapter eval
python kaggle_anti086/training/day10_adapter_eval.py --base-model-path /kaggle/input/nemotron-model --day10-adapter-dir /kaggle/working/anti086_adapters/day10_solver_teacher --eval-path artifacts/sprint11/day5_rule_holdout_answerable_512.jsonl --out-report artifacts/sprint11/day10_adapter_eval_report.json --out-predictions artifacts/sprint11/day10_adapter_eval_predictions.jsonl --max-rows 128

## Stage H: error mining + candidate decision
python kaggle_anti086/training/day10_adapter_error_mining.py --predictions artifacts/sprint11/day10_adapter_eval_predictions.jsonl --out-report artifacts/sprint11/day10_adapter_error_mining_report.json --out-rows artifacts/sprint11/day10_adapter_error_mining_rows.jsonl
python kaggle_anti086/training/day10_candidate_decision.py --eval-report artifacts/sprint11/day10_adapter_eval_report.json --error-mining-report artifacts/sprint11/day10_adapter_error_mining_report.json --package-guard-report artifacts/sprint11/day10_adapter_package_guard_report.json --out artifacts/sprint11/day10_candidate_decision_report.json

## Stage I: package rehearsal, dry-run only
python kaggle_anti086/training/day10_adapter_package_guard.py --adapter-dir /kaggle/working/anti086_adapters/day10_solver_teacher --out artifacts/sprint11/day10_adapter_package_guard_report.json --dry-run-package

# No real submission package command is present. Public-LB calibration must be explicit before any future package/submission decision.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write Day 10 PASS 10A staged command plan.")
    parser.add_argument("--out", default="artifacts/sprint11/day10_kaggle_command_plan.txt")
    args = parser.parse_args(argv)
    safe_write_text(args.out, COMMAND_PLAN, field_name="day10_kaggle_command_plan")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
