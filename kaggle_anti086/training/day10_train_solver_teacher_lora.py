from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.v2_corpus_io import file_record, read_json, read_jsonl, write_json_checked
from kaggle_anti086.training.training_config_schema import load_training_config


DEFAULT_POLICY = {
    "min_full_training_rows": 2000,
    "min_smoke_rows": 256,
    "min_format_only_rows": 20,
    "max_subfamily_skew_warning_count": 2,
}


def build_training_readiness(
    *,
    config_path: str | Path,
    teacher_audit_path: str | Path,
    overlap_audit_path: str | Path,
    learnability_audit_path: str | Path,
    capacity_audit_path: str | Path,
    dry_run: bool,
    smoke_steps: int | None = None,
    kaggle_mode: bool = False,
    smoke_report: str | Path | None = None,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    policy = dict(DEFAULT_POLICY | (policy or {}))
    config = load_training_config(config_path)
    teacher_audit = read_json(teacher_audit_path)
    overlap = read_json(overlap_audit_path)
    learnability = read_json(learnability_audit_path)
    capacity = read_json(capacity_audit_path)
    corpus_path = Path(str(config.get("train_teacher_path", "artifacts/sprint11/day10_solver_teacher_direct.jsonl")))
    row_count = int(teacher_audit.get("row_count", 0))
    family_counts = dict(teacher_audit.get("family_counts", {}))
    format_only_count = int(family_counts.get("format_only", 0))
    subfamily_skew_count = len(learnability.get("subfamily_balance_warnings", {}) or {})
    failures: list[str] = []
    warnings: list[str] = []
    if str(config.get("stage")) not in {"v4_solver_teacher_lora_small", "v4_solver_teacher_lora_wide"}:
        failures.append("not_day10_solver_teacher_stage")
    if config.get("full_prompt_loss") is not False:
        failures.append("full_prompt_loss_forbidden")
    if config.get("train_on_user") is not False:
        failures.append("train_on_user_forbidden")
    if config.get("assistant_only_loss") is not True:
        failures.append("assistant_only_loss_required")
    if teacher_audit.get("status") != "PASS":
        failures.append("teacher_audit_not_pass")
    if overlap.get("status") != "PASS":
        failures.append("overlap_audit_not_pass")
    if learnability.get("status") != "PASS":
        failures.append("learnability_audit_not_pass")
    if capacity.get("status") != "PASS":
        failures.append("capacity_audit_not_pass")
    if not corpus_path.exists():
        failures.append("teacher_corpus_missing")
    if row_count < policy["min_smoke_rows"]:
        failures.append("row_count_below_smoke_minimum")
    if smoke_steps is not None and (smoke_steps <= 0 or smoke_steps > 5):
        failures.append("smoke_steps_must_be_1_to_5")
    if smoke_steps is not None and not kaggle_mode:
        failures.append("smoke_training_requires_kaggle_mode")
    if not dry_run and smoke_steps is None:
        if not smoke_report:
            failures.append("full_training_requires_smoke_report")
        elif not _smoke_report_allows(smoke_report):
            failures.append("smoke_report_not_pass")
    full_blockers = []
    if row_count < policy["min_full_training_rows"]:
        full_blockers.append("row_count_below_full_training_minimum")
    if format_only_count < policy["min_format_only_rows"]:
        full_blockers.append("format_only_coverage_below_minimum")
    if subfamily_skew_count > policy["max_subfamily_skew_warning_count"]:
        full_blockers.append("subfamily_skew_warning_count_exceeds_policy")
    if failures:
        training_gate_status = "FAIL"
    elif full_blockers:
        training_gate_status = "BLOCK_FULL_TRAINING"
    else:
        training_gate_status = "PASS"
    dry_run_allowed = not any(item in failures for item in ("teacher_audit_not_pass", "overlap_audit_not_pass", "learnability_audit_not_pass", "capacity_audit_not_pass"))
    smoke_training_allowed = not failures and row_count >= policy["min_smoke_rows"] and format_only_count >= 1
    full_training_allowed = not failures and not full_blockers and bool(smoke_report)
    report = {
        "status": "PASS" if dry_run_allowed and not failures else "FAIL",
        "training_gate_status": training_gate_status,
        "dry_run": bool(dry_run),
        "dry_run_allowed": dry_run_allowed,
        "smoke_training_allowed": smoke_training_allowed,
        "full_training_allowed": full_training_allowed,
        "full_training_blockers": full_blockers,
        "row_count": row_count,
        "format_only_count": format_only_count,
        "subfamily_skew_warning_count": subfamily_skew_count,
        "family_counts": family_counts,
        "config_path": str(config_path),
        "teacher_corpus": file_record(corpus_path) if corpus_path.exists() else {"path": str(corpus_path), "exists": False},
        "policy": policy,
        "warnings": warnings,
        "failures": failures,
        "no_training_performed": bool(dry_run),
        "packaging_allowed": False,
        "submission_allowed": False,
        "no_leaderboard_evidence": True,
        "no_0_95_evidence": True,
    }
    if dry_run and dry_run_allowed:
        report["status"] = "PASS"
    return report


def build_manifest(config_path: str | Path, summary: dict[str, Any]) -> dict[str, Any]:
    config = load_training_config(config_path)
    manifest = {
        "schema_version": 1,
        "created_by": "SPRINT-11_DAY10_PASS10B",
        "stage": config.get("stage"),
        "config": file_record(config_path),
        "teacher_corpus": summary.get("teacher_corpus", {}),
        "dry_run_allowed": summary.get("dry_run_allowed", False),
        "smoke_training_allowed": summary.get("smoke_training_allowed", False),
        "full_training_allowed": summary.get("full_training_allowed", False),
        "full_training_blockers": summary.get("full_training_blockers", []),
        "packaging_allowed": False,
        "submission_allowed": False,
        "no_training_performed": True,
        "no_adapter_packaged": True,
        "no_submission_created": True,
    }
    return manifest


def _smoke_report_allows(path: str | Path) -> bool:
    p = Path(path)
    if not p.exists():
        return False
    data = read_json(p)
    return data.get("status") == "PASS" and bool(data.get("smoke_training_completed", True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Day 10 solver-teacher LoRA dry-run/smoke/full gate.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--teacher-audit", required=True)
    parser.add_argument("--overlap-audit", required=True)
    parser.add_argument("--learnability-audit", required=True)
    parser.add_argument("--capacity-audit", required=True)
    parser.add_argument("--out-summary", required=True)
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--kaggle-mode", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=None)
    parser.add_argument("--smoke-report", default=None)
    args = parser.parse_args(argv)
    summary = build_training_readiness(
        config_path=args.config,
        teacher_audit_path=args.teacher_audit,
        overlap_audit_path=args.overlap_audit,
        learnability_audit_path=args.learnability_audit,
        capacity_audit_path=args.capacity_audit,
        dry_run=args.dry_run,
        smoke_steps=args.smoke_steps,
        kaggle_mode=args.kaggle_mode,
        smoke_report=args.smoke_report,
    )
    manifest = build_manifest(args.config, summary)
    write_json_checked(args.out_summary, summary, field_name="day10_train_solver_teacher_summary")
    write_json_checked(args.out_manifest, manifest, field_name="day10_train_solver_teacher_manifest")
    print(json.dumps({"status": summary["status"], "training_gate_status": summary["training_gate_status"], "full_training_allowed": summary["full_training_allowed"]}, sort_keys=True))
    return 0 if args.dry_run and summary["dry_run_allowed"] else (0 if summary["training_gate_status"] == "PASS" else 2)


if __name__ == "__main__":
    raise SystemExit(main())
