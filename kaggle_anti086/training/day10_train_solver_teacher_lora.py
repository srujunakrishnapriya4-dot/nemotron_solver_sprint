from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.v2_corpus_io import file_record, read_json, read_jsonl, write_json_checked
from kaggle_anti086.kaggle_path_safety import require_writable_output_dir
from kaggle_anti086.training.gpu_memory_audit import capture_gpu_memory_snapshot, cuda_empty_cache
from kaggle_anti086.training.lora_backend import prepare_model_for_v2a_training, run_lora_training, validate_saved_adapter
from kaggle_anti086.training.model_loader import load_base_model, load_tokenizer
from kaggle_anti086.training.sft_dataset import SFTItem, Sprint11SFTDataset, build_sft_dataset_report
from kaggle_anti086.training.training_config_schema import _target_modules
from kaggle_anti086.training.training_config_schema import load_training_config
from kaggle_anti086.training.training_log_history import build_training_log_history


DEFAULT_POLICY = {
    "min_full_training_rows": 4096,
    "min_smoke_rows": 2000,
    "min_format_only_rows": 100,
    "max_subfamily_skew_warning_count": 0,
}


def build_training_readiness(
    *,
    config_path: str | Path,
    teacher_audit_path: str | Path,
    overlap_audit_path: str | Path,
    learnability_audit_path: str | Path,
    mix_repair_report_path: str | Path | None = None,
    capacity_audit_path: str | Path,
    abstain_policy_path: str | Path | None = None,
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
    mix_repair = read_json(mix_repair_report_path) if mix_repair_report_path else {}
    capacity = read_json(capacity_audit_path)
    corpus_path = Path(str(config.get("train_teacher_path", "artifacts/sprint11/day10_solver_teacher_direct.jsonl")))
    row_count = int(mix_repair.get("direct_answer_rows", teacher_audit.get("row_count", 0)))
    family_counts = dict(mix_repair.get("by_family", teacher_audit.get("family_counts", {})))
    format_only_count = int(mix_repair.get("format_only_count", family_counts.get("format_only", 0)))
    subfamily_skew_count = len(learnability.get("subfamily_balance_warnings", {}) or {})
    failures: list[str] = []
    warnings: list[str] = []
    if str(config.get("stage")) not in {"v4_solver_teacher_lora_small", "v4_solver_teacher_lora_wide", "v4_solver_teacher_lora_smoke_bf16_qv"}:
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
    if mix_repair_report_path and mix_repair.get("status") != "PASS":
        failures.append("mix_repair_report_not_pass")
    if capacity.get("status") != "PASS":
        failures.append("capacity_audit_not_pass")
    if not corpus_path.exists():
        failures.append("teacher_corpus_missing")
    if row_count < policy["min_smoke_rows"]:
        failures.append("row_count_below_smoke_minimum")
    if abstain_policy_path and not Path(abstain_policy_path).exists():
        failures.append("abstain_policy_missing")
    if smoke_steps is not None and (smoke_steps <= 0 or smoke_steps > 5):
        failures.append("smoke_steps_must_be_1_to_5")
    if smoke_steps is not None and not kaggle_mode:
        failures.append("smoke_training_requires_kaggle_mode")
    if config.get("smoke_bf16_runtime_only"):
        if bool(config.get("load_in_4bit", True)):
            failures.append("smoke_bf16_runtime_only_requires_load_in_4bit_false")
        if set(_target_modules(config)) != {"q_proj", "v_proj"}:
            failures.append("smoke_bf16_runtime_only_targets_must_be_q_proj_v_proj")
        if not dry_run and smoke_steps is None:
            failures.append("smoke_bf16_runtime_only_forbids_full_training")
        if bool(config.get("full_training_allowed", True)):
            failures.append("smoke_bf16_runtime_only_full_training_allowed_must_be_false")
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
    if not mix_repair and subfamily_skew_count > policy["max_subfamily_skew_warning_count"]:
        full_blockers.append("subfamily_skew_warning_count_exceeds_policy")
    for blocker in (mix_repair.get("remaining_blockers", []) if mix_repair else []):
        if blocker not in {"target_abstain_rows_not_met"} and blocker not in full_blockers:
            full_blockers.append(str(blocker))
    if failures:
        training_gate_status = "FAIL"
    elif full_blockers:
        training_gate_status = "BLOCK_FULL_TRAINING"
    else:
        training_gate_status = "PASS"
    dry_run_allowed = not any(item in failures for item in ("teacher_audit_not_pass", "overlap_audit_not_pass", "learnability_audit_not_pass", "capacity_audit_not_pass"))
    smoke_training_allowed = not failures and bool(mix_repair.get("smoke_training_eligible", row_count >= policy["min_smoke_rows"]))
    full_training_allowed = False
    report = {
        "status": "PASS" if dry_run_allowed and not failures else "FAIL",
        "training_gate_status": training_gate_status,
        "dry_run": bool(dry_run),
        "trained": False,
        "smoke_training_completed": False,
        "smoke_steps": smoke_steps,
        "steps_completed": 0,
        "intended_steps": smoke_steps if smoke_steps is not None else int(config.get("num_steps", 0) or 0),
        "loss_start": None,
        "loss_end": None,
        "loss_min": None,
        "loss_max": None,
        "loss_nan_detected": False,
        "grad_nan_detected": False,
        "adapter_dir": "",
        "adapter_config_exists": False,
        "adapter_model_exists": False,
        "dataset_report": {},
        "memory_snapshots": [],
        "training_log_history_path": "",
        "run_provenance_path": "",
        "dry_run_allowed": dry_run_allowed,
        "smoke_training_allowed": smoke_training_allowed,
        "full_training_allowed": full_training_allowed,
        "full_training_blockers": full_blockers,
        "row_count": row_count,
        "format_only_count": format_only_count,
        "subfamily_skew_warning_count": subfamily_skew_count,
        "family_counts": family_counts,
        "mix_repair_status": mix_repair.get("status"),
        "mix_repair_report": str(mix_repair_report_path) if mix_repair_report_path else "",
        "config_path": str(config_path),
        "teacher_corpus": file_record(corpus_path) if corpus_path.exists() else {"path": str(corpus_path), "exists": False},
        "policy": policy,
        "warnings": warnings,
        "failures": failures,
        "no_training_performed": bool(dry_run),
        "no_adapter_packaged": True,
        "no_submission_created": True,
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
        "created_by": "SPRINT-11_DAY10_PASS10D",
        "stage": config.get("stage"),
        "config": file_record(config_path),
        "teacher_corpus": summary.get("teacher_corpus", {}),
        "adapter_dir": summary.get("adapter_dir", ""),
        "trained": summary.get("trained", False),
        "smoke_training_completed": summary.get("smoke_training_completed", False),
        "smoke_steps": summary.get("smoke_steps"),
        "steps_completed": summary.get("steps_completed", 0),
        "dataset_report": summary.get("dataset_report", {}),
        "training_log_history_path": summary.get("training_log_history_path", ""),
        "run_provenance_path": summary.get("run_provenance_path", ""),
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


def run_day10_smoke_backend(
    *,
    config_path: str | Path,
    summary: dict[str, Any],
    smoke_steps: int,
    out_summary_path: str | Path,
) -> dict[str, Any]:
    config = load_training_config(config_path)
    smoke_config = dict(config)
    smoke_config["num_steps"] = int(smoke_steps)
    smoke_config["train_direct_path"] = str(smoke_config.get("train_teacher_path", "artifacts/sprint11/day10_solver_teacher_direct.jsonl"))
    smoke_config["train_solver_corrected_path"] = ""
    output_dir = build_day10_smoke_output_dir(smoke_config, config_path=config_path, smoke_steps=smoke_steps)
    memory_snapshots = [capture_gpu_memory_snapshot("day10_before_tokenizer_load", kaggle_mode=True)]
    tokenizer = None
    model = None
    lora_model = None
    failures: list[str] = []
    warnings: list[str] = []
    dataset_report: dict[str, Any] = {}
    train_report: dict[str, Any] = {}
    validation: dict[str, Any] = {"status": "FAIL", "adapter_config_exists": False, "adapter_model_exists": False, "failures": ["adapter_not_validated"]}
    try:
        tokenizer = load_tokenizer(str(smoke_config["base_model_path"]))
        sft_items = load_day10_teacher_sft_items(smoke_config)
        dataset_report = build_sft_dataset_report(smoke_config, sft_items)
        if dataset_report.get("status") != "PASS":
            failures.extend(dataset_report.get("failures", ["dataset_report_not_pass"]))
            raise RuntimeError("dataset_report_not_pass")
        dataset = Sprint11SFTDataset(sft_items, tokenizer, int(smoke_config.get("max_seq_len", 1024)), prevalidate=True)
        memory_snapshots.append(capture_gpu_memory_snapshot("day10_before_model_load", kaggle_mode=True))
        model = load_base_model(
            str(smoke_config["base_model_path"]),
            load_in_4bit=bool(smoke_config.get("load_in_4bit", False)),
            bf16=_resolve_bf16(smoke_config),
        )
        memory_snapshots.append(capture_gpu_memory_snapshot("day10_after_model_load", kaggle_mode=True))
        lora_model = prepare_model_for_v2a_training(model, smoke_config)
        memory_snapshots.append(capture_gpu_memory_snapshot("day10_after_lora_prepare", kaggle_mode=True))
        train_report = run_lora_training(lora_model, tokenizer, dataset, smoke_config, output_dir)
        validation = validate_saved_adapter(output_dir)
        memory_snapshots.append(capture_gpu_memory_snapshot("day10_after_smoke_train", kaggle_mode=True))
    except Exception as exc:
        failures.append(f"smoke_backend_exception:{type(exc).__name__}:{exc}")
    finally:
        lora_model = None
        model = None
        tokenizer = None
        cuda_empty_cache("day10_smoke_backend_cleanup")
        memory_snapshots.append(capture_gpu_memory_snapshot("day10_after_cleanup", kaggle_mode=True))

    steps_completed = int(train_report.get("steps_completed", 0) or 0)
    loss_nan = bool(train_report.get("loss_nan_detected", False))
    grad_nan = bool(train_report.get("grad_nan_detected", False))
    if train_report and train_report.get("status") != "PASS":
        failures.append("backend_training_status_not_pass")
    if validation.get("status") != "PASS":
        failures.extend(str(item) for item in validation.get("failures", ["adapter_validation_failed"]))
    if steps_completed != int(smoke_steps):
        failures.append("smoke_steps_incomplete")
    if loss_nan:
        failures.append("nan_loss_detected")
    if grad_nan:
        failures.append("nan_grad_detected")

    log_history_path = _sidecar_path(out_summary_path, "training_log_history")
    log_report = build_training_log_history(train_report)
    write_json_checked(log_history_path, log_report, field_name="day10_training_log_history")
    provenance_path = _sidecar_path(out_summary_path, "run_provenance")
    provenance = build_day10_run_provenance(config_path, smoke_config, summary=summary, output_adapter_dir=output_dir)
    write_json_checked(provenance_path, provenance, field_name="day10_run_provenance")
    backend_status = "PASS" if not failures else "FAIL"
    backend_summary = {
        "backend_status": backend_status,
        "trained": backend_status == "PASS",
        "smoke_training_completed": backend_status == "PASS",
        "smoke_steps": int(smoke_steps),
        "steps_completed": steps_completed,
        "intended_steps": int(smoke_steps),
        "loss_start": train_report.get("loss_start"),
        "loss_end": train_report.get("loss_end"),
        "loss_min": train_report.get("loss_min"),
        "loss_max": train_report.get("loss_max"),
        "loss_nan_detected": loss_nan,
        "grad_nan_detected": grad_nan,
        "adapter_dir": str(output_dir),
        "adapter_config_exists": bool(validation.get("adapter_config_exists", False)),
        "adapter_model_exists": bool(validation.get("adapter_model_exists", False)),
        "dataset_report": dataset_report,
        "memory_snapshots": memory_snapshots,
        "training_log_history_path": str(log_history_path),
        "run_provenance_path": str(provenance_path),
        "backend_train_report": train_report,
        "backend_adapter_validation": validation,
        "warnings": warnings,
        "failures": failures,
        "full_training_allowed": False,
        "packaging_allowed": False,
        "submission_allowed": False,
        "no_leaderboard_evidence": True,
        "no_0_95_evidence": True,
    }
    return backend_summary


def load_day10_teacher_sft_items(config: dict[str, Any]) -> list[SFTItem]:
    path = Path(str(config.get("train_teacher_path", config.get("train_direct_path", "artifacts/sprint11/day10_solver_teacher_direct.jsonl"))))
    items: list[SFTItem] = []
    for row in read_jsonl(path):
        adapted = _adapt_day10_teacher_row_for_sft(row)
        if adapted["answer"] == "ABSTAIN":
            raise ValueError(f"abstain_row_in_direct_teacher_corpus:{adapted.get('id')}")
        items.append(SFTItem(row=adapted, source="direct_answer", sampling_weight=float(config.get("teacher_direct_weight", 1.0))))
    return items


def _adapt_day10_teacher_row_for_sft(row: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(row.get("metadata", {}))
    adapted = dict(row)
    adapted["verification_status"] = "verified" if row.get("verified") is True else str(row.get("verification_status", ""))
    adapted["rule_id"] = row.get("rule_id") or metadata.get("source_rule_id")
    adapted["leakage_group"] = row.get("leakage_group") or metadata.get("source_leakage_group")
    adapted["source"] = row.get("source") or "day10_solver_teacher_direct"
    adapted["metadata"] = metadata
    return adapted


def build_day10_smoke_output_dir(config: dict[str, Any], *, config_path: str | Path, smoke_steps: int) -> Path:
    target_root = Path("/kaggle/working/anti086_adapters")
    seed = json.dumps(
        {
            "stage": config.get("stage"),
            "config": file_record(config_path).get("sha256"),
            "corpus": file_record(config.get("train_teacher_path", config.get("train_direct_path", ""))).get("sha256"),
            "steps": int(smoke_steps),
            "created": datetime.now(timezone.utc).isoformat(),
        },
        sort_keys=True,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short_hash = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:8]
    output_dir = target_root / f"day10_solver_teacher_smoke_{stamp}_{short_hash}"
    require_writable_output_dir(output_dir, field_name="output_adapter_dir")
    if output_dir.exists():
        raise FileExistsError(f"refusing_to_overwrite_adapter_dir:{output_dir}")
    return output_dir


def build_day10_run_provenance(config_path: str | Path, config: dict[str, Any], *, summary: dict[str, Any], output_adapter_dir: str | Path) -> dict[str, Any]:
    return {
        "status": "PASS",
        "created_by": "SPRINT-11_DAY10_PASS10D",
        "config": file_record(config_path),
        "teacher_corpus": file_record(config.get("train_teacher_path", config.get("train_direct_path", ""))),
        "teacher_audit": file_record(config.get("teacher_audit_path", "artifacts/sprint11/day10_solver_teacher_audit.json")),
        "teacher_overlap_audit": file_record(config.get("teacher_overlap_audit_path", "artifacts/sprint11/day10_teacher_overlap_audit.json")),
        "teacher_learnability_audit": file_record(config.get("teacher_learnability_audit_path", "artifacts/sprint11/day10_teacher_learnability_audit.json")),
        "mix_repair_report": file_record(summary.get("mix_repair_report", "")) if summary.get("mix_repair_report") else {},
        "output_adapter_dir": str(output_adapter_dir),
        "stage": config.get("stage"),
        "rank": int(config.get("rank", 0) or 0),
        "target_modules": _target_modules(config),
        "num_steps": int(config.get("num_steps", 0) or 0),
        "assistant_only_loss": bool(config.get("assistant_only_loss", False)),
        "full_prompt_loss": bool(config.get("full_prompt_loss", True)),
        "train_on_user": bool(config.get("train_on_user", True)),
        "packaging_allowed": False,
        "submission_allowed": False,
        "no_leaderboard_evidence": True,
        "no_0_95_evidence": True,
    }


def _resolve_bf16(config: dict[str, Any]) -> bool:
    value = config.get("bf16", True)
    if value == "auto":
        return True
    return bool(value)


def _sidecar_path(path: str | Path, suffix: str) -> Path:
    target = Path(path)
    return target.with_name(f"{target.stem}_{suffix}{target.suffix}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Day 10 solver-teacher LoRA dry-run/smoke/full gate.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--teacher-audit", required=True)
    parser.add_argument("--overlap-audit", required=True)
    parser.add_argument("--learnability-audit", required=True)
    parser.add_argument("--mix-repair-report", default=None)
    parser.add_argument("--capacity-audit", required=True)
    parser.add_argument("--abstain-policy-path", default=None)
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
        mix_repair_report_path=args.mix_repair_report,
        capacity_audit_path=args.capacity_audit,
        abstain_policy_path=args.abstain_policy_path,
        dry_run=args.dry_run,
        smoke_steps=args.smoke_steps,
        kaggle_mode=args.kaggle_mode,
        smoke_report=args.smoke_report,
    )
    manifest = build_manifest(args.config, summary)
    if not args.dry_run and args.smoke_steps is not None and args.kaggle_mode and summary.get("smoke_training_allowed"):
        backend_summary = run_day10_smoke_backend(
            config_path=args.config,
            summary=summary,
            smoke_steps=args.smoke_steps,
            out_summary_path=args.out_summary,
        )
        summary.update(backend_summary)
        summary["status"] = "PASS" if backend_summary["backend_status"] == "PASS" else "FAIL"
        summary["training_gate_status"] = "PASS" if backend_summary["backend_status"] == "PASS" else "FAIL"
        summary["no_training_performed"] = False if backend_summary["backend_status"] == "PASS" else summary.get("no_training_performed", False)
        manifest = build_manifest(args.config, summary)
    elif not args.dry_run and args.smoke_steps is None:
        summary["status"] = "FAIL"
        summary["training_gate_status"] = "BLOCK_FULL_TRAINING_NOT_IMPLEMENTED_PASS10D"
        summary["full_training_allowed"] = False
        if "full_training_not_implemented_in_pass10d" not in summary["failures"]:
            summary["failures"].append("full_training_not_implemented_in_pass10d")
        manifest = build_manifest(args.config, summary)
    write_json_checked(args.out_summary, summary, field_name="day10_train_solver_teacher_summary")
    write_json_checked(args.out_manifest, manifest, field_name="day10_train_solver_teacher_manifest")
    print(json.dumps({"status": summary["status"], "training_gate_status": summary["training_gate_status"], "full_training_allowed": summary["full_training_allowed"]}, sort_keys=True))
    return 0 if args.dry_run and summary["dry_run_allowed"] else (0 if summary["training_gate_status"] == "PASS" and summary.get("status") == "PASS" else 2)


if __name__ == "__main__":
    raise SystemExit(main())
