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
from kaggle_anti086.training.day10_train_solver_teacher_lora import (
    _adapt_day10_teacher_row_for_sft,
    _resolve_bf16,
)
from kaggle_anti086.training.gpu_memory_audit import capture_gpu_memory_snapshot, cuda_empty_cache
from kaggle_anti086.training.lora_backend import prepare_model_for_v2a_training, run_lora_training, validate_saved_adapter
from kaggle_anti086.training.model_loader import load_base_model, load_tokenizer
from kaggle_anti086.training.sft_dataset import SFTItem, Sprint11SFTDataset, build_sft_dataset_report
from kaggle_anti086.training.training_config_schema import _target_modules, load_training_config
from kaggle_anti086.training.training_log_history import build_training_log_history


DAY11A_POLICY = {
    "min_direct_rows": 4096,
    "min_format_only_rows": 100,
}


def validate_day11a_config(config: dict[str, Any]) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    stage = str(config.get("stage", ""))
    targets = _target_modules(config)
    output_dir = str(config.get("output_adapter_dir", ""))
    if stage not in {"v2a_bf16_qv_50", "v2a_bf16_qv_150", "v2a_bf16_qv_300"}:
        failures.append("unsafe_lora_config:unknown_day11a_stage")
    if stage in {"v2a_bf16_qv_150", "v2a_bf16_qv_300"}:
        if config.get("requires_previous_candidate_pass") is not True:
            failures.append("unsafe_lora_config:scaled_config_requires_previous_candidate_pass")
        if not str(config.get("requires_day11a_decision", "")).startswith("scale_to_"):
            failures.append("unsafe_lora_config:scaled_config_requires_day11a_decision")
    if bool(config.get("load_in_4bit", True)):
        failures.append("unsafe_lora_config:load_in_4bit_forbidden")
    if set(targets) != {"q_proj", "v_proj"}:
        failures.append("unsafe_lora_config:target_modules_must_be_q_proj_v_proj")
    if "o_proj" in targets:
        failures.append("unsafe_lora_config:o_proj_unproven_forbidden")
    if "lm_head" in targets:
        failures.append("unsafe_lora_config:lm_head_forbidden")
    if int(config.get("rank", 0) or 0) > 32:
        failures.append("unsafe_lora_config:rank_gt_32")
    if config.get("assistant_only_loss") is not True:
        failures.append("unsafe_lora_config:assistant_only_loss_required")
    if config.get("full_prompt_loss") is not False:
        failures.append("unsafe_lora_config:full_prompt_loss_forbidden")
    if config.get("train_on_user") is not False:
        failures.append("unsafe_lora_config:train_on_user_forbidden")
    if config.get("real_candidate_training") is not True:
        failures.append("unsafe_lora_config:real_candidate_training_required")
    if config.get("smoke_bf16_runtime_only") is True:
        failures.append("unsafe_lora_config:smoke_only_config_not_day11a_candidate")
    if config.get("public_submission_allowed") is not False:
        failures.append("unsafe_lora_config:public_submission_allowed_forbidden")
    if config.get("submission_ready") is not False:
        failures.append("unsafe_lora_config:submission_ready_forbidden")
    try:
        require_writable_output_dir(output_dir, field_name="output_adapter_dir")
    except SystemExit as exc:
        failures.append(f"unsafe_lora_config:unsafe_output_adapter_dir:{exc}")
    normalized = output_dir.replace("\\", "/")
    if not normalized.startswith("/kaggle/working/anti086_adapters/"):
        failures.append("unsafe_lora_config:output_adapter_dir_must_be_kaggle_working_adapters")
    return failures, warnings


def build_day11a_readiness(
    *,
    config_path: str | Path,
    teacher_audit_path: str | Path,
    overlap_audit_path: str | Path,
    learnability_audit_path: str | Path,
    mix_repair_report_path: str | Path,
    capacity_audit_path: str | Path,
    adapter_dir: str | Path | None = None,
    dry_run: bool = False,
    kaggle_mode: bool = False,
    train: bool = False,
    skip_train_if_adapter_exists: bool = False,
) -> dict[str, Any]:
    config = load_training_config(config_path)
    failures, warnings = validate_day11a_config(config)
    teacher_audit = _read_required_json(teacher_audit_path, failures, "teacher_audit")
    overlap_audit = _read_required_json(overlap_audit_path, failures, "overlap_audit")
    learnability_audit = _read_required_json(learnability_audit_path, failures, "learnability_audit")
    mix_repair = _read_required_json(mix_repair_report_path, failures, "mix_repair")
    capacity_audit = _read_required_json(capacity_audit_path, failures, "capacity_audit")
    for name, report in (
        ("teacher_audit", teacher_audit),
        ("overlap_audit", overlap_audit),
        ("learnability_audit", learnability_audit),
        ("mix_repair", mix_repair),
        ("capacity_audit", capacity_audit),
    ):
        if report and report.get("status") != "PASS":
            failures.append(f"audit_not_pass:{name}")
    if mix_repair and int(mix_repair.get("direct_answer_rows", 0) or 0) < DAY11A_POLICY["min_direct_rows"]:
        failures.append("mix_repair_not_pass:direct_rows_below_4096")
    if mix_repair and int(mix_repair.get("format_only_count", 0) or 0) < DAY11A_POLICY["min_format_only_rows"]:
        failures.append("mix_repair_not_pass:format_only_below_100")
    if capacity_audit and capacity_audit.get("small_config_viable") is False:
        failures.append("capacity_audit_not_pass:small_config_not_viable")
    direct_path = Path(str(config.get("teacher_corpus_path", config.get("train_teacher_path", ""))))
    abstain_path = Path(str(config.get("abstain_policy_path", "")))
    direct_row_count = 0
    abstain_contamination_count = 0
    if not direct_path.exists():
        failures.append("missing_direct_corpus")
    else:
        rows = read_jsonl(direct_path)
        direct_row_count = len(rows)
        abstain_contamination_count = sum(1 for row in rows if _row_is_abstain(row))
        if abstain_contamination_count:
            failures.append("direct_corpus_has_abstain")
    if not abstain_path.exists():
        failures.append("missing_abstain_policy_path")
    if train and not kaggle_mode:
        failures.append("not_kaggle_train_mode")
    if not dry_run and not train and not skip_train_if_adapter_exists:
        failures.append("not_kaggle_train_mode")
    forbidden = _forbidden_submission_artifacts(REPO_ROOT)
    if forbidden:
        failures.append("forbidden_submission_artifact_found")
    reuse_adapter = False
    adapter_validation = {}
    if skip_train_if_adapter_exists:
        candidate = Path(adapter_dir) if adapter_dir else Path(str(config.get("output_adapter_dir", "")))
        adapter_validation = validate_saved_adapter(candidate) if candidate.exists() else {"status": "FAIL", "failures": ["adapter_dir_missing"]}
        if adapter_validation.get("status") == "PASS":
            reuse_adapter = True
        elif train:
            warnings.append("skip_train_if_adapter_exists_requested_but_adapter_invalid")
    status = "PASS" if not failures else "FAIL"
    if reuse_adapter:
        status = "PASS_REUSED_ADAPTER" if not failures else "FAIL"
    return {
        "status": status,
        "stage": config.get("stage"),
        "config_path": str(config_path),
        "config": config,
        "dry_run": bool(dry_run),
        "trained": False,
        "reused_adapter": reuse_adapter,
        "adapter_dir": str(adapter_dir or config.get("output_adapter_dir", "")),
        "adapter_validation": adapter_validation,
        "train_requested": bool(train),
        "kaggle_mode": bool(kaggle_mode),
        "direct_corpus": file_record(direct_path, row_count=direct_row_count) if direct_path.exists() else {"path": str(direct_path), "exists": False},
        "abstain_policy": file_record(abstain_path) if abstain_path.exists() else {"path": str(abstain_path), "exists": False},
        "direct_abstain_rows": abstain_contamination_count,
        "teacher_audit_status": teacher_audit.get("status") if teacher_audit else None,
        "overlap_audit_status": overlap_audit.get("status") if overlap_audit else None,
        "learnability_audit_status": learnability_audit.get("status") if learnability_audit else None,
        "mix_repair_status": mix_repair.get("status") if mix_repair else None,
        "capacity_audit_status": capacity_audit.get("status") if capacity_audit else None,
        "num_steps": int(config.get("num_steps", 0) or 0),
        "target_modules": _target_modules(config),
        "load_in_4bit": bool(config.get("load_in_4bit", False)),
        "memory_snapshots": [],
        "training_log_history_path": "",
        "run_provenance_path": "",
        "no_submission_created": True,
        "packaging_allowed": False,
        "submission_allowed": False,
        "submit_recommended": False,
        "leaderboard_claim": False,
        "no_0_93_evidence": True,
        "no_0_95_evidence": True,
        "real_gpu_training_status": "REAL_GPU_TRAINING_NOT_RUN_IN_CODEX" if not train else "KAGGLE_TRAINING_REQUESTED",
        "warnings": warnings,
        "failures": failures,
    }


def run_day11a_train_backend(
    *,
    config_path: str | Path,
    summary: dict[str, Any],
    out_summary_path: str | Path,
    adapter_dir: str | Path | None = None,
) -> dict[str, Any]:
    config = dict(summary["config"])
    config["train_teacher_path"] = str(config.get("teacher_corpus_path", config.get("train_teacher_path", "")))
    output_dir = Path(adapter_dir) if adapter_dir else build_day11a_output_dir(config, config_path=config_path)
    memory_snapshots = [capture_gpu_memory_snapshot("day11a_before_tokenizer_load", kaggle_mode=True)]
    failures: list[str] = []
    warnings: list[str] = []
    train_report: dict[str, Any] = {}
    validation: dict[str, Any] = {"status": "FAIL", "adapter_config_exists": False, "adapter_model_exists": False, "failures": ["adapter_not_validated"]}
    dataset_report: dict[str, Any] = {}
    tokenizer = model = lora_model = None
    try:
        tokenizer = load_tokenizer(str(config["base_model_path"]))
        sft_items = load_day11a_teacher_sft_items(config)
        dataset_report = build_sft_dataset_report(config, sft_items)
        if dataset_report.get("status") != "PASS":
            failures.extend(dataset_report.get("failures", ["dataset_report_not_pass"]))
            raise RuntimeError("dataset_report_not_pass")
        dataset = Sprint11SFTDataset(sft_items, tokenizer, int(config.get("max_seq_len", 1024)), prevalidate=True)
        memory_snapshots.append(capture_gpu_memory_snapshot("day11a_before_model_load", kaggle_mode=True))
        model = load_base_model(str(config["base_model_path"]), load_in_4bit=False, bf16=_resolve_bf16(config))
        memory_snapshots.append(capture_gpu_memory_snapshot("day11a_after_model_load", kaggle_mode=True))
        lora_model = prepare_model_for_v2a_training(model, config)
        memory_snapshots.append(capture_gpu_memory_snapshot("day11a_after_lora_prepare", kaggle_mode=True))
        train_report = run_lora_training(lora_model, tokenizer, dataset, config, output_dir)
        validation = validate_saved_adapter(output_dir)
        memory_snapshots.append(capture_gpu_memory_snapshot("day11a_after_train", kaggle_mode=True))
    except Exception as exc:
        failures.append(f"train_failed:{type(exc).__name__}:{exc}")
    finally:
        tokenizer = model = lora_model = None
        cuda_empty_cache("day11a_train_cleanup")
        memory_snapshots.append(capture_gpu_memory_snapshot("day11a_after_cleanup", kaggle_mode=True))
    steps_completed = int(train_report.get("steps_completed", 0) or 0)
    expected_steps = int(config.get("num_steps", 0) or 0)
    if train_report and train_report.get("status") != "PASS":
        failures.append("train_failed:backend_status_not_pass")
    if validation.get("status") != "PASS":
        failures.extend(validation.get("failures", ["adapter_not_validated"]))
    if steps_completed != expected_steps:
        failures.append("smoke_or_train_steps_incomplete")
    if bool(train_report.get("loss_nan_detected", False)) or bool(train_report.get("grad_nan_detected", False)):
        failures.append("train_failed:nan_detected")
    log_history_path = _sidecar_path(out_summary_path, "training_log_history")
    write_json_checked(log_history_path, build_training_log_history(train_report), field_name="day11a_training_log_history")
    provenance_path = _sidecar_path(out_summary_path, "run_provenance")
    write_json_checked(provenance_path, build_day11a_run_provenance(config_path, config, output_dir=output_dir), field_name="day11a_run_provenance")
    backend_status = "PASS" if not failures else "FAIL"
    return {
        "backend_status": backend_status,
        "status": backend_status,
        "trained": backend_status == "PASS",
        "steps_completed": steps_completed,
        "intended_steps": expected_steps,
        "loss_start": train_report.get("loss_start"),
        "loss_end": train_report.get("loss_end"),
        "loss_min": train_report.get("loss_min"),
        "loss_max": train_report.get("loss_max"),
        "loss_nan_detected": bool(train_report.get("loss_nan_detected", False)),
        "grad_nan_detected": bool(train_report.get("grad_nan_detected", False)),
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
        "packaging_allowed": False,
        "submission_allowed": False,
        "submit_recommended": False,
        "no_submission_created": True,
        "leaderboard_claim": False,
        "no_0_93_evidence": True,
        "no_0_95_evidence": True,
    }


def load_day11a_teacher_sft_items(config: dict[str, Any]) -> list[SFTItem]:
    path = Path(str(config.get("teacher_corpus_path", config.get("train_teacher_path", ""))))
    items: list[SFTItem] = []
    for row in read_jsonl(path):
        adapted = _adapt_day10_teacher_row_for_sft(row)
        if _row_is_abstain(adapted):
            raise ValueError(f"abstain_row_in_direct_teacher_corpus:{adapted.get('id')}")
        items.append(SFTItem(row=adapted, source="direct_answer", sampling_weight=float(config.get("teacher_direct_weight", 1.0))))
    return items


def build_manifest(config_path: str | Path, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_by": "DAY11A_REAL_V2_CANDIDATE",
        "stage": summary.get("stage"),
        "config": file_record(config_path),
        "direct_corpus": summary.get("direct_corpus", {}),
        "abstain_policy": summary.get("abstain_policy", {}),
        "adapter_dir": summary.get("adapter_dir", ""),
        "trained": summary.get("trained", False),
        "reused_adapter": summary.get("reused_adapter", False),
        "steps_completed": summary.get("steps_completed", 0),
        "intended_steps": summary.get("intended_steps", summary.get("num_steps", 0)),
        "training_log_history_path": summary.get("training_log_history_path", ""),
        "run_provenance_path": summary.get("run_provenance_path", ""),
        "packaging_allowed": False,
        "submission_allowed": False,
        "submit_recommended": False,
        "no_submission_created": True,
        "leaderboard_claim": False,
        "no_0_95_evidence": True,
    }


def build_day11a_output_dir(config: dict[str, Any], *, config_path: str | Path) -> Path:
    root = Path("/kaggle/working/anti086_adapters")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    seed = json.dumps(
        {
            "stage": config.get("stage"),
            "config": file_record(config_path).get("sha256"),
            "corpus": file_record(config.get("teacher_corpus_path", config.get("train_teacher_path", ""))).get("sha256"),
            "steps": config.get("num_steps"),
            "created": stamp,
        },
        sort_keys=True,
    )
    short_hash = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:8]
    output_dir = root / f"v2a_bf16_qv_50_{stamp}_{short_hash}"
    require_writable_output_dir(output_dir, field_name="output_adapter_dir")
    if output_dir.exists():
        raise FileExistsError(f"refusing_to_overwrite_adapter_dir:{output_dir}")
    return output_dir


def build_day11a_run_provenance(config_path: str | Path, config: dict[str, Any], *, output_dir: str | Path) -> dict[str, Any]:
    return {
        "status": "PASS",
        "created_by": "DAY11A_REAL_V2_CANDIDATE",
        "config": file_record(config_path),
        "teacher_corpus": file_record(config.get("teacher_corpus_path", config.get("train_teacher_path", ""))),
        "output_adapter_dir": str(output_dir),
        "stage": config.get("stage"),
        "rank": int(config.get("rank", 0) or 0),
        "target_modules": _target_modules(config),
        "load_in_4bit": bool(config.get("load_in_4bit", False)),
        "bf16": config.get("bf16"),
        "num_steps": int(config.get("num_steps", 0) or 0),
        "packaging_allowed": False,
        "submission_allowed": False,
        "submit_recommended": False,
        "no_leaderboard_evidence": True,
        "no_0_95_evidence": True,
    }


def _read_required_json(path: str | Path, failures: list[str], name: str) -> dict[str, Any]:
    try:
        return read_json(path)
    except Exception as exc:
        failures.append(f"missing_audit:{name}:{type(exc).__name__}")
        return {}


def _row_is_abstain(row: dict[str, Any]) -> bool:
    return str(row.get("answer", "")).strip().upper() == "ABSTAIN" or str(row.get("expected_behavior", "")).lower() == "abstain"


def _forbidden_submission_artifacts(root: Path) -> list[str]:
    forbidden = []
    for name in ("submission.zip", "adapter_model.safetensors"):
        candidate = root / name
        if candidate.exists():
            forbidden.append(str(candidate))
    return forbidden


def _sidecar_path(path: str | Path, suffix: str) -> Path:
    target = Path(path)
    return target.with_name(f"{target.stem}_{suffix}{target.suffix}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Day 11A BF16 Q/V v2 candidate train/dry-run gate.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--teacher-audit", required=True)
    parser.add_argument("--overlap-audit", required=True)
    parser.add_argument("--learnability-audit", required=True)
    parser.add_argument("--mix-repair-report", required=True)
    parser.add_argument("--capacity-audit", required=True)
    parser.add_argument("--out-summary", required=True)
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--kaggle-mode", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--skip-train-if-adapter-exists", action="store_true")
    parser.add_argument("--no-submit", action="store_true", default=True)
    args = parser.parse_args(argv)
    summary = build_day11a_readiness(
        config_path=args.config,
        teacher_audit_path=args.teacher_audit,
        overlap_audit_path=args.overlap_audit,
        learnability_audit_path=args.learnability_audit,
        mix_repair_report_path=args.mix_repair_report,
        capacity_audit_path=args.capacity_audit,
        adapter_dir=args.adapter_dir,
        dry_run=args.dry_run,
        kaggle_mode=args.kaggle_mode,
        train=args.train,
        skip_train_if_adapter_exists=args.skip_train_if_adapter_exists,
    )
    if args.train and summary["status"] == "PASS" and not summary.get("reused_adapter"):
        backend = run_day11a_train_backend(config_path=args.config, summary=summary, out_summary_path=args.out_summary, adapter_dir=args.adapter_dir)
        summary.update(backend)
    manifest = build_manifest(args.config, summary)
    write_json_checked(args.out_summary, summary, field_name="day11a_train_summary")
    write_json_checked(args.out_manifest, manifest, field_name="day11a_train_manifest")
    print(json.dumps({"status": summary["status"], "trained": summary.get("trained", False), "submit_recommended": False}, sort_keys=True))
    return 0 if summary["status"] in {"PASS", "PASS_REUSED_ADAPTER"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
