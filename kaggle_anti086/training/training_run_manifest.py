from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone
from typing import Any

from kaggle_anti086.data.v2_corpus_io import file_record, read_json
from kaggle_anti086.kaggle_path_safety import require_writable_output_dir
from kaggle_anti086.training.training_config_schema import _target_modules


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def build_run_manifest(config: dict[str, Any], *, config_path: str | Path, collator_audit_path: str | Path, output_root: str | Path | None = None) -> dict[str, Any]:
    stage = str(config.get("stage"))
    if stage != "v2a_base_lora":
        raise ValueError("Day 8 training manifest only supports v2a_base_lora")
    seed = json.dumps({"stage": stage, "config": file_record(config_path), "audit": file_record(collator_audit_path)}, sort_keys=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + sha256_text(seed)[:8]
    base_output = Path(output_root) if output_root is not None else Path(str(config["output_adapter_dir"])).parent
    output_adapter_dir = base_output / f"v2a_{run_id}"
    require_writable_output_dir(output_adapter_dir, field_name="output_adapter_dir")
    if output_adapter_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing adapter dir: {output_adapter_dir}")
    sampling_policy = Path("artifacts/sprint11/train_v2_sampling_policy.json")
    return {
        "run_id": run_id,
        "stage": stage,
        "config_path": str(config_path),
        "config_sha256": file_record(config_path).get("sha256"),
        "train_direct_path": str(config["train_direct_path"]),
        "train_direct_sha256": file_record(config["train_direct_path"]).get("sha256"),
        "train_solver_corrected_path": str(config["train_solver_corrected_path"]),
        "train_solver_corrected_sha256": file_record(config["train_solver_corrected_path"]).get("sha256"),
        "sampling_policy_report_sha256": file_record(sampling_policy).get("sha256") if sampling_policy.exists() else None,
        "real_collator_audit_sha256": file_record(collator_audit_path).get("sha256"),
        "output_adapter_dir": str(output_adapter_dir),
        "rank": int(config["rank"]),
        "target_modules": _target_modules(config),
        "learning_rate": float(config["learning_rate"]),
        "num_steps": int(config["num_steps"]),
        "assistant_only_loss": bool(config["assistant_only_loss"]),
        "full_prompt_loss": bool(config["full_prompt_loss"]),
        "train_on_user": bool(config["train_on_user"]),
        "training_allowed_today": True,
        "packaging_allowed": False,
        "submission_allowed": False,
        "git_commit": git_commit(),
        "gpu_visible": False,
        "base_model_path": str(config.get("base_model_path")),
        "parent_adapter_path": str(config.get("parent_adapter_path")),
        "collator_audit_status": read_json(collator_audit_path).get("status") if Path(collator_audit_path).exists() else "MISSING",
    }
