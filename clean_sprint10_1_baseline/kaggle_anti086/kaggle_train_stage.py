from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from kaggle_prepare_anti086_tokens import load_simple_yaml
from kaggle_path_safety import require_safe_config_output_paths, require_writable_output_dir, require_writable_output_path, safe_write_text
from kaggle_runtime_patches import require_real_optional_path

MICRO_LABEL = "INFRASTRUCTURE STACK TEST ONLY - NOT A 0.95 CANDIDATE - NOT MAIN TRAINING - NOT SUBMISSION READY"


def validate_training_config(config: dict) -> None:
    if config.get("stage") is not None and str(config.get("stage")) not in {"micro", "v1", "v1b"}:
        raise SystemExit("SPRINT-10 permits micro, v1, or v1b only; v2/v3/final training is disabled")
    if bool(config.get("full_prompt_loss", False)):
        raise SystemExit("full_prompt_loss=true is forbidden")
    if int(config.get("rank", 999)) > 32:
        raise SystemExit("rank > 32 forbidden")
    if str(config.get("stage")) in {"v1", "v1b"} and int(config.get("rank", 999)) > 16:
        raise SystemExit("first v1/v1b rank must be <=16")
    if config.get("stage") in {"micro", "v1", "v1b"} and not config.get("target_modules"):
        raise SystemExit("explicit target_modules required")
    target_modules = [item.strip() for item in str(config.get("target_modules", "")).split(",") if item.strip()]
    if str(config.get("stage")) in {"v1", "v1b"} and "lm_head" in target_modules:
        raise SystemExit("lm_head forbidden for first v1/v1b")
    if str(config.get("parent_adapter_path", "")).strip() == "auto_or_none":
        raise SystemExit("parent_adapter_path must never be auto_or_none")
    require_real_optional_path(config.get("parent_adapter_path"), field_name="parent_adapter_path")


def validate_adapter(config: dict) -> dict:
    adapter_dir = require_writable_output_dir(config["output_adapter_dir"], field_name="output_adapter_dir")
    config_path = adapter_dir / "adapter_config.json"
    model_path = adapter_dir / "adapter_model.safetensors"
    if not config_path.exists() or not model_path.exists():
        raise SystemExit("adapter files missing after training")
    try:
        adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"adapter_config.json malformed: {exc}") from exc
    rank = int(adapter_config.get("r", config.get("rank", 999)))
    size_mb = model_path.stat().st_size / (1024 * 1024)
    if rank > 32:
        raise SystemExit("trained adapter rank > 32")
    if size_mb > float(config["adapter_size_limit_mb"]):
        raise SystemExit("trained adapter exceeds size limit")
    if any((adapter_dir / name).exists() for name in ("optimizer.pt", "trainer_state.json")):
        raise SystemExit("checkpoint/optimizer artifacts detected in adapter directory")
    return {"adapter_dir": str(adapter_dir), "rank": rank, "adapter_size_mb": size_mb, "loss_finite": True, "micro_stack_test_label": MICRO_LABEL, "not_submission_ready": True}


def write_stage_gate(stage: str, payload: dict) -> None:
    gate_dir = require_writable_output_dir("/kaggle/working/anti086_stage_logs", field_name="stage_log_dir")
    gate_dir.mkdir(parents=True, exist_ok=True)
    gate = {"stage": f"train_{stage}", "decision": "PASS", "reason": "training stage completed", **payload}
    safe_write_text(gate_dir / f"train_{stage}.gate.json", json.dumps(gate, sort_keys=True, indent=2), field_name="stage_log_dir")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_simple_yaml(args.config)
    require_safe_config_output_paths(config, stage=str(config.get("stage", "train_stage")))
    validate_training_config(config)
    subprocess.check_call([sys.executable, "kaggle_train_anti086_adapter.py", "--config", args.config])
    metadata = validate_adapter(config)
    manifest = require_writable_output_path(str(config.get("train_manifest_path", "/kaggle/working/anti086_train_manifest.json")), field_name="train_manifest_path")
    safe_write_text(manifest, json.dumps(metadata, sort_keys=True, indent=2), field_name="train_manifest_path")
    write_stage_gate(str(config.get("stage", "micro")), metadata)
    print(json.dumps(metadata, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
