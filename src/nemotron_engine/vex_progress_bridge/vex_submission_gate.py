from __future__ import annotations

import json
from pathlib import Path
import zipfile


class VexSubmissionGateError(ValueError):
    pass


FORBIDDEN_FILES = {"optimizer.pt", "scheduler.pt", "trainer_state.json", "training_args.bin", "pytorch_model.bin"}


def evaluate_vex_submission_gate(
    adapter_dir: str | Path,
    *,
    rank: int,
    adapter_size_limit_mb: int = 1500,
    full_prompt_loss: bool = False,
    eval_summary: dict | None = None,
    token_manifest_valid: bool = True,
    failed_lineage: bool = False,
) -> dict:
    reasons: list[str] = []
    root = Path(adapter_dir)
    config = root / "adapter_config.json"
    model = root / "adapter_model.safetensors"
    if rank > 32:
        reasons.append("rank_gt_32")
    if full_prompt_loss:
        reasons.append("full_prompt_loss")
    if failed_lineage:
        reasons.append("failed_sprint4_lineage")
    if not token_manifest_valid:
        reasons.append("invalid_token_manifest")
    if not config.is_file() or not model.is_file():
        reasons.append("missing_adapter_files")
    size_mb = model.stat().st_size / (1024 * 1024) if model.exists() else 0.0
    if size_mb > adapter_size_limit_mb:
        reasons.append("adapter_size_exceeds_limit")
    for path in root.rglob("*") if root.exists() else []:
        if path.is_file() and path.name in FORBIDDEN_FILES:
            reasons.append(f"forbidden_file:{path.name}")
    if not eval_summary:
        reasons.append("missing_eval_evidence")
    else:
        if eval_summary.get("empty_output_count", 0) > 0:
            reasons.append("empty_outputs")
        if eval_summary.get("prompt_copy_count", 0) > 0:
            reasons.append("prompt_copy_outputs")
        if eval_summary.get("exact_answer_format_pass_rate", 0.0) < 0.8:
            reasons.append("answer_format_pass_rate_low")
    return {"decision": "ACCEPT" if not reasons else "REJECT", "reasons": sorted(set(reasons)), "adapter_size_mb": size_mb}


def package_vex_adapter(adapter_dir: str | Path, zip_path: str | Path, manifest_path: str | Path) -> dict:
    root = Path(adapter_dir)
    config = root / "adapter_config.json"
    model = root / "adapter_model.safetensors"
    if not config.is_file() or not model.is_file():
        raise VexSubmissionGateError("missing adapter files")
    target = Path(zip_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(config, "adapter_config.json")
        archive.write(model, "adapter_model.safetensors")
    with zipfile.ZipFile(target) as archive:
        names = sorted(archive.namelist())
    if names != ["adapter_config.json", "adapter_model.safetensors"]:
        raise VexSubmissionGateError("zip contains forbidden files")
    manifest = {"zip_path": str(target), "contents": names, "zip_size_bytes": target.stat().st_size}
    Path(manifest_path).write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
    return manifest
