from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

from .adapter_inventory import FORBIDDEN_NAMES, FORBIDDEN_SUFFIXES, REQUIRED_ADAPTER_FILES, AdapterInventoryError, extract_rank


class AdapterPackagePlanError(ValueError):
    pass


def validate_existing_adapter_for_package(adapter_dir: str | Path) -> tuple[Path, Path]:
    root = Path(adapter_dir)
    config = root / "adapter_config.json"
    model = root / "adapter_model.safetensors"
    if not config.is_file():
        raise AdapterPackagePlanError(f"missing adapter_config.json: {config}")
    if not model.is_file():
        raise AdapterPackagePlanError(f"missing adapter_model.safetensors: {model}")
    try:
        payload = json.loads(config.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AdapterPackagePlanError(f"invalid adapter_config.json: {exc}") from exc
    if not isinstance(payload, dict):
        raise AdapterPackagePlanError("adapter_config.json must be a JSON object")
    rank = extract_rank(payload)
    if rank is not None and rank > 32:
        raise AdapterPackagePlanError(f"rank_gt_32: {rank}")
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if name in FORBIDDEN_NAMES or (name.endswith(FORBIDDEN_SUFFIXES) and name not in REQUIRED_ADAPTER_FILES):
            raise AdapterPackagePlanError(f"forbidden file in adapter dir: {path.relative_to(root)}")
    return config, model


def package_existing_adapter(adapter_dir: str | Path, zip_path: str | Path) -> dict:
    config, model = validate_existing_adapter_for_package(adapter_dir)
    target = Path(zip_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(config, "adapter_config.json")
        archive.write(model, "adapter_model.safetensors")
    with zipfile.ZipFile(target) as archive:
        names = sorted(archive.namelist())
    if names != ["adapter_config.json", "adapter_model.safetensors"]:
        raise AdapterPackagePlanError(f"unexpected zip contents: {names}")
    return {
        "zip_path": str(target),
        "zip_size_bytes": target.stat().st_size,
        "zip_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "contents": names,
    }
