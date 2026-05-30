from __future__ import annotations

import os
from pathlib import Path
import shutil
import stat
import sys
from typing import Iterable


CUTLASS_PATH = Path("/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/nvidia_cutlass_dsl/python_packages")
PTXAS_BLACKWELL_PATH = Path("/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/triton/backends/nvidia/bin/ptxas-blackwell")
PTXAS_WORKING_PATH = Path("/tmp/ptxas-blackwell")


def patch_cutlass_path() -> dict:
    exists = CUTLASS_PATH.exists()
    if exists and str(CUTLASS_PATH) not in sys.path:
        sys.path.insert(0, str(CUTLASS_PATH))
    return {"cutlass_path": str(CUTLASS_PATH), "cutlass_path_exists": exists, "cutlass_path_inserted": exists}


def patch_ptxas_blackwell() -> dict:
    if not PTXAS_BLACKWELL_PATH.exists():
        return {"ptxas_source": str(PTXAS_BLACKWELL_PATH), "ptxas_source_exists": False, "ptxas_patched": False}
    shutil.copy2(PTXAS_BLACKWELL_PATH, PTXAS_WORKING_PATH)
    PTXAS_WORKING_PATH.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    os.environ["TRITON_PTXAS_PATH"] = str(PTXAS_WORKING_PATH)
    os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = str(PTXAS_WORKING_PATH)
    return {
        "ptxas_source": str(PTXAS_BLACKWELL_PATH),
        "ptxas_working_path": str(PTXAS_WORKING_PATH),
        "ptxas_source_exists": True,
        "ptxas_patched": True,
    }


def apply_runtime_patches() -> dict:
    report = {}
    report.update(patch_cutlass_path())
    report.update(patch_ptxas_blackwell())
    return report


def normalize_optional_path(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "false"}:
        return None
    return text


def require_real_optional_path(value, *, field_name: str) -> str | None:
    text = normalize_optional_path(value)
    if text is None:
        return None
    if text == "auto_or_none":
        raise SystemExit(f"{field_name} must not be auto_or_none")
    if not Path(text).exists():
        raise SystemExit(f"{field_name} does not exist: {text}")
    return text


def discover_parent_adapter(search_roots: Iterable[str | Path] = ("/kaggle/input", "/kaggle/working")) -> str | None:
    candidates: list[Path] = []
    for root in search_roots:
        base = Path(root)
        if not base.exists():
            continue
        for config_path in base.rglob("adapter_config.json"):
            adapter_dir = config_path.parent
            if (adapter_dir / "adapter_model.safetensors").exists():
                candidates.append(adapter_dir)
    if not candidates:
        return None
    def score(path: Path) -> tuple[int, str]:
        text = str(path).lower()
        preferred = any(token in text for token in ("nemotron-adapter", "tinker", "huikang", "parent"))
        return (1 if preferred else 0, str(path))
    return str(sorted(candidates, key=score, reverse=True)[0])


def patch_yaml_value(path: str | Path, key: str, value: str) -> None:
    yaml_path = Path(path)
    lines = yaml_path.read_text(encoding="utf-8").splitlines()
    replaced = False
    out = []
    for line in lines:
        if line.strip().startswith(f"{key}:"):
            out.append(f"{key}: {value}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append(f"{key}: {value}")
    yaml_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def ensure_parent_adapter_in_config(config_path: str | Path, config: dict) -> str:
    current = normalize_optional_path(config.get("parent_adapter_path"))
    if current and current != "auto_or_none":
        require_real_optional_path(current, field_name="parent_adapter_path")
        return current
    discovered = discover_parent_adapter()
    if not discovered:
        raise SystemExit("parent_adapter_path is none and no parent adapter was discovered under /kaggle/input or /kaggle/working")
    patch_yaml_value(config_path, "parent_adapter_path", discovered)
    config["parent_adapter_path"] = discovered
    return discovered
