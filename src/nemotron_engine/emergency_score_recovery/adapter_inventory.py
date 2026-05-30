from __future__ import annotations

from dataclasses import dataclass, fields
import json
from pathlib import Path
from typing import Any

from nemotron_engine.core.schemas import stable_hash


class AdapterInventoryError(ValueError):
    """Raised when adapter inventory cannot be inspected safely."""


REQUIRED_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
FORBIDDEN_SUFFIXES = (".csv", ".jsonl", ".pt", ".pth", ".ckpt")
FORBIDDEN_NAMES = {"optimizer.pt", "scheduler.pt", "trainer_state.json", "training_args.bin", "pytorch_model.bin"}
KNOWN_GOOD_PARENT_HINTS = ("huikang", "nemotron-adapter", "transformers/default/20")


@dataclass(frozen=True)
class AdapterRecord:
    adapter_path: str
    adapter_size_mb: float
    rank: int | None
    target_modules: tuple[str, ...]
    modules_to_save: tuple[str, ...]
    base_model_name_or_path: str | None
    resembles_known_good_parent: bool
    huge_suspicious: bool
    packageable: bool
    package_rejection_reasons: tuple[str, ...] = ()
    inventory_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "adapter_path", str(self.adapter_path))
        object.__setattr__(self, "adapter_size_mb", round(float(self.adapter_size_mb), 6))
        object.__setattr__(self, "target_modules", tuple(str(item) for item in self.target_modules))
        object.__setattr__(self, "modules_to_save", tuple(str(item) for item in self.modules_to_save))
        object.__setattr__(self, "package_rejection_reasons", tuple(str(item) for item in self.package_rejection_reasons))
        _set_or_check_hash(self, "inventory_hash")


def find_adapter_dirs(root: str | Path) -> tuple[Path, ...]:
    root_path = Path(root)
    if not root_path.exists():
        return ()
    matches: list[Path] = []
    for config in root_path.rglob("adapter_config.json"):
        candidate = config.parent
        if (candidate / "adapter_model.safetensors").is_file():
            matches.append(candidate)
    return tuple(sorted(set(matches), key=lambda path: path.as_posix()))


def scan_adapter_roots(roots: list[str | Path] | tuple[str | Path, ...]) -> tuple[AdapterRecord, ...]:
    records: list[AdapterRecord] = []
    for root in roots:
        for adapter_dir in find_adapter_dirs(root):
            records.append(inspect_adapter_dir(adapter_dir))
    return tuple(sorted(records, key=lambda item: item.adapter_path))


def inspect_adapter_dir(adapter_dir: str | Path, *, known_good_parent_hints: tuple[str, ...] = KNOWN_GOOD_PARENT_HINTS) -> AdapterRecord:
    root = Path(adapter_dir)
    config_path = root / "adapter_config.json"
    model_path = root / "adapter_model.safetensors"
    reasons: list[str] = []
    if not config_path.is_file():
        reasons.append("missing_adapter_config")
        payload: dict[str, Any] = {}
    else:
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                reasons.append("adapter_config_not_object")
                payload = {}
        except Exception:
            reasons.append("invalid_adapter_config_json")
            payload = {}
    if not model_path.is_file():
        reasons.append("missing_adapter_model")
        size_mb = 0.0
    else:
        size_mb = model_path.stat().st_size / (1024 * 1024)
    rank = extract_rank(payload)
    if rank is not None and rank > 32:
        reasons.append("rank_gt_32")
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if name in FORBIDDEN_NAMES or (name.endswith(FORBIDDEN_SUFFIXES) and name not in REQUIRED_ADAPTER_FILES):
            reasons.append(f"forbidden_file:{path.relative_to(root).as_posix()}")
    target_modules = _tuple_text(payload.get("target_modules"))
    modules_to_save = _tuple_text(payload.get("modules_to_save"))
    base_model = payload.get("base_model_name_or_path")
    path_text = root.as_posix().lower()
    resembles_parent = any(hint.lower() in path_text for hint in known_good_parent_hints)
    huge = size_mb > 1500 and not resembles_parent
    if huge:
        reasons.append("huge_custom_adapter")
    return AdapterRecord(
        adapter_path=str(root),
        adapter_size_mb=size_mb,
        rank=rank,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        base_model_name_or_path=str(base_model) if base_model else None,
        resembles_known_good_parent=resembles_parent,
        huge_suspicious=huge,
        packageable=not reasons,
        package_rejection_reasons=tuple(sorted(set(reasons))),
    )


def extract_rank(payload: dict[str, Any]) -> int | None:
    for key in ("r", "rank", "lora_rank"):
        if key in payload and payload[key] is not None:
            return int(payload[key])
    peft = payload.get("peft_config")
    if isinstance(peft, dict) and peft.get("r") is not None:
        return int(peft["r"])
    return None


def _tuple_text(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple, set)):
        return tuple(sorted(str(item) for item in value))
    return (str(value),)


def _set_or_check_hash(instance: object, hash_field: str) -> None:
    expected = stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})
    current = getattr(instance, hash_field)
    if not current:
        object.__setattr__(instance, hash_field, expected)
    elif current != expected:
        raise AdapterInventoryError(f"{hash_field} does not match payload.")
