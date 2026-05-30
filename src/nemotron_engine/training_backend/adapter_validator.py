"""Real LoRA adapter output validation for Pass 11."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import fnmatch
import json
import os
from pathlib import Path
from typing import Any, Mapping

from nemotron_engine.core.schemas import stable_hash
from nemotron_engine.packaging.submission_manifest import compute_file_sha256
from nemotron_engine.training.lora_config import LoRAConfigError, validate_adapter_config_json

from .backend_contracts import _reject_forbidden_claims


class AdapterValidationError(ValueError):
    """Raised when adapter validation contracts are inconsistent."""


ADAPTER_MODEL_FILES = ("adapter_model.safetensors", "adapter_model.bin")
FORBIDDEN_PATTERNS = (
    "pytorch_model.bin",
    "pytorch_model*",
    "model.safetensors",
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "rng_state*.pth",
    "sft*.jsonl",
    "dpo*.jsonl",
    "train*.jsonl",
    "kaggle.json",
    ".env",
    "token*",
    "credentials*",
    "api_key*",
)
DEFAULT_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
)


@dataclass(frozen=True)
class AdapterValidationConfig:
    adapter_dir: str | Path
    allow_tokenizer_files: bool = False
    allow_symlinks: bool = False
    allow_multiple_adapter_model_files: bool = False
    max_rank: int = 32
    allowed_tokenizer_filenames: tuple[str, ...] = DEFAULT_TOKENIZER_FILES
    metadata: Mapping[str, Any] = field(default_factory=dict)
    config_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "adapter_dir", str(self.adapter_dir))
        for name in ("allow_tokenizer_files", "allow_symlinks", "allow_multiple_adapter_model_files"):
            if not isinstance(getattr(self, name), bool):
                raise AdapterValidationError(f"{name} must be boolean.")
        if isinstance(self.max_rank, bool) or not isinstance(self.max_rank, int) or self.max_rank < 1:
            raise AdapterValidationError("max_rank must be a positive integer.")
        names = tuple(str(item) for item in self.allowed_tokenizer_filenames)
        if any(not item.strip() or "/" in item or "\\" in item for item in names):
            raise AdapterValidationError("allowed tokenizer filenames must be simple non-empty filenames.")
        metadata = dict(self.metadata)
        _reject_forbidden_claims(metadata)
        object.__setattr__(self, "allowed_tokenizer_filenames", names)
        object.__setattr__(self, "metadata", metadata)
        expected = _payload_hash(self, "config_hash")
        if not self.config_hash:
            object.__setattr__(self, "config_hash", expected)
        elif self.config_hash != expected:
            raise AdapterValidationError("config_hash does not match adapter validation config payload.")


@dataclass(frozen=True)
class AdapterValidationReport:
    adapter_dir: str
    passed: bool
    adapter_config_valid: bool
    model_file_present: bool
    adapter_model_files: tuple[str, ...] = ()
    artifact_hashes: Mapping[str, str] = field(default_factory=dict)
    adapter_hash: str | None = None
    rank: int | None = None
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    report_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("passed", "adapter_config_valid", "model_file_present"):
            if not isinstance(getattr(self, name), bool):
                raise AdapterValidationError(f"{name} must be boolean.")
        if not isinstance(self.adapter_dir, str) or not self.adapter_dir.strip():
            raise AdapterValidationError("adapter_dir must be non-empty.")
        if self.rank is not None and (isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 1 or self.rank > 32):
            raise AdapterValidationError("rank must be in [1, 32].")
        models = tuple(str(item) for item in self.adapter_model_files)
        hashes = {str(key): str(value) for key, value in sorted(self.artifact_hashes.items())}
        if any(not _is_sha256(value) for value in hashes.values()):
            raise AdapterValidationError("artifact hashes must be sha256 digests.")
        errors = tuple(str(item) for item in self.errors)
        warnings = tuple(str(item) for item in self.warnings)
        if self.passed:
            if errors:
                raise AdapterValidationError("passed=True cannot include errors.")
            if warnings:
                raise AdapterValidationError("passed=True cannot include warnings.")
            if self.adapter_config_valid is not True or self.model_file_present is not True:
                raise AdapterValidationError("passed=True requires config and model file.")
            if not hashes:
                raise AdapterValidationError("passed=True requires artifact_hashes.")
            if not models:
                raise AdapterValidationError("passed=True requires adapter_model_files.")
            if not self.adapter_hash:
                raise AdapterValidationError("passed=True requires adapter_hash.")
            expected_adapter_hash = stable_hash({"artifact_hashes": hashes, "rank": self.rank})
            if self.adapter_hash != expected_adapter_hash:
                raise AdapterValidationError("adapter_hash does not match adapter artifact payload.")
        if self.adapter_hash is not None and not isinstance(self.adapter_hash, str):
            raise AdapterValidationError("adapter_hash must be a string or None.")
        object.__setattr__(self, "adapter_model_files", models)
        object.__setattr__(self, "artifact_hashes", hashes)
        object.__setattr__(self, "errors", errors)
        object.__setattr__(self, "warnings", warnings)
        expected = _payload_hash(self, "report_hash")
        if not self.report_hash:
            object.__setattr__(self, "report_hash", expected)
        elif self.report_hash != expected:
            raise AdapterValidationError("report_hash does not match adapter validation report payload.")


def validate_adapter_output(config_or_path: AdapterValidationConfig | str | Path) -> AdapterValidationReport:
    config = config_or_path if isinstance(config_or_path, AdapterValidationConfig) else AdapterValidationConfig(config_or_path)
    root = Path(config.adapter_dir)
    errors: list[str] = []
    warnings: list[str] = []
    rank = None
    config_valid = False
    model_files: tuple[str, ...] = ()
    artifact_hashes: dict[str, str] = {}
    if not root.exists():
        errors.append("adapter_dir missing")
    elif not root.is_dir():
        errors.append("adapter_dir is not a directory")
    else:
        all_paths = tuple(sorted((path for path in root.rglob("*")), key=lambda item: item.as_posix()))
        for path in all_paths:
            rel = path.relative_to(root).as_posix()
            if (path.is_symlink() or _is_reparse_point(path)) and not config.allow_symlinks:
                errors.append(f"symlink or reparse point forbidden: {rel}")
            if path.is_dir():
                continue
            name = path.name
            if _is_forbidden(name):
                errors.append(f"forbidden file: {rel}")
            elif name == "adapter_config.json" or name in ADAPTER_MODEL_FILES:
                pass
            elif config.allow_tokenizer_files and name in config.allowed_tokenizer_filenames:
                pass
            else:
                errors.append(f"unexpected adapter file: {rel}")
        config_path = root / "adapter_config.json"
        if not config_path.exists():
            errors.append("missing adapter_config.json")
        elif not config_path.is_file():
            errors.append("adapter_config.json is not a file")
        else:
            try:
                payload = json.loads(config_path.read_text(encoding="utf-8"))
                validate_adapter_config_file(config_path)
                rank = _extract_rank(payload)
                if rank is not None and rank > config.max_rank:
                    errors.append("adapter rank exceeds max_rank")
                config_valid = True
            except (OSError, json.JSONDecodeError, LoRAConfigError, AdapterValidationError) as exc:
                errors.append(f"invalid adapter_config.json: {exc}")
        model_files = tuple(name for name in ADAPTER_MODEL_FILES if (root / name).is_file())
        if not model_files:
            errors.append("missing adapter model file")
        if len(model_files) > 1 and not config.allow_multiple_adapter_model_files:
            errors.append("more than one adapter model file")
        if not errors:
            artifact_hashes = compute_adapter_artifact_hashes(root)
    adapter_hash = stable_hash({"artifact_hashes": artifact_hashes, "rank": rank}) if artifact_hashes and not errors else None
    return AdapterValidationReport(
        adapter_dir=str(root),
        passed=not errors,
        adapter_config_valid=config_valid and not any("adapter_config" in item for item in errors),
        model_file_present=bool(model_files),
        adapter_model_files=model_files,
        artifact_hashes=artifact_hashes,
        adapter_hash=adapter_hash,
        rank=rank,
        errors=tuple(sorted(set(errors))),
        warnings=tuple(sorted(set(warnings))),
    )


def validate_adapter_config_file(path: str | Path) -> bool:
    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdapterValidationError("adapter_config.json must be valid JSON.") from exc
    if not isinstance(payload, Mapping):
        raise AdapterValidationError("adapter_config.json must contain an object.")
    validate_adapter_config_json(payload)
    rank = _extract_rank(payload)
    if rank is not None and rank > 32:
        raise AdapterValidationError("adapter rank must be <= 32.")
    return True


def compute_adapter_artifact_hashes(adapter_dir: str | Path) -> dict[str, str]:
    root = Path(adapter_dir)
    if not root.exists() or not root.is_dir():
        raise AdapterValidationError("adapter_dir must exist and be a directory.")
    result: dict[str, str] = {}
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.as_posix()):
        rel = path.relative_to(root).as_posix()
        result[rel] = compute_file_sha256(path)
    return result


def _extract_rank(payload: Mapping[str, Any]) -> int | None:
    ranks: list[int] = []
    for key in ("r", "rank", "lora_rank"):
        if key in payload:
            ranks.append(_rank(payload[key], key))
    peft = payload.get("peft_config")
    if isinstance(peft, Mapping) and "r" in peft:
        ranks.append(_rank(peft["r"], "peft_config.r"))
    return max(ranks) if ranks else None


def _rank(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AdapterValidationError(f"{label} must be an integer.")
    if value < 1 or value > 32:
        raise AdapterValidationError(f"{label} must be in [1, 32].")
    return value


def _is_forbidden(name: str) -> bool:
    lower = name.lower()
    return any(fnmatch.fnmatch(lower, pattern) for pattern in FORBIDDEN_PATTERNS)


def _is_reparse_point(path: Path) -> bool:
    if os.name != "nt":
        return False
    try:
        return bool(path.lstat().st_file_attributes & 0x400)
    except (AttributeError, OSError):
        return False


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def _payload_hash(instance: object, hash_field: str) -> str:
    return stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})


__all__ = [
    "AdapterValidationConfig",
    "AdapterValidationError",
    "AdapterValidationReport",
    "compute_adapter_artifact_hashes",
    "validate_adapter_config_file",
    "validate_adapter_output",
]
