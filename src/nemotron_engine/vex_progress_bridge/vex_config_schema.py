from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from nemotron_engine.core.schemas import stable_hash


class VexConfigError(ValueError):
    pass


@dataclass(frozen=True)
class VexConfig:
    mode: str
    variant_name: str
    base_model_path: str
    parent_adapter_path: str | None
    token_corpus_path: str
    output_adapter_dir: str
    rank: int
    lora_alpha: int
    lora_dropout: float
    learning_rate: float
    max_rows: int
    max_seq_len: int
    micro_batch_size: int
    grad_accum: int
    num_steps: int
    adapter_size_limit_mb: int = 1500
    reset_weights: bool = False
    save_embedding_layers: bool = False
    require_vex_private_inputs: bool = False
    public_safe_fallback: bool = False
    synthetic_verified_count: int = 0
    notebook_audit_found: bool = True
    config_hash: str = ""

    def __post_init__(self) -> None:
        if self.mode not in {"public_safe_micro", "vex_replica_if_inputs_available", "public_safe_synthetic"}:
            raise VexConfigError(f"invalid mode: {self.mode}")
        if self.rank <= 0 or self.rank > 32:
            raise VexConfigError("rank must be between 1 and 32")
        if self.mode != "vex_replica_if_inputs_available" and (self.learning_rate <= 0 or self.learning_rate > 0.000001):
            raise VexConfigError("learning_rate must be in (0, 1e-6] for VEX-safe configs")
        if self.mode == "vex_replica_if_inputs_available":
            if not self.require_vex_private_inputs:
                raise VexConfigError("main VEX replica mode must require private/token-swap inputs")
            if not self.notebook_audit_found:
                raise VexConfigError("main VEX replica mode blocked without real notebook audit")
            if not self.reset_weights:
                raise VexConfigError("VEX replica config must explicitly preserve extracted reset_weights=true")
        if self.mode == "public_safe_synthetic" and self.synthetic_verified_count != 0:
            raise VexConfigError("synthetic_verified_count must be 0 until a verified generator exists")
        for name in ("max_rows", "max_seq_len", "micro_batch_size", "grad_accum", "num_steps", "adapter_size_limit_mb"):
            if int(getattr(self, name)) <= 0:
                raise VexConfigError(f"{name} must be positive")
        if self.save_embedding_layers:
            raise VexConfigError("save_embedding_layers is forbidden unless a known-good structure requires it")
        _set_or_check_hash(self, "config_hash")


def load_vex_config(path: str | Path) -> VexConfig:
    payload = _load_simple_yaml(Path(path))
    return VexConfig(
        variant_name=str(payload.get("variant_name", "")),
        mode=str(payload.get("mode", "")),
        base_model_path=str(payload.get("base_model_path", "")),
        parent_adapter_path=_none(payload.get("parent_adapter_path")),
        token_corpus_path=str(payload.get("token_corpus_path", "")),
        output_adapter_dir=str(payload.get("output_adapter_dir", "")),
        rank=int(payload.get("rank", 0)),
        lora_alpha=int(payload.get("lora_alpha", 0)),
        lora_dropout=float(payload.get("lora_dropout", 0.0)),
        learning_rate=float(payload.get("learning_rate", 0.0)),
        max_rows=int(payload.get("max_rows", 0)),
        max_seq_len=int(payload.get("max_seq_len", 0)),
        micro_batch_size=int(payload.get("micro_batch_size", 0)),
        grad_accum=int(payload.get("grad_accum", 0)),
        num_steps=int(payload.get("num_steps", 0)),
        adapter_size_limit_mb=int(payload.get("adapter_size_limit_mb", 1500)),
        reset_weights=bool(payload.get("reset_weights", False)),
        save_embedding_layers=bool(payload.get("save_embedding_layers", False)),
        require_vex_private_inputs=bool(payload.get("require_vex_private_inputs", False)),
        public_safe_fallback=bool(payload.get("public_safe_fallback", False)),
        synthetic_verified_count=int(payload.get("synthetic_verified_count", 0)),
        notebook_audit_found=bool(payload.get("notebook_audit_found", True)),
    )


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        payload[key.strip()] = _scalar(value.strip())
    return payload


def _scalar(value: str) -> Any:
    value = value.strip().strip('"').strip("'")
    if value.lower() in {"none", "null", ""}:
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if any(ch in value.lower() for ch in (".", "e")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _set_or_check_hash(instance: object, hash_field: str) -> None:
    expected = stable_hash({field.name: getattr(instance, field.name) for field in fields(instance) if field.name != hash_field})
    current = getattr(instance, hash_field)
    if not current:
        object.__setattr__(instance, hash_field, expected)
    elif current != expected:
        raise VexConfigError(f"{hash_field} does not match payload")
