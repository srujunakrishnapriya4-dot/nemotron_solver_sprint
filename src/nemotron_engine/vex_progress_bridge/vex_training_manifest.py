from __future__ import annotations

from dataclasses import dataclass, fields

from nemotron_engine.core.schemas import stable_hash


class VexTrainingManifestError(ValueError):
    pass


@dataclass(frozen=True)
class VexTrainingManifest:
    token_manifest_hash: str
    config_hash: str
    row_count: int
    supervised_token_count: int
    target_modules: tuple[str, ...]
    adapter_source: str
    full_prompt_loss: bool = False
    manifest_hash: str = ""

    def __post_init__(self) -> None:
        if self.row_count <= 0:
            raise VexTrainingManifestError("row_count must be positive")
        if self.supervised_token_count <= 0:
            raise VexTrainingManifestError("supervised_token_count must be positive")
        if self.full_prompt_loss:
            raise VexTrainingManifestError("full prompt loss is forbidden")
        object.__setattr__(self, "target_modules", tuple(self.target_modules))
        _set_or_check_hash(self, "manifest_hash")


def _set_or_check_hash(instance: object, hash_field: str) -> None:
    expected = stable_hash({field.name: getattr(instance, field.name) for field in fields(instance) if field.name != hash_field})
    current = getattr(instance, hash_field)
    if not current:
        object.__setattr__(instance, hash_field, expected)
    elif current != expected:
        raise VexTrainingManifestError(f"{hash_field} does not match payload")
