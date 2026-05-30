from __future__ import annotations

from dataclasses import dataclass, fields
import json
from pathlib import Path

from nemotron_engine.core.schemas import stable_hash


class AdapterScorebookError(ValueError):
    pass


ALLOWED_STATUSES = {"baseline", "candidate", "rejected", "best_known"}


@dataclass(frozen=True)
class ScorebookEntry:
    adapter_name: str
    adapter_path: str
    submission_zip_hash: str | None
    public_score: float | None
    private_score: float | None
    notebook_version: str
    notes: str
    status: str
    entry_hash: str = ""

    def __post_init__(self) -> None:
        if self.status not in ALLOWED_STATUSES:
            raise AdapterScorebookError(f"invalid status: {self.status}")
        if self.public_score is not None and not 0 <= float(self.public_score) <= 1:
            raise AdapterScorebookError("public_score must be in [0,1]")
        if self.private_score is not None and not 0 <= float(self.private_score) <= 1:
            raise AdapterScorebookError("private_score must be in [0,1]")
        _set_or_check_hash(self, "entry_hash")


def default_scorebook_entries() -> tuple[ScorebookEntry, ...]:
    return (
        ScorebookEntry(
            adapter_name="known_public_parent_direct",
            adapter_path="/kaggle/input/models/huikang/nemotron-adapter/transformers/default/20",
            submission_zip_hash=None,
            public_score=0.85,
            private_score=None,
            notebook_version="public_parent_direct_package",
            notes="Known-good direct adapter package; emergency fallback.",
            status="best_known",
        ),
        ScorebookEntry(
            adapter_name="failed_sprint4_child",
            adapter_path="/kaggle/working/custom_adapter",
            submission_zip_hash=None,
            public_score=0.49,
            private_score=None,
            notebook_version="sprint4_full_prompt_loss",
            notes="Rejected: labels=input_ids full-prompt continuation collapsed behavior.",
            status="rejected",
        ),
    )


def write_scorebook_template(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "entries": [entry.__dict__ for entry in default_scorebook_entries()],
        "scorebook_hash": stable_hash([entry.entry_hash for entry in default_scorebook_entries()]),
    }
    target.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    return target


def load_scorebook(path: str | Path) -> tuple[ScorebookEntry, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise AdapterScorebookError("scorebook entries must be a list")
    return tuple(ScorebookEntry(**entry) for entry in entries)


def _set_or_check_hash(instance: object, hash_field: str) -> None:
    expected = stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})
    current = getattr(instance, hash_field)
    if not current:
        object.__setattr__(instance, hash_field, expected)
    elif current != expected:
        raise AdapterScorebookError(f"{hash_field} does not match payload.")
