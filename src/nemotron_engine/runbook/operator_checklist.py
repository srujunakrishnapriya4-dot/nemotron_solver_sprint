"""Human operator checklist contracts for Pass 15."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping

from nemotron_engine.core.schemas import stable_hash

from .runbook_config import RunbookConfig, reject_forbidden_runbook_claims, validate_runbook_config


class OperatorChecklistError(ValueError):
    """Raised when an operator checklist is inconsistent."""


REQUIRED_CHECKLIST_DESCRIPTIONS = (
    "Pass 14 lock audit returned LOCK.",
    "Pass 14 checkpoint ZIP exists.",
    "Scoped full suite baseline recorded.",
    "Final rehearsal manifest hash recorded.",
    "Artifact audit has no forbidden artifacts.",
    "No Kaggle/API submission was performed by automation.",
    "Human understands dry-run readiness is not leaderboard readiness.",
    "External submission, if performed, is manual only.",
)


@dataclass(frozen=True)
class ChecklistItem:
    item_id: str
    description: str
    required: bool
    checked: bool
    evidence_hash: str | None
    blocker_if_unchecked: bool
    metadata: dict[str, Any] = field(default_factory=dict)
    item_hash: str = ""

    def __post_init__(self) -> None:
        item_id = _require_non_empty(self.item_id, "item_id")
        description = _require_non_empty(self.description, "description")
        for name in ("required", "checked", "blocker_if_unchecked"):
            if not isinstance(getattr(self, name), bool):
                raise OperatorChecklistError(f"{name} must be boolean.")
        if self.evidence_hash is not None:
            _require_non_empty(self.evidence_hash, "evidence_hash")
        metadata = dict(self.metadata)
        reject_forbidden_runbook_claims(metadata, error_cls=OperatorChecklistError)
        if self.required and self.checked and self.evidence_hash is None and metadata.get("informational") is not True:
            raise OperatorChecklistError("checked required item requires evidence_hash unless informational.")
        object.__setattr__(self, "item_id", item_id)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "metadata", metadata)
        expected = compute_checklist_item_hash(self)
        if not self.item_hash:
            object.__setattr__(self, "item_hash", expected)
        elif self.item_hash != expected:
            raise OperatorChecklistError("item_hash does not match checklist item payload.")


@dataclass(frozen=True)
class OperatorChecklist:
    items: tuple[ChecklistItem, ...]
    runbook_config_hash: str
    all_required_checked: bool
    blockers: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    checklist_hash: str = ""

    def __post_init__(self) -> None:
        items = tuple(_item(item) for item in self.items)
        if not items:
            raise OperatorChecklistError("operator checklist requires items.")
        ids = tuple(item.item_id for item in items)
        if len(ids) != len(set(ids)):
            raise OperatorChecklistError("checklist item ids must be unique.")
        _require_non_empty(self.runbook_config_hash, "runbook_config_hash")
        if not isinstance(self.all_required_checked, bool):
            raise OperatorChecklistError("all_required_checked must be boolean.")
        expected_blockers = tuple(sorted(item.item_id for item in items if item.required and item.blocker_if_unchecked and not item.checked))
        expected_all = all(item.checked for item in items if item.required)
        blockers = tuple(sorted(str(item) for item in self.blockers))
        if self.all_required_checked != expected_all:
            raise OperatorChecklistError("all_required_checked does not match required item states.")
        if blockers != expected_blockers:
            raise OperatorChecklistError("blockers do not match unchecked required blocker items.")
        required_descriptions = {item.description for item in items if item.required}
        missing = tuple(description for description in REQUIRED_CHECKLIST_DESCRIPTIONS if description not in required_descriptions)
        if missing:
            raise OperatorChecklistError(f"operator checklist missing required items: {missing}")
        for item in items:
            if item.item_hash != compute_checklist_item_hash(item):
                raise OperatorChecklistError("item_hash does not match checklist item payload.")
        metadata = dict(self.metadata)
        reject_forbidden_runbook_claims(metadata, error_cls=OperatorChecklistError)
        object.__setattr__(self, "items", tuple(sorted(items, key=lambda item: item.item_id)))
        object.__setattr__(self, "blockers", blockers)
        object.__setattr__(self, "metadata", metadata)
        expected = compute_operator_checklist_hash(self)
        if not self.checklist_hash:
            object.__setattr__(self, "checklist_hash", expected)
        elif self.checklist_hash != expected:
            raise OperatorChecklistError("checklist_hash does not match operator checklist payload.")


def build_operator_checklist(
    config: RunbookConfig | Mapping[str, Any],
    *,
    checked: bool = False,
    evidence_hashes: Mapping[str, str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> OperatorChecklist:
    cfg = validate_runbook_config(config)
    evidence = dict(evidence_hashes or {})
    items: list[ChecklistItem] = []
    for index, description in enumerate(REQUIRED_CHECKLIST_DESCRIPTIONS, start=1):
        item_id = f"pass15-check-{index:02d}"
        item_evidence = evidence.get(item_id) or evidence.get(description)
        if checked and item_evidence is None:
            item_evidence = cfg.final_rehearsal_manifest_hash or cfg.config_hash
        items.append(
            ChecklistItem(
                item_id=item_id,
                description=description,
                required=True,
                checked=checked,
                evidence_hash=item_evidence if checked else None,
                blocker_if_unchecked=True,
                metadata={},
            )
        )
    return OperatorChecklist(
        items=tuple(items),
        runbook_config_hash=cfg.config_hash,
        all_required_checked=all(item.checked for item in items if item.required),
        blockers=tuple(sorted(item.item_id for item in items if item.required and item.blocker_if_unchecked and not item.checked)),
        metadata=dict(metadata or {}),
    )


def validate_operator_checklist(checklist: OperatorChecklist | Mapping[str, Any], config: RunbookConfig | Mapping[str, Any] | None = None) -> OperatorChecklist:
    cfg = validate_runbook_config(config) if config is not None else None
    normalized = checklist if isinstance(checklist, OperatorChecklist) else OperatorChecklist(**dict(checklist))
    if normalized.checklist_hash != compute_operator_checklist_hash(normalized):
        raise OperatorChecklistError("checklist_hash does not match operator checklist payload.")
    if cfg is not None and normalized.runbook_config_hash != cfg.config_hash:
        raise OperatorChecklistError("operator checklist runbook_config_hash mismatch.")
    return normalized


def compute_operator_checklist_hash(checklist: OperatorChecklist | Mapping[str, Any]) -> str:
    payload = dict(checklist) if isinstance(checklist, Mapping) else {item.name: getattr(checklist, item.name) for item in fields(OperatorChecklist)}
    payload.pop("checklist_hash", None)
    return stable_hash(payload)


def compute_checklist_item_hash(item: ChecklistItem | Mapping[str, Any]) -> str:
    payload = dict(item) if isinstance(item, Mapping) else {field_item.name: getattr(item, field_item.name) for field_item in fields(ChecklistItem)}
    payload.pop("item_hash", None)
    return stable_hash(payload)


def _item(value: ChecklistItem | Mapping[str, Any]) -> ChecklistItem:
    return value if isinstance(value, ChecklistItem) else ChecklistItem(**dict(value))


def _require_non_empty(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OperatorChecklistError(f"{field_name} must be non-empty.")
    return value.strip()


__all__ = [
    "ChecklistItem",
    "OperatorChecklist",
    "OperatorChecklistError",
    "REQUIRED_CHECKLIST_DESCRIPTIONS",
    "build_operator_checklist",
    "compute_checklist_item_hash",
    "compute_operator_checklist_hash",
    "validate_operator_checklist",
]
