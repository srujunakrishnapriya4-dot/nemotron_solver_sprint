from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from nemotron_engine.core.schemas import stable_hash

from .adapter_inventory import AdapterRecord
from .adapter_scorebook import ScorebookEntry


class SubmissionDecisionError(ValueError):
    pass


@dataclass(frozen=True)
class SubmissionDecision:
    decision: str
    reasons: tuple[str, ...]
    decision_hash: str = ""

    def __post_init__(self) -> None:
        if self.decision not in {"SUBMIT_BASELINE", "SUBMIT_CANDIDATE", "DO_NOT_SUBMIT"}:
            raise SubmissionDecisionError(f"invalid decision: {self.decision}")
        object.__setattr__(self, "reasons", tuple(str(reason) for reason in self.reasons))
        _set_or_check_hash(self, "decision_hash")


def decide_submission(
    adapter: AdapterRecord,
    *,
    scorebook_entries: tuple[ScorebookEntry, ...] = (),
    probe_evidence: dict[str, Any] | None = None,
    training_metadata: dict[str, Any] | None = None,
    operator_selected: bool = False,
    known_good_baseline: bool = False,
) -> SubmissionDecision:
    reasons: list[str] = []
    probe_evidence = probe_evidence or {}
    training_metadata = training_metadata or {}
    scorebook_status = _scorebook_status(adapter.adapter_path, scorebook_entries)

    if known_good_baseline or adapter.resembles_known_good_parent or scorebook_status in {"baseline", "best_known"}:
        if adapter.rank is not None and adapter.rank > 32:
            return SubmissionDecision("DO_NOT_SUBMIT", ("rank_gt_32",))
        if not adapter.packageable and any(reason != "huge_custom_adapter" for reason in adapter.package_rejection_reasons):
            return SubmissionDecision("DO_NOT_SUBMIT", adapter.package_rejection_reasons)
        return SubmissionDecision("SUBMIT_BASELINE", ("known_good_baseline",))

    if scorebook_status == "rejected":
        reasons.append("scorebook_rejected")
    if not adapter.packageable:
        reasons.extend(adapter.package_rejection_reasons)
    if adapter.rank is not None and adapter.rank > 32:
        reasons.append("rank_gt_32")
    if training_metadata.get("labels_equal_input_ids") or training_metadata.get("loss_mode") == "full_prompt":
        reasons.append("labels_input_ids_failed_branch")
    train_loss = training_metadata.get("train_loss")
    if train_loss is not None and float(train_loss) > 20 and not training_metadata.get("override_high_loss"):
        reasons.append("train_loss_gt_20")
    if adapter.huge_suspicious:
        reasons.append("huge_custom_adapter")
    if not probe_evidence:
        reasons.append("missing_probe_evidence")
    elif not probe_evidence.get("sane_outputs", False):
        reasons.append("nonsense_probe_outputs")
    if not operator_selected:
        reasons.append("operator_not_selected")

    if reasons:
        return SubmissionDecision("DO_NOT_SUBMIT", tuple(sorted(set(reasons))))
    return SubmissionDecision("SUBMIT_CANDIDATE", ("candidate_passed_emergency_gate",))


def _scorebook_status(adapter_path: str, entries: tuple[ScorebookEntry, ...]) -> str | None:
    normalized = adapter_path.replace("\\", "/").rstrip("/")
    for entry in entries:
        if entry.adapter_path.replace("\\", "/").rstrip("/") == normalized:
            return entry.status
    return None


def _set_or_check_hash(instance: object, hash_field: str) -> None:
    expected = stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})
    current = getattr(instance, hash_field)
    if not current:
        object.__setattr__(instance, hash_field, expected)
    elif current != expected:
        raise SubmissionDecisionError(f"{hash_field} does not match payload.")
