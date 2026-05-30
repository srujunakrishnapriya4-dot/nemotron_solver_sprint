from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

from src.branches.branch_state import (
    BranchPhase,
    BranchState,
    BranchStep,
    CritiqueEvidence,
    SymbolicEvidence,
    VerifierEvidence,
)
from src.common.constants import MAX_REPAIR_ATTEMPTS, VERIFIER_PASS_THRESHOLD
from src.common.schemas import FailureType


def _normalize_text(text: str | None) -> str:
    return " ".join((text or "").strip().split())


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class FailureCategory(str, Enum):
    ARITHMETIC_ERROR = "arithmetic_error"
    LOGIC_GAP = "logic_gap"
    MISSING_CASE = "missing_case"
    SYMBOLIC_MISMATCH = "symbolic_mismatch"
    FALSE_ASSUMPTION = "false_assumption"
    COVERAGE_GAP = "coverage_gap"
    VERIFIER_REJECTION = "verifier_rejection"
    TIMEOUT_OR_UNSUPPORTED = "timeout_or_unsupported"


class FailureSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RepairType(str, Enum):
    LOCAL_ARITHMETIC_FIX = "local_arithmetic_fix"
    SYMBOLIC_REWRITE = "symbolic_rewrite"
    INSERT_CASE_SPLIT = "insert_case_split"
    RESTORE_DROPPED_CASES = "restore_dropped_cases"
    ASSUMPTION_RESET = "assumption_reset"
    LOGIC_BRIDGE = "logic_bridge"
    VERIFIER_GUIDED_REVISION = "verifier_guided_revision"
    ROLLBACK_AND_RESAMPLE = "rollback_and_resample"
    UNSUPPORTED_ESCALATION = "unsupported_escalation"


class EvidenceKind(str, Enum):
    STEP = "step"
    SYMBOLIC = "symbolic"
    VERIFIER = "verifier"
    CRITIQUE = "critique"
    METADATA = "metadata"


@dataclass(frozen=True)
class FailureLocation:
    step_index: int | None
    step_id: str | None
    prefix_keep_steps: int
    evidence_kind: EvidenceKind
    evidence_id: str | None = None
    node_id: str | None = None
    obligation_id: str | None = None
    summary: str = ""


@dataclass(frozen=True)
class FailureDiagnostic:
    code: str
    summary: str
    weight: float
    evidence_kind: EvidenceKind
    evidence_id: str | None = None
    step_id: str | None = None
    step_index: int | None = None
    node_id: str | None = None
    obligation_id: str | None = None


@dataclass(frozen=True)
class BranchFailureDiagnosis:
    category: FailureCategory
    schema_failure_type: FailureType
    severity: FailureSeverity
    recoverability: float
    repair_type: RepairType
    location: FailureLocation
    summary: str
    diagnostics: tuple[FailureDiagnostic, ...] = field(default_factory=tuple)
    evidence_ids: tuple[str, ...] = field(default_factory=tuple)
    suggested_operator_bias: tuple[str, ...] = field(default_factory=tuple)
    can_repair_in_place: bool = True

    @property
    def should_retry_locally(self) -> bool:
        return self.can_repair_in_place and self.recoverability >= 0.35


@dataclass(frozen=True)
class _EvidenceRef:
    kind: EvidenceKind
    evidence_id: str | None
    score: float
    summary: str
    step_id: str | None = None
    supporting_step_ids: tuple[str, ...] = field(default_factory=tuple)


class FailureClassifier:
    """
    Typed, deterministic branch failure diagnosis.

    The classifier uses branch-state evidence, not raw string-only heuristics, to
    produce a repair-routable diagnosis with locality and recoverability.
    """

    def classify(self, branch: BranchState) -> BranchFailureDiagnosis:
        active_steps = branch.active_steps()
        symbolic = branch.active_symbolic_evidence()
        verifier = branch.active_verifier_evidence()
        critique = branch.active_critique_evidence()

        diagnostics: list[FailureDiagnostic] = []

        timeout_diag = self._timeout_or_unsupported(branch)
        if timeout_diag is not None:
            diagnostics.append(timeout_diag)

        symbolic_diags = self._symbolic_diagnostics(symbolic, active_steps)
        verifier_diags = self._verifier_diagnostics(verifier, active_steps)
        critique_diags = self._critique_diagnostics(critique, active_steps)
        diagnostics.extend(symbolic_diags)
        diagnostics.extend(verifier_diags)
        diagnostics.extend(critique_diags)

        if not diagnostics:
            diagnostics.append(
                FailureDiagnostic(
                    code="verifier_fallback_low_score",
                    summary="No explicit evidence attached; defaulting to verifier-style rejection.",
                    weight=max(0.25, 1.0 - branch.composite_score()),
                    evidence_kind=EvidenceKind.METADATA,
                )
            )

        best = self._select_best_diagnostic(diagnostics)
        location = self._localize(best, branch)
        category = self._category_from_code(best.code)
        severity = self._severity_for(category, best.weight, branch)
        recoverability = self._recoverability_for(category, best.weight, branch, verifier)
        repair_type = self._repair_type_for(category, branch)
        summary = self._build_summary(category, best, location, recoverability)
        evidence_ids = tuple(
            diag.evidence_id for diag in diagnostics if diag.evidence_id is not None
        )
        suggested_operator_bias = tuple(branch.route.repair_neighbors)

        return BranchFailureDiagnosis(
            category=category,
            schema_failure_type=self._schema_failure_type(category),
            severity=severity,
            recoverability=recoverability,
            repair_type=repair_type,
            location=location,
            summary=summary,
            diagnostics=tuple(diagnostics),
            evidence_ids=evidence_ids,
            suggested_operator_bias=suggested_operator_bias,
            can_repair_in_place=repair_type not in {
                RepairType.ROLLBACK_AND_RESAMPLE,
                RepairType.UNSUPPORTED_ESCALATION,
            },
        )

    def classify_many(self, branches: list[BranchState]) -> tuple[BranchFailureDiagnosis, ...]:
        return tuple(self.classify(branch) for branch in branches)

    def _timeout_or_unsupported(self, branch: BranchState) -> FailureDiagnostic | None:
        reason = _normalize_text(str(branch.metadata.get("failure_reason", "")))
        unsupported = bool(branch.metadata.get("unsupported", False))
        timed_out = branch.phase == BranchPhase.FAILED and (
            "timeout" in reason or bool(branch.metadata.get("timed_out", False))
        )
        if timed_out or unsupported:
            summary = reason or ("Unsupported branch state" if unsupported else "Timeout reached")
            return FailureDiagnostic(
                code="timeout_or_unsupported",
                summary=summary,
                weight=0.98 if unsupported else 0.90,
                evidence_kind=EvidenceKind.METADATA,
            )
        return None

    def _symbolic_diagnostics(
        self,
        evidences: tuple[SymbolicEvidence, ...],
        steps: tuple[BranchStep, ...],
    ) -> list[FailureDiagnostic]:
        out: list[FailureDiagnostic] = []
        step_index_by_id = {step.step_id: step.index for step in steps}
        for evidence in evidences:
            if evidence.passed:
                continue
            check_name = evidence.check_name.lower()
            summary = evidence.summary.lower()
            category_code = "symbolic_mismatch"
            if any(token in check_name or token in summary for token in ("arithmetic", "numeric", "integer", "bounds")):
                category_code = "arithmetic_error"
            elif any(token in summary for token in ("contradiction", "inconsistent", "violates", "mismatch")):
                category_code = "symbolic_mismatch"
            step_id = evidence.supporting_step_ids[-1] if evidence.supporting_step_ids else None
            out.append(
                FailureDiagnostic(
                    code=category_code,
                    summary=evidence.summary,
                    weight=max(0.55, evidence.score),
                    evidence_kind=EvidenceKind.SYMBOLIC,
                    evidence_id=evidence.evidence_id,
                    step_id=step_id,
                    step_index=step_index_by_id.get(step_id) if step_id else None,
                )
            )
        return out

    def _verifier_diagnostics(
        self,
        evidences: tuple[VerifierEvidence, ...],
        steps: tuple[BranchStep, ...],
    ) -> list[FailureDiagnostic]:
        out: list[FailureDiagnostic] = []
        latest_step = steps[-1] if steps else None
        for evidence in evidences:
            summary = evidence.summary.lower()
            if any(token in summary for token in ("missing case", "not exhaustive", "unhandled case")):
                code = "missing_case"
                weight = max(0.65, 1.0 - evidence.completeness)
            elif any(token in summary for token in ("coverage", "partial", "incomplete")):
                code = "coverage_gap"
                weight = max(0.60, 1.0 - evidence.completeness)
            elif any(token in summary for token in ("assumption", "without proof", "unjustified")):
                code = "false_assumption"
                weight = max(0.60, 1.0 - evidence.logical_consistency)
            elif evidence.open_obligation_burden >= 0.35:
                code = "coverage_gap"
                weight = max(0.62, evidence.open_obligation_burden)
            elif evidence.prefix_quality < 0.45 or evidence.step_quality < 0.45:
                code = "logic_gap"
                weight = max(0.52, 1.0 - max(evidence.prefix_quality, evidence.step_quality))
            elif evidence.probability < VERIFIER_PASS_THRESHOLD:
                code = "verifier_rejection"
                weight = max(0.50, 1.0 - evidence.probability)
            else:
                code = "logic_gap"
                weight = max(0.40, 1.0 - evidence.logical_consistency)
            out.append(
                FailureDiagnostic(
                    code=code,
                    summary=evidence.summary,
                    weight=weight,
                    evidence_kind=EvidenceKind.VERIFIER,
                    evidence_id=evidence.evidence_id,
                    step_id=evidence.failure_step_id or (latest_step.step_id if latest_step else None),
                    step_index=evidence.failure_step_index if evidence.failure_step_index is not None else (latest_step.index if latest_step else None),
                    node_id=evidence.failure_node_id,
                    obligation_id=evidence.failure_obligation_id,
                )
            )
        return out

    def _critique_diagnostics(
        self,
        evidences: tuple[CritiqueEvidence, ...],
        steps: tuple[BranchStep, ...],
    ) -> list[FailureDiagnostic]:
        out: list[FailureDiagnostic] = []
        latest_step = steps[-1] if steps else None
        for evidence in evidences:
            summary = evidence.summary.lower()
            if any(token in summary for token in ("missing case", "forgot a case", "case split")):
                code = "missing_case"
            elif any(token in summary for token in ("coverage", "not complete", "incomplete")):
                code = "coverage_gap"
            elif any(token in summary for token in ("assumption", "assumed", "wlog")):
                code = "false_assumption"
            elif any(token in summary for token in ("gap", "does not follow", "unsupported leap")):
                code = "logic_gap"
            elif evidence.changed_answer:
                code = "verifier_rejection"
            else:
                continue
            out.append(
                FailureDiagnostic(
                    code=code,
                    summary=evidence.summary,
                    weight=max(0.45, evidence.critique_score),
                    evidence_kind=EvidenceKind.CRITIQUE,
                    evidence_id=evidence.evidence_id,
                    step_id=latest_step.step_id if latest_step else None,
                    step_index=latest_step.index if latest_step else None,
                )
            )
        return out

    def _select_best_diagnostic(self, diagnostics: list[FailureDiagnostic]) -> FailureDiagnostic:
        priority = {
            "timeout_or_unsupported": 0,
            "symbolic_mismatch": 1,
            "arithmetic_error": 2,
            "false_assumption": 3,
            "missing_case": 4,
            "coverage_gap": 5,
            "logic_gap": 6,
            "verifier_rejection": 7,
        }
        return sorted(
            diagnostics,
            key=lambda diag: (
                -diag.weight,
                priority.get(diag.code, 99),
                diag.step_index if diag.step_index is not None else 10**6,
                diag.code,
                diag.evidence_id or "",
            ),
        )[0]

    def _localize(self, diagnostic: FailureDiagnostic, branch: BranchState) -> FailureLocation:
        steps = branch.active_steps()
        chosen_step = None
        if diagnostic.step_id is not None:
            for step in steps:
                if step.step_id == diagnostic.step_id:
                    chosen_step = step
                    break
        if chosen_step is None and diagnostic.step_index is not None:
            for step in steps:
                if step.index == diagnostic.step_index:
                    chosen_step = step
                    break
        if chosen_step is None and steps:
            chosen_step = steps[-1]

        prefix_keep = 0
        if chosen_step is not None:
            prefix_keep = max(0, chosen_step.index)
        elif steps:
            prefix_keep = max(0, len(steps) - 1)

        return FailureLocation(
            step_index=chosen_step.index if chosen_step else None,
            step_id=chosen_step.step_id if chosen_step else None,
            prefix_keep_steps=prefix_keep,
            evidence_kind=diagnostic.evidence_kind,
            evidence_id=diagnostic.evidence_id,
            node_id=chosen_step.node_id if chosen_step else diagnostic.node_id or branch.latest_node_id,
            obligation_id=diagnostic.obligation_id,
            summary=_normalize_text(diagnostic.summary),
        )

    def _category_from_code(self, code: str) -> FailureCategory:
        mapping = {
            "arithmetic_error": FailureCategory.ARITHMETIC_ERROR,
            "logic_gap": FailureCategory.LOGIC_GAP,
            "missing_case": FailureCategory.MISSING_CASE,
            "symbolic_mismatch": FailureCategory.SYMBOLIC_MISMATCH,
            "false_assumption": FailureCategory.FALSE_ASSUMPTION,
            "coverage_gap": FailureCategory.COVERAGE_GAP,
            "verifier_rejection": FailureCategory.VERIFIER_REJECTION,
            "timeout_or_unsupported": FailureCategory.TIMEOUT_OR_UNSUPPORTED,
        }
        return mapping[code]

    def _severity_for(
        self,
        category: FailureCategory,
        weight: float,
        branch: BranchState,
    ) -> FailureSeverity:
        if category == FailureCategory.TIMEOUT_OR_UNSUPPORTED:
            return FailureSeverity.CRITICAL
        if weight >= 0.85 or branch.phase == BranchPhase.FAILED:
            return FailureSeverity.HIGH
        if weight >= 0.55:
            return FailureSeverity.MEDIUM
        return FailureSeverity.LOW

    def _recoverability_for(
        self,
        category: FailureCategory,
        weight: float,
        branch: BranchState,
        verifier: tuple[VerifierEvidence, ...],
    ) -> float:
        repair_pressure = min(branch.active_cursor.repair_count / max(MAX_REPAIR_ATTEMPTS, 1), 1.0)
        verifier_repairability = verifier[-1].repairability if verifier else 0.5

        base = {
            FailureCategory.ARITHMETIC_ERROR: 0.88,
            FailureCategory.SYMBOLIC_MISMATCH: 0.72,
            FailureCategory.MISSING_CASE: 0.70,
            FailureCategory.COVERAGE_GAP: 0.64,
            FailureCategory.LOGIC_GAP: 0.54,
            FailureCategory.FALSE_ASSUMPTION: 0.42,
            FailureCategory.VERIFIER_REJECTION: 0.48,
            FailureCategory.TIMEOUT_OR_UNSUPPORTED: 0.12,
        }[category]
        adjusted = base * (1.0 - 0.35 * repair_pressure)
        adjusted *= 0.75 + 0.25 * verifier_repairability
        adjusted *= 1.0 - 0.15 * max(0.0, weight - 0.5)
        return round(_clamp01(adjusted), 4)

    def _repair_type_for(self, category: FailureCategory, branch: BranchState) -> RepairType:
        if category == FailureCategory.ARITHMETIC_ERROR:
            return RepairType.LOCAL_ARITHMETIC_FIX
        if category == FailureCategory.SYMBOLIC_MISMATCH:
            return RepairType.SYMBOLIC_REWRITE
        if category == FailureCategory.MISSING_CASE:
            return RepairType.INSERT_CASE_SPLIT
        if category == FailureCategory.COVERAGE_GAP:
            return RepairType.RESTORE_DROPPED_CASES
        if category == FailureCategory.FALSE_ASSUMPTION:
            return RepairType.ASSUMPTION_RESET
        if category == FailureCategory.LOGIC_GAP:
            return RepairType.LOGIC_BRIDGE
        if category == FailureCategory.VERIFIER_REJECTION:
            if branch.route.repair_threshold >= 0.65:
                return RepairType.ROLLBACK_AND_RESAMPLE
            return RepairType.VERIFIER_GUIDED_REVISION
        return RepairType.UNSUPPORTED_ESCALATION

    def _schema_failure_type(self, category: FailureCategory) -> FailureType:
        mapping = {
            FailureCategory.ARITHMETIC_ERROR: FailureType.ARITHMETIC_ERROR,
            FailureCategory.LOGIC_GAP: FailureType.LOGIC_ERROR,
            FailureCategory.MISSING_CASE: FailureType.MISSING_CASE,
            FailureCategory.SYMBOLIC_MISMATCH: FailureType.SYMBOLIC_MISMATCH,
            FailureCategory.FALSE_ASSUMPTION: FailureType.FALSE_ASSUMPTION,
            FailureCategory.COVERAGE_GAP: FailureType.COVERAGE_GAP,
            FailureCategory.VERIFIER_REJECTION: FailureType.LOGIC_ERROR,
            FailureCategory.TIMEOUT_OR_UNSUPPORTED: FailureType.TIMEOUT,
        }
        return mapping[category]

    def _build_summary(
        self,
        category: FailureCategory,
        diagnostic: FailureDiagnostic,
        location: FailureLocation,
        recoverability: float,
    ) -> str:
        locality = "global"
        if location.step_index is not None:
            locality = f"step={location.step_index}"
        return (
            f"{category.value} detected at {locality}; "
            f"{_normalize_text(diagnostic.summary)}; "
            f"recoverability={recoverability:.2f}"
        )


def classify_branch_failure(branch: BranchState) -> BranchFailureDiagnosis:
    return FailureClassifier().classify(branch)


__all__ = [
    "FailureCategory",
    "FailureSeverity",
    "RepairType",
    "EvidenceKind",
    "FailureLocation",
    "FailureDiagnostic",
    "BranchFailureDiagnosis",
    "FailureClassifier",
    "classify_branch_failure",
]
