from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )


class OperatorFamily(str, Enum):
    ALGEBRAIC = "algebraic"
    NUMBER_THEORETIC = "number_theoretic"
    COMBINATORIAL = "combinatorial"
    GEOMETRIC = "geometric"
    SEARCH = "search"
    VERIFICATION = "verification"
    REPAIR = "repair"
    STRUCTURAL = "structural"


class ToolCapability(str, Enum):
    NONE = "none"
    SYMPY = "sympy"
    BRUTE_FORCE = "brute_force"
    GEOMETRY = "geometry"
    NUMBER_THEORY = "number_theory"
    SEARCH_ENUMERATOR = "search_enumerator"


class PreconditionKind(str, Enum):
    DOMAIN = "domain"
    ARCHETYPE = "archetype"
    GOAL = "goal"
    CONSTRAINT = "constraint"
    INVARIANT = "invariant"
    ENTITY = "entity"
    TOOL = "tool"
    SEARCH_SPACE = "search_space"
    STATE = "state"


class PostconditionKind(str, Enum):
    CONSTRAINT_ADDED = "constraint_added"
    GOAL_REFINED = "goal_refined"
    INVARIANT_ADDED = "invariant_added"
    SEARCH_SPACE_REDUCED = "search_space_reduced"
    CASE_SPLIT_CREATED = "case_split_created"
    CONTRADICTION_EXPOSED = "contradiction_exposed"
    REPRESENTATION_CHANGED = "representation_changed"
    VERIFICATION_SIGNAL_ADDED = "verification_signal_added"


class ApplicabilityStatus(str, Enum):
    APPLICABLE = "applicable"
    WEAK = "weak"
    INAPPLICABLE = "inapplicable"


class FailureMode(str, Enum):
    PRECONDITION_FAIL = "precondition_fail"
    DOMAIN_MISMATCH = "domain_mismatch"
    ARCHETYPE_MISMATCH = "archetype_mismatch"
    TOOL_UNAVAILABLE = "tool_unavailable"
    INVARIANT_CONFLICT = "invariant_conflict"
    SEARCH_SPACE_TOO_LARGE = "search_space_too_large"
    LOW_CONFIDENCE = "low_confidence"
    UNKNOWN = "unknown"


class MiningEvidenceKind(str, Enum):
    OPERATOR_STEP = "operator_step"
    OPERATOR_SEQUENCE = "operator_sequence"
    REPAIR_TRANSITION = "repair_transition"
    VERIFIER_GAIN = "verifier_gain"
    TOOL_CONFIRMED = "tool_confirmed"


class OperatorStateEffectKind(str, Enum):
    GOAL_PROGRESS = "goal_progress"
    SEARCH_SPACE_REDUCTION = "search_space_reduction"
    INVARIANT_STRENGTHENING = "invariant_strengthening"
    REPRESENTATION_SHIFT = "representation_shift"
    OBLIGATION_DISCHARGE = "obligation_discharge"
    VERIFICATION_GAIN = "verification_gain"
    CASE_SPLIT = "case_split"
    REPAIR_LOCALIZATION = "repair_localization"


class RetrievalCompatibilityKind(str, Enum):
    ARCHETYPE = "archetype"
    EVIDENCE_KIND = "evidence_kind"
    FAILURE_MODE = "failure_mode"
    OPERATOR_PREFIX = "operator_prefix"
    REPAIR_CONTEXT = "repair_context"
    TAG = "tag"


class OperatorCondition(StrictModel):
    condition_id: str
    kind: PreconditionKind
    field_name: str
    operator: str = "exists"
    expected_value: str | None = None
    weight: float = Field(1.0, ge=0.0)
    negate: bool = False
    description: str = ""


class OperatorPostcondition(StrictModel):
    postcondition_id: str
    kind: PostconditionKind
    description: str
    expected_effect: str | None = None
    confidence_hint: float = Field(0.5, ge=0.0, le=1.0)


class OperatorNeighbor(StrictModel):
    operator_name: str
    reason: str
    weight: float = Field(1.0, ge=0.0)


class OperatorStateEffect(StrictModel):
    effect_id: str
    kind: OperatorStateEffectKind
    target_field: str
    direction: str = "increase"
    description: str
    confidence_hint: float = Field(0.5, ge=0.0, le=1.0)


class RetrievalCompatibility(StrictModel):
    compatibility_id: str
    kind: RetrievalCompatibilityKind
    value: str
    weight: float = Field(1.0, ge=0.0)
    description: str = ""


class OperatorUsageStats(StrictModel):
    attempts: int = Field(0, ge=0)
    successes: int = Field(0, ge=0)
    failures: int = Field(0, ge=0)
    repairs_triggered: int = Field(0, ge=0)
    verifier_gain_mean: float = 0.0
    symbolic_success_rate: float = Field(0.5, ge=0.0, le=1.0)
    recent_success_rate: float = Field(0.5, ge=0.0, le=1.0)

    @property
    def success_rate(self) -> float:
        if self.attempts <= 0:
            return 0.5
        return self.successes / max(self.attempts, 1)

    def updated(
        self,
        *,
        success: bool,
        verifier_gain: float = 0.0,
        symbolic_ok: bool | None = None,
        repair_triggered: bool = False,
        ema_alpha: float = 0.10,
    ) -> "OperatorUsageStats":
        attempts = self.attempts + 1
        successes = self.successes + (1 if success else 0)
        failures = self.failures + (0 if success else 1)
        repairs_triggered = self.repairs_triggered + (1 if repair_triggered else 0)

        new_recent = (1.0 - ema_alpha) * self.recent_success_rate + ema_alpha * (1.0 if success else 0.0)

        if symbolic_ok is None:
            new_symbolic = self.symbolic_success_rate
        else:
            new_symbolic = (1.0 - ema_alpha) * self.symbolic_success_rate + ema_alpha * (1.0 if symbolic_ok else 0.0)

        new_mean_gain = ((self.verifier_gain_mean * self.attempts) + verifier_gain) / max(attempts, 1)

        return OperatorUsageStats(
            attempts=attempts,
            successes=successes,
            failures=failures,
            repairs_triggered=repairs_triggered,
            verifier_gain_mean=new_mean_gain,
            symbolic_success_rate=max(0.0, min(1.0, new_symbolic)),
            recent_success_rate=max(0.0, min(1.0, new_recent)),
        )


class OperatorDescriptor(StrictModel):
    operator_name: str
    family: OperatorFamily
    description: str
    compatible_domains: list[str] = Field(default_factory=list)
    compatible_archetypes: list[str] = Field(default_factory=list)
    tool_capabilities: list[ToolCapability] = Field(default_factory=list)
    preconditions: list[OperatorCondition] = Field(default_factory=list)
    postconditions: list[OperatorPostcondition] = Field(default_factory=list)
    failure_modes: list[FailureMode] = Field(default_factory=list)
    repair_neighbors: list[OperatorNeighbor] = Field(default_factory=list)
    state_effects: list[OperatorStateEffect] = Field(default_factory=list)
    retrieval_compatibility: list[RetrievalCompatibility] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    mined: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if not self.state_effects and self.postconditions:
            effects: list[OperatorStateEffect] = []
            for index, item in enumerate(self.postconditions):
                kind_map = {
                    PostconditionKind.CONSTRAINT_ADDED: OperatorStateEffectKind.GOAL_PROGRESS,
                    PostconditionKind.GOAL_REFINED: OperatorStateEffectKind.GOAL_PROGRESS,
                    PostconditionKind.INVARIANT_ADDED: OperatorStateEffectKind.INVARIANT_STRENGTHENING,
                    PostconditionKind.SEARCH_SPACE_REDUCED: OperatorStateEffectKind.SEARCH_SPACE_REDUCTION,
                    PostconditionKind.CASE_SPLIT_CREATED: OperatorStateEffectKind.CASE_SPLIT,
                    PostconditionKind.CONTRADICTION_EXPOSED: OperatorStateEffectKind.OBLIGATION_DISCHARGE,
                    PostconditionKind.REPRESENTATION_CHANGED: OperatorStateEffectKind.REPRESENTATION_SHIFT,
                    PostconditionKind.VERIFICATION_SIGNAL_ADDED: OperatorStateEffectKind.VERIFICATION_GAIN,
                }
                effects.append(
                    OperatorStateEffect(
                        effect_id=f"{self.operator_name}::effect::{index}",
                        kind=kind_map.get(item.kind, OperatorStateEffectKind.GOAL_PROGRESS),
                        target_field=item.kind.value,
                        direction="increase",
                        description=item.description,
                        confidence_hint=item.confidence_hint,
                    )
                )
            object.__setattr__(self, "state_effects", effects)

        if not self.retrieval_compatibility:
            compat: list[RetrievalCompatibility] = []
            for index, archetype in enumerate(self.compatible_archetypes):
                compat.append(
                    RetrievalCompatibility(
                        compatibility_id=f"{self.operator_name}::arch::{index}",
                        kind=RetrievalCompatibilityKind.ARCHETYPE,
                        value=archetype,
                        weight=1.0,
                        description=f"operator aligns with {archetype} traces",
                    )
                )
            for index, effect in enumerate(self.state_effects):
                if effect.kind in {OperatorStateEffectKind.OBLIGATION_DISCHARGE, OperatorStateEffectKind.VERIFICATION_GAIN}:
                    compat.append(
                        RetrievalCompatibility(
                            compatibility_id=f"{self.operator_name}::evidence::{index}",
                            kind=RetrievalCompatibilityKind.EVIDENCE_KIND,
                            value="symbolic_check",
                            weight=0.9,
                            description=f"{self.operator_name} benefits from symbolic-check-backed retrieval",
                        )
                    )
            for index, mode in enumerate(self.failure_modes):
                compat.append(
                    RetrievalCompatibility(
                        compatibility_id=f"{self.operator_name}::failure::{index}",
                        kind=RetrievalCompatibilityKind.FAILURE_MODE,
                        value=mode.value,
                        weight=0.8,
                        description=f"operator can address {mode.value} failures",
                    )
                )
            for index, neighbor in enumerate(self.repair_neighbors):
                compat.append(
                    RetrievalCompatibility(
                        compatibility_id=f"{self.operator_name}::repair::{index}",
                        kind=RetrievalCompatibilityKind.REPAIR_CONTEXT,
                        value=neighbor.operator_name,
                        weight=neighbor.weight,
                        description=neighbor.reason,
                    )
                )
            for index, tag in enumerate(self.tags):
                compat.append(
                    RetrievalCompatibility(
                        compatibility_id=f"{self.operator_name}::tag::{index}",
                        kind=RetrievalCompatibilityKind.TAG,
                        value=tag,
                        weight=0.6,
                        description=f"tag-aligned retrieval support: {tag}",
                    )
                )
            object.__setattr__(self, "retrieval_compatibility", compat)


class OperatorApplicability(StrictModel):
    operator_name: str
    status: ApplicabilityStatus
    score: float = Field(0.0, ge=0.0, le=1.0)
    passed_conditions: list[str] = Field(default_factory=list)
    failed_conditions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    failure_modes: list[FailureMode] = Field(default_factory=list)
    retrieval_compatibility_score: float = Field(0.0, ge=0.0, le=1.0)
    compatible_retrieval_signals: list[str] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class OperatorPolicyScore(StrictModel):
    operator_name: str
    family: OperatorFamily
    base_route_score: float = 0.0
    family_score: float = 0.0
    applicability_score: float = 0.0
    retrieval_score: float = 0.0
    success_history_score: float = 0.0
    uncertainty_smoothing: float = 0.0
    final_score: float = 0.0
    normalized_score: float = 0.0
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class OperatorMiningEvidence(StrictModel):
    kind: MiningEvidenceKind
    operator_name: str
    source_branch_id: str
    source_problem_id: str | None = None
    step_index: int | None = None
    symbolic_valid: bool | None = None
    verifier_score: float | None = None
    answer: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class OperatorMiningCandidate(StrictModel):
    operator_name: str
    family: OperatorFamily
    description: str
    compatible_domains: list[str] = Field(default_factory=list)
    compatible_archetypes: list[str] = Field(default_factory=list)
    tool_capabilities: list[ToolCapability] = Field(default_factory=list)
    preconditions: list[OperatorCondition] = Field(default_factory=list)
    postconditions: list[OperatorPostcondition] = Field(default_factory=list)
    failure_modes: list[FailureMode] = Field(default_factory=list)
    repair_neighbors: list[OperatorNeighbor] = Field(default_factory=list)
    support_count: int = Field(0, ge=0)
    mean_verifier_gain: float = 0.0
    symbolic_success_rate: float = Field(0.5, ge=0.0, le=1.0)
    operator_purity: float = Field(0.0, ge=0.0, le=1.0)
    reliability: float = Field(0.0, ge=0.0, le=1.0)
    promotion_safe: bool = False
    evidence: list[OperatorMiningEvidence] = Field(default_factory=list)
    promotion_thresholds: dict[str, Any] = Field(default_factory=dict)
    provenance_summary: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class OperatorMiningReport(StrictModel):
    total_traces_seen: int = 0
    structured_steps_seen: int = 0
    evidence_items: int = 0
    candidates_emitted: int = 0
    candidates_promoted: int = 0
    warnings: list[str] = Field(default_factory=list)


CORE_OPERATOR_ORDER: list[str] = [
    "substitution",
    "symbolic_manipulation",
    "modular_arithmetic",
    "parity_mod_reduction",
    "invariant_introduction",
    "symmetry_reduction",
    "bounding",
    "extremal_argument",
    "contradiction",
    "induction",
    "case_work",
    "double_counting",
    "bijection",
    "pigeonhole",
    "constructive_build",
    "coordinate_change",
    "inversion",
    "vieta",
    "am_gm",
    "cauchy_schwarz",
    "brute_force_small",
]


OPERATOR_FAMILY_BY_NAME: dict[str, OperatorFamily] = {
    "substitution": OperatorFamily.ALGEBRAIC,
    "symbolic_manipulation": OperatorFamily.ALGEBRAIC,
    "modular_arithmetic": OperatorFamily.NUMBER_THEORETIC,
    "parity_mod_reduction": OperatorFamily.NUMBER_THEORETIC,
    "invariant_introduction": OperatorFamily.STRUCTURAL,
    "symmetry_reduction": OperatorFamily.STRUCTURAL,
    "bounding": OperatorFamily.ALGEBRAIC,
    "extremal_argument": OperatorFamily.COMBINATORIAL,
    "contradiction": OperatorFamily.STRUCTURAL,
    "induction": OperatorFamily.STRUCTURAL,
    "case_work": OperatorFamily.SEARCH,
    "double_counting": OperatorFamily.COMBINATORIAL,
    "bijection": OperatorFamily.COMBINATORIAL,
    "pigeonhole": OperatorFamily.COMBINATORIAL,
    "constructive_build": OperatorFamily.SEARCH,
    "coordinate_change": OperatorFamily.GEOMETRIC,
    "inversion": OperatorFamily.GEOMETRIC,
    "vieta": OperatorFamily.ALGEBRAIC,
    "am_gm": OperatorFamily.ALGEBRAIC,
    "cauchy_schwarz": OperatorFamily.ALGEBRAIC,
    "brute_force_small": OperatorFamily.VERIFICATION,
}


def operator_family_for_name(operator_name: str) -> OperatorFamily:
    return OPERATOR_FAMILY_BY_NAME.get(operator_name, OperatorFamily.STRUCTURAL)


__all__ = [
    "StrictModel",
    "OperatorFamily",
    "ToolCapability",
    "PreconditionKind",
    "PostconditionKind",
    "ApplicabilityStatus",
    "FailureMode",
    "MiningEvidenceKind",
    "OperatorStateEffectKind",
    "RetrievalCompatibilityKind",
    "OperatorCondition",
    "OperatorPostcondition",
    "OperatorNeighbor",
    "OperatorStateEffect",
    "RetrievalCompatibility",
    "OperatorUsageStats",
    "OperatorDescriptor",
    "OperatorApplicability",
    "OperatorPolicyScore",
    "OperatorMiningEvidence",
    "OperatorMiningCandidate",
    "OperatorMiningReport",
    "CORE_OPERATOR_ORDER",
    "OPERATOR_FAMILY_BY_NAME",
    "operator_family_for_name",
]
