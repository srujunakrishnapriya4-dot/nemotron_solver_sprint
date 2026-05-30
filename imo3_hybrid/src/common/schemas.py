"""
Core Pydantic schemas shared across the entire pipeline.
These are immutable contracts between modules.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional, Any
from pydantic import BaseModel, Field


class ProblemDomain(str, Enum):
    ALGEBRA = "algebra"
    NUMBER_THEORY = "number_theory"
    COMBINATORICS = "combinatorics"
    GEOMETRY = "geometry"
    FUNCTIONAL_EQUATION = "functional_equation"
    MIXED = "mixed"
    UNKNOWN = "unknown"

    def __str__(self) -> str:
        return self.value


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    VERY_HARD = "very_hard"

    def __str__(self) -> str:
        return self.value


class OperatorType(str, Enum):
    SUBSTITUTION = "substitution"
    SYMMETRY_REDUCTION = "symmetry_reduction"
    PARITY_MOD_REDUCTION = "parity_mod_reduction"
    INVARIANT_INTRODUCTION = "invariant_introduction"
    EXTREMAL_ARGUMENT = "extremal_argument"
    CONTRADICTION = "contradiction"
    BIJECTION = "bijection"
    GENERATING_FUNCTION = "generating_function"
    COORDINATE_CHANGE = "coordinate_change"
    INVERSION = "inversion"
    BOUNDING = "bounding"
    BRUTE_FORCE_SMALL = "brute_force_small"
    MODULAR_ARITHMETIC = "modular_arithmetic"
    INDUCTION = "induction"
    DOUBLE_COUNTING = "double_counting"
    CAUCHY_SCHWARZ = "cauchy_schwarz"
    AM_GM = "am_gm"
    VIETA = "vieta"
    PIGEONHOLE = "pigeonhole"


class FailureType(str, Enum):
    ARITHMETIC_ERROR = "arithmetic"
    LOGIC_ERROR = "logic"
    MISSING_CASE = "missing_case"
    SYMBOLIC_MISMATCH = "symbolic_mismatch"
    FALSE_ASSUMPTION = "false_assumption"
    COVERAGE_GAP = "coverage_gap"
    TIMEOUT = "timeout"

    def __str__(self) -> str:
        return self.value


class ToolType(str, Enum):
    NONE = "none"
    SYMBOLIC = "symbolic"
    VERIFIER = "verifier"
    RETRIEVAL = "retrieval"
    PYTHON = "python"
    SEARCH = "search"


class EdgeType(str, Enum):
    DERIVE = "derive"
    EXPAND = "expand"
    SUBGOAL_EXPAND = "subgoal_expand"
    MERGE = "merge"
    REPAIR = "repair"
    ROLLBACK = "rollback"
    REFUTE = "refute"


class Archetype(str, Enum):
    PARITY = "parity"
    INVARIANT = "invariant"
    EXTREMAL = "extremal"
    PIGEONHOLE = "pigeonhole"
    MODULAR = "modular"
    SYMMETRY = "symmetry"
    BOUNDING = "bounding"
    CONTRADICTION = "contradiction"
    CONSTRUCTION = "construction"
    INDUCTION = "induction"
    BIJECTION = "bijection"
    CASE_WORK = "case_work"
    SYMBOLIC_MANIPULATION = "symbolic_manipulation"
    BRUTE_FORCE_SMALLSPACE = "brute_force_smallspace"
    GENERATING_FUNCTION = "generating_function"

    def __str__(self) -> str:
        return self.value


class AnswerType(str, Enum):
    UNKNOWN = "unknown"
    INTEGER = "integer"
    NONNEGATIVE_INTEGER = "non_negative_integer"
    SET = "set"
    CONSTRUCTION = "construction"
    REAL = "real"
    RATIONAL = "rational"

    def __str__(self) -> str:
        return self.value


class QuantifierType(str, Enum):
    FORALL = "forall"
    EXISTS = "exists"
    FIND_ALL = "find_all"
    COUNT = "count"
    CONSTRUCT = "construct"
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"

    def __str__(self) -> str:
        return self.value


class RelationType(str, Enum):
    DOMAIN_MEMBERSHIP = "domain_membership"
    INTEGRALITY = "integrality"
    POSITIVITY = "positivity"
    BOUNDEDNESS = "boundedness"
    CONGRUENCE = "congruence"
    DIVISIBILITY = "divisibility"
    EQUALITY = "equality"
    INEQUALITY = "inequality"
    ORDERING = "ordering"
    OBJECTIVE = "objective"

    def __str__(self) -> str:
        return self.value


class ConstraintOrigin(str, Enum):
    EXPLICIT_DOMAIN = "explicit_domain"
    EXPLICIT_NORMALIZED = "explicit_normalized"
    EXPLICIT_OBJECTIVE = "explicit_objective"
    INFERRED_READY = "inferred_ready"


class ConstraintStrength(str, Enum):
    HARD = "hard"


class SymbolRole(str, Enum):
    VARIABLE = "variable"


class DomainSpec(BaseModel):
    domain: str | None = None
    qualifiers: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class IndexedSymbolSpec(BaseModel):
    base_name: str
    index_text: str
    index_tokens: list[str] = Field(default_factory=list)


class SymbolAlias(BaseModel):
    alias_text: str
    normalized_alias: str
    scope_id: str
    source: str | None = None
    is_branch_local: bool = False


class SymbolRef(BaseModel):
    symbol_id: str
    canonical_name: str
    display_name: str
    base_name: str
    role: SymbolRole
    declaration_scope_id: str
    family_symbol_id: str | None = None
    indexed: IndexedSymbolSpec | None = None
    domain: DomainSpec = Field(default_factory=DomainSpec)
    annotations: dict[str, str] = Field(default_factory=dict)
    symmetry_group_ids: list[str] = Field(default_factory=list)
    aliases: list[SymbolAlias] = Field(default_factory=list)
    mention_count: int = 0


class LatexSpan(BaseModel):
    raw_text: str
    normalized_text: str
    start: int
    end: int


class NormalizationChange(BaseModel):
    stage: str
    before: str
    after: str


class NormalizationReport(BaseModel):
    original_text: str
    normalized_text: str
    changes: list[NormalizationChange] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class QuantifiedVariable(BaseModel):
    quantifier_type: QuantifierType
    variables: list[str] = Field(default_factory=list)
    raw_text: str
    span_start: int | None = None
    span_end: int | None = None


class ConstraintProvenance(BaseModel):
    source_text: str
    normalized_source_text: str
    span_start: int
    span_end: int
    stage: str
    source_channel: str


class MathOperand(BaseModel):
    raw_text: str
    normalized_text: str
    scope_id: str
    symbol_ids: list[str] = Field(default_factory=list)
    canonical_mentions: list[str] = Field(default_factory=list)
    operand_kind: str


class ObjectiveSpec(BaseModel):
    objective_mode: QuantifierType
    target_text: str
    normalized_target_text: str
    subject_variables: list[str] = Field(default_factory=list)


class ConstraintNode(BaseModel):
    constraint_id: str
    relation_type: RelationType
    origin: ConstraintOrigin
    strength: ConstraintStrength
    lhs: MathOperand | None = None
    rhs: MathOperand | None = None
    operator: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    domain_assumptions: list[str] = Field(default_factory=list)
    quantified_variables: list[QuantifiedVariable] = Field(default_factory=list)
    explicit: bool = True
    inferred_ready: bool = False
    provenance: ConstraintProvenance

    def summary_text(self) -> str:
        provenance_text = getattr(self.provenance, "normalized_source_text", "") or getattr(self.provenance, "source_text", "")
        if provenance_text:
            return provenance_text
        lhs = getattr(self.lhs, "normalized_text", "") if self.lhs else ""
        rhs = getattr(self.rhs, "normalized_text", "") if self.rhs else ""
        if lhs and self.operator and rhs:
            return f"{lhs} {self.operator} {rhs}"
        if lhs and self.operator:
            return f"{lhs} {self.operator}".strip()
        return lhs or rhs or self.relation_type.value


class ConstraintExtractionReport(BaseModel):
    normalized_text: str
    stage_counts: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    unresolved_operands: list[str] = Field(default_factory=list)
    quantifiers: list[QuantifiedVariable] = Field(default_factory=list)


class ConstraintExtractionResult(BaseModel):
    constraints: list[ConstraintNode] = Field(default_factory=list)
    objective_constraints: list[ConstraintNode] = Field(default_factory=list)
    report: ConstraintExtractionReport


class ParseQualityReport(BaseModel):
    missing_fields: list[str] = Field(default_factory=list)
    confidence_by_field: dict[str, float] = Field(default_factory=dict)
    fallback_used: bool = False
    semantic_gaps: list[str] = Field(default_factory=list)
    suggested_downstream_penalties: dict[str, float] = Field(default_factory=dict)
    overall_confidence: float = Field(0.0, ge=0.0, le=1.0)

    def model_post_init(self, __context: Any) -> None:
        if self.overall_confidence <= 0.0 and self.confidence_by_field:
            avg = sum(float(v) for v in self.confidence_by_field.values()) / max(1, len(self.confidence_by_field))
            object.__setattr__(self, "overall_confidence", max(0.0, min(1.0, avg)))

    def get(self, key: str, default: Any = None) -> Any:
        if key == "overall_confidence":
            return self.overall_confidence
        if hasattr(self, key):
            return getattr(self, key)
        return self.confidence_by_field.get(key, default)


class VariableDecl(BaseModel):
    symbol: SymbolRef
    declared_domain: DomainSpec | None = None
    source_text: str | None = None


class ConstraintGraphEdge(BaseModel):
    source_node_id: str
    target_node_id: str
    relation_label: str


class CanonicalProblemGraph(BaseModel):
    node_ids: list[str] = Field(default_factory=list)
    edges: list[ConstraintGraphEdge] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def render_constraint_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    summary = getattr(value, "summary_text", None)
    if callable(summary):
        try:
            rendered = summary()
        except Exception:
            rendered = None
        if isinstance(rendered, str) and rendered.strip():
            return rendered.strip()
    provenance = getattr(value, "provenance", None)
    for attr in ("normalized_source_text", "source_text"):
        rendered = getattr(provenance, attr, None)
        if isinstance(rendered, str) and rendered.strip():
            return rendered.strip()
    rendered = getattr(value, "normalized_text", None)
    if isinstance(rendered, str) and rendered.strip():
        return rendered.strip()
    lhs = getattr(getattr(value, "lhs", None), "normalized_text", None)
    rhs = getattr(getattr(value, "rhs", None), "normalized_text", None)
    operator = getattr(value, "operator", None)
    if isinstance(lhs, str) and lhs.strip():
        pieces = [lhs.strip()]
        if isinstance(operator, str) and operator.strip():
            pieces.append(operator.strip())
        if isinstance(rhs, str) and rhs.strip():
            pieces.append(rhs.strip())
        return " ".join(pieces).strip()
    return str(value).strip()


class ParsedProblem(BaseModel):
    problem_id: str
    raw_text: str
    normalized_text: str = ""
    latex_spans: list[Any] = Field(default_factory=list)
    variables: list[VariableDecl] = Field(default_factory=list)
    knowns: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    constraints: list[Any] = Field(default_factory=list)
    domain: ProblemDomain = ProblemDomain.MIXED
    target: str = ""
    answer_type: AnswerType | str = AnswerType.NONNEGATIVE_INTEGER
    objective: ObjectiveSpec | None = None
    problem_domain_seed: ProblemDomain = ProblemDomain.UNKNOWN
    difficulty_seed: float = Field(0.5, ge=0.0, le=1.0)
    likely_archetypes: list[str] = Field(default_factory=list)
    archetype_cues: list[Any] = Field(default_factory=list)
    symmetries: list[str] = Field(default_factory=list)
    parity_cues: list[str] = Field(default_factory=list)
    integrality_constraints: list[str] = Field(default_factory=list)
    symbol_table: dict[str, str] = Field(default_factory=dict)
    parse_quality: ParseQualityReport | dict[str, Any] = Field(default_factory=dict)
    canonical_problem_graph: CanonicalProblemGraph | None = None
    proof_targets: list[str] = Field(default_factory=list)
    proof_obligation_hints: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if not self.normalized_text:
            object.__setattr__(self, "normalized_text", self.raw_text)
        if self.problem_domain_seed == ProblemDomain.UNKNOWN and self.domain is not None:
            object.__setattr__(self, "problem_domain_seed", self.domain)
        if self.domain == ProblemDomain.MIXED and self.problem_domain_seed not in {ProblemDomain.UNKNOWN, ProblemDomain.MIXED}:
            object.__setattr__(self, "domain", self.problem_domain_seed)
        if not self.target and self.objective is not None:
            object.__setattr__(
                self,
                "target",
                self.objective.normalized_target_text or self.objective.target_text,
            )
        if not self.likely_archetypes and self.archetype_cues:
            archetypes = [getattr(item, "value", str(item)) for item in self.archetype_cues if str(getattr(item, "value", item)).strip()]
            object.__setattr__(self, "likely_archetypes", list(dict.fromkeys(archetypes)))
        if not self.unknowns and self.variables:
            names = [
                item.symbol.canonical_name
                for item in self.variables
                if getattr(getattr(item, "symbol", None), "canonical_name", "")
            ]
            object.__setattr__(self, "unknowns", list(dict.fromkeys(names)))
        if not self.symbol_table and self.variables:
            symbol_map = {
                item.symbol.canonical_name: item.symbol.symbol_id
                for item in self.variables
                if getattr(getattr(item, "symbol", None), "canonical_name", "")
            }
            object.__setattr__(self, "symbol_table", symbol_map)
        if not self.proof_targets and self.target:
            object.__setattr__(self, "proof_targets", [self.target])

    def constraint_texts(self) -> list[str]:
        return [text for text in (render_constraint_text(item) for item in self.constraints) if text]


class BudgetPlan(BaseModel):
    branch_budget: int = 32
    retrieval_depth: int = 3
    max_search_depth: int = 6
    max_search_nodes: int = 64
    repair_budget: int = 1
    self_consistency_samples: int = 32
    critique_top_k: int = 5
    widen_on_uncertainty: bool = False
    use_retrieval: bool = True
    use_symbolic: bool = True
    use_brute_force: bool = False
    repair_aggressiveness: float = Field(0.5, ge=0.0, le=1.0)
    resample_aggressiveness: float = Field(0.5, ge=0.0, le=1.0)
    critique_aggressiveness: float = Field(0.5, ge=0.0, le=1.0)
    verifier_disagreement_widening: bool = True
    entropy_widening: bool = True
    difficulty_conditioned: bool = True


class ProblemTypePrediction(BaseModel):
    problem_id: str
    problem_type_probs: dict[str, float]
    top_domain: ProblemDomain
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    entropy: float = Field(0.0, ge=0.0)
    is_mixed: bool = False
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class ArchetypePrediction(BaseModel):
    problem_id: str
    archetype_probs: dict[str, float]
    top_archetypes: list[str] = Field(default_factory=list)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    entropy: float = Field(0.0, ge=0.0)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class DifficultyEstimate(BaseModel):
    problem_id: str
    difficulty: Difficulty
    difficulty_score: float = Field(0.5, ge=0.0, le=1.0)
    uncertainty: float = Field(0.0, ge=0.0, le=1.0)
    budget_plan: BudgetPlan
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class RouteDecision(BaseModel):
    problem_id: str

    problem_type_probs: dict[str, float] = Field(default_factory=dict)
    archetype_probs: dict[str, float] = Field(default_factory=dict)
    difficulty: Difficulty = Difficulty.MEDIUM
    difficulty_score: float = Field(0.5, ge=0.0, le=1.0)
    operator_prior: dict[str, float] = Field(default_factory=dict)
    budget_plan: BudgetPlan = Field(default_factory=BudgetPlan)

    retrieval_depth: int = 3
    branch_budget: int = 32
    repair_threshold: float = Field(0.5, ge=0.0, le=1.0)
    route_uncertainty: float = Field(0.0, ge=0.0, le=1.0)
    verifier_mode: str = "default"

    problem_type: dict[str, float] = Field(default_factory=dict)
    archetypes: dict[str, float] = Field(default_factory=dict)
    use_retrieval: bool = True
    use_symbolic: bool = True
    use_brute_force: bool = False

    retrieval_tags: list[str] = Field(default_factory=list)
    repair_neighbors: list[str] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    route_rationale: list[str] = Field(default_factory=list)
    compute_signals: dict[str, float] = Field(default_factory=dict)
    calibration_safe_split: str | None = None
    calibration_metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if not self.problem_type and self.problem_type_probs:
            object.__setattr__(self, "problem_type", dict(self.problem_type_probs))
        if not self.archetypes and self.archetype_probs:
            object.__setattr__(self, "archetypes", dict(self.archetype_probs))
        if self.branch_budget <= 0 and self.budget_plan.branch_budget > 0:
            object.__setattr__(self, "branch_budget", self.budget_plan.branch_budget)
        if self.retrieval_depth <= 0 and self.budget_plan.retrieval_depth > 0:
            object.__setattr__(self, "retrieval_depth", self.budget_plan.retrieval_depth)
        object.__setattr__(self, "use_retrieval", self.budget_plan.use_retrieval)
        object.__setattr__(self, "use_symbolic", self.budget_plan.use_symbolic)
        object.__setattr__(self, "use_brute_force", self.budget_plan.use_brute_force)


class GraphNode(BaseModel):
    node_id: str
    parent_id: Optional[str] = None
    branch_id: str = ""
    state_summary: str = ""
    active_invariants: list[str] = Field(default_factory=list)
    remaining_constraints: list[str] = Field(default_factory=list)
    operator_history: list[str] = Field(default_factory=list)
    partial_solution: str = ""
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    verifier_score: float = Field(0.0, ge=0.0, le=1.0)
    symbolic_valid: bool = False
    depth: int = 0
    is_terminal: bool = False


class GraphEdge(BaseModel):
    edge_id: str
    source_id: str
    target_id: str
    operator_applied: str
    edge_type: str = "expand"


class ReasoningStep(BaseModel):
    step_num: int
    description: str
    operator_used: Optional[str] = None
    symbolic_expression: Optional[str] = None
    symbolic_valid: bool = True
    python_code: Optional[str] = None
    python_result: Optional[str] = None


class BranchTrace(BaseModel):
    branch_id: str
    problem_id: str
    steps: list[ReasoningStep] = Field(default_factory=list)
    full_reasoning: str = ""
    answer: Optional[str] = None
    answer_canonical: Optional[str] = None
    failure_type: Optional[FailureType] = None
    failure_location: Optional[str] = None
    symbolic_valid: bool = False
    branch_score: float = 0.0
    verifier_score: float = 0.0
    tool_consistency: float = 0.0
    logical_consistency: float = 0.0
    completeness: float = 0.0
    repairability: float = 0.0
    answer_correctness_likelihood: float = 0.0
    symbolic_agreement: float = 0.0
    step_quality: float = 0.0
    prefix_quality: float = 0.0
    prm_step_quality: float = 0.0
    prm_prefix_quality: float = 0.0
    retrieval_support: float = 0.0
    retrieval_compatibility: float = 0.0
    operator_reliability: float = 0.0
    repair_locality: float = 0.0
    discharge_fraction: float = 0.0
    open_obligation_burden: float = 0.0
    verifier_decomposition: dict[str, float] = Field(default_factory=dict)
    self_critiqued: bool = False
    repaired: bool = False
    repair_count: int = 0
    operator_sequence: list[str] = Field(default_factory=list)
    archetype_used: Optional[str] = None
    retrieval_used: bool = False
    generation_time_sec: float = 0.0
    failure_step_index: int | None = None
    failure_step_id: str | None = None
    failure_node_id: str | None = None
    failure_obligation_id: str | None = None
    proof_state_fingerprint: str | None = None
    proof_obligations: list[dict[str, Any]] = Field(default_factory=list)
    source_split: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerifierLabel(BaseModel):
    branch_id: str
    step_correctness: list[int] = Field(default_factory=list)
    symbolic_agreement: float = Field(0.0, ge=0.0, le=1.0)
    logical_consistency: float = Field(0.0, ge=0.0, le=1.0)
    completeness: float = Field(0.0, ge=0.0, le=1.0)
    final_answer_correct: int = 0
    final_answer_correct_probability: float = Field(0.0, ge=0.0, le=1.0)
    repairability: float = Field(0.0, ge=0.0, le=1.0)
    repair_type: Optional[str] = None
    failure_type: Optional[str] = None
    failure_location: Optional[str] = None
    composite_score: float = Field(0.0, ge=0.0, le=1.0)


class OperatorLibraryEntry(BaseModel):
    operator: str
    description: str = ""
    preconditions: list[str] = Field(default_factory=list)
    postconditions: list[str] = Field(default_factory=list)
    compatible_archetypes: list[str] = Field(default_factory=list)
    compatible_domains: list[str] = Field(default_factory=list)
    tool_compatibility: list[str] = Field(default_factory=list)
    fallback_operator: Optional[str] = None
    success_rate: float = Field(0.5, ge=0.0, le=1.0)
    failure_modes: list[str] = Field(default_factory=list)
    example_applications: list[str] = Field(default_factory=list)
    mined_from_cluster_id: Optional[str] = None


class RetrievedTrace(BaseModel):
    trace_id: str
    problem: str
    solution: str
    answer: str
    domain: str
    archetypes: list[str]
    operators_used: list[str]
    similarity_score: float
    source: str = "mined"
    operator_support: dict[str, float] = Field(default_factory=dict)
    repair_operator_hints: list[str] = Field(default_factory=list)
    branch_continuation_bias: dict[str, float] = Field(default_factory=dict)
    compatibility_score: float = Field(0.0, ge=0.0, le=1.0)
    reliability_score: float = Field(0.0, ge=0.0, le=1.0)
    operator_reliability: dict[str, float] = Field(default_factory=dict)
    support_coverage: float = Field(0.0, ge=0.0, le=1.0)
    compatibility_reasons: list[str] = Field(default_factory=list)
    relevant_obligation_claims: list[str] = Field(default_factory=list)
    relevant_evidence_kinds: list[str] = Field(default_factory=list)
    failure_mode_support: list[str] = Field(default_factory=list)
    strategy_summary: str = ""
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class CandidateAnswer(BaseModel):
    answer: str
    answer_canonical: str
    branch_ids: list[str] = Field(default_factory=list)
    verifier_score: float = 0.0
    tool_consistency: float = 0.0
    answer_agreement: float = 0.0
    branch_novelty: float = 0.0
    symbolic_check: float = 0.0
    composite_score: float = 0.0
    cluster_size: int = 1
    entropy_penalty: float = 0.0


class FinalPrediction(BaseModel):
    problem_id: str
    final_answer: int
    confidence: float
    winning_cluster: CandidateAnswer
    num_branches_generated: int
    num_branches_survived: int
    solve_time_sec: float
    method_used: str = "hybrid"
    signal_decomposition: dict[str, float] = Field(default_factory=dict)
    calibration_summary: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)


class AttemptEntropySource(str, Enum):
    TRUE_LOGPROBS = "true_logprobs"
    PROXY_CONFIDENCE = "proxy_confidence"
    UNAVAILABLE = "unavailable"


class AttemptRecord(BaseModel):
    attempt_id: str
    seed: int | None = None
    raw_text: str = ""
    extracted_answer: str | None = None
    valid_answer: bool = False
    mean_token_entropy: float | None = None
    entropy_source: AttemptEntropySource = AttemptEntropySource.UNAVAILABLE
    runtime_sec: float = Field(0.0, ge=0.0)
    stopped_reason: str = "completed"
    tool_call_count: int = Field(0, ge=0)
    tool_error_count: int = Field(0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AttemptBatchResult(BaseModel):
    attempts: list[AttemptRecord] = Field(default_factory=list)
    stopped_reason: str = "completed"
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrainingRow(BaseModel):
    problem_id: str
    prompt: str
    target: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class FailureBucket(str, Enum):
    SEARCH_FAILURE = "search_failure"
    VERIFIER_FAILURE = "verifier_failure"
    SYMBOLIC_FAILURE = "symbolic_failure"
    RETRIEVAL_FAILURE = "retrieval_failure"
    OPERATOR_FAILURE = "operator_failure"
    AGGREGATION_FAILURE = "aggregation_failure"
    ROUTING_BUDGET_FAILURE = "routing_budget_failure"
    UNKNOWN_FAILURE = "unknown_failure"


class FailureCause(BaseModel):
    failure_bucket: FailureBucket
    subtype: str
    weight: float = Field(0.0, ge=0.0, le=1.0)
    rationale: str = ""
    branch_id: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class HistoricalSolveRecord(BaseModel):
    problem_id: str
    raw_problem_text: str = ""
    normalized_problem_text: str = ""
    domain: str = "unknown"
    difficulty: str = "unknown"
    gold_answer: str | None = None
    predicted_answer: str | None = None
    branch_traces: list[BranchTrace] = Field(default_factory=list)
    route_snapshot: dict[str, Any] = Field(default_factory=dict)
    verifier_snapshot: dict[str, Any] = Field(default_factory=dict)
    symbolic_snapshot: dict[str, Any] = Field(default_factory=dict)
    final_selector_diagnostics: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class FailureRecord(BaseModel):
    problem_id: str
    gold_answer: str | None = None
    predicted_answer: str | None = None
    primary_failure: FailureBucket
    secondary_failures: list[str] = Field(default_factory=list)
    confidence_of_attribution: float = Field(0.0, ge=0.0, le=1.0)
    route_snapshot: dict[str, Any] = Field(default_factory=dict)
    verifier_snapshot: dict[str, Any] = Field(default_factory=dict)
    symbolic_snapshot: dict[str, Any] = Field(default_factory=dict)
    branch_snapshot: dict[str, Any] = Field(default_factory=dict)
    recommended_fix: str = ""
    attributions: list[FailureCause] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
