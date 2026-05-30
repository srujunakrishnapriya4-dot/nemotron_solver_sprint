"""Strict schemas for the Pass 2 canonical proof spine."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


class SchemaValidationError(ValueError):
    """Raised when a proof-spine schema is invalid."""


class RegistryError(ValueError):
    """Raised when registry validation fails."""


class ManifestError(ValueError):
    """Raised when artifact manifest validation fails."""


class CanonicalizationError(ValueError):
    """Raised when prompt canonicalization fails."""


class ProgramExecutionError(ValueError):
    """Raised when a DSL program cannot execute safely."""


class ProgramVerificationError(ValueError):
    """Raised when program verification fails."""


class AmbiguityError(ValueError):
    """Raised when ambiguity reporting cannot be built."""


class LeakageError(ValueError):
    """Raised when leakage checks cannot be completed."""


class ProblemSource(str, Enum):
    OFFICIAL_TRAIN = "official_train"
    PUBLIC_LIKE = "public_like"
    SYNTHETIC_DSL = "synthetic_dsl"
    MUTATION = "mutation"
    LLM_STRESS = "llm_stress"
    FORBIDDEN_PRIMITIVE = "forbidden_primitive"
    MANUAL_FIXTURE = "manual_fixture"


class SplitName(str, Enum):
    TRAIN = "train"
    DEV = "dev"
    HARD_DEV = "hard_dev"
    PRIVATE_LIKE = "private_like"
    FORBIDDEN_HOLDOUT = "forbidden_holdout"
    STRESS_ONLY = "stress_only"


class ParsePermission(str, Enum):
    NO_SFT = "no_sft"
    STRESS_EVAL_ONLY = "stress_eval_only"
    SOLVER_ALLOWED = "solver_allowed"


class DomainKind(str, Enum):
    INTEGER = "integer"
    DECIMAL = "decimal"
    FRACTION = "fraction"
    DIGIT_SEQUENCE = "digit_sequence"
    BITSTRING = "bitstring"
    SYMBOL_SEQUENCE = "symbol_sequence"
    CIPHER_SEQUENCE = "cipher_sequence"
    ROMAN_LIKE = "roman_like"
    EQUATION = "equation"
    TOKEN_SEQUENCE = "token_sequence"
    UNKNOWN = "unknown"


class ProgramStepKind(str, Enum):
    PARSE = "parse"
    SELECT = "select"
    TRANSFORM = "transform"
    FORMAT = "format"


class VerificationStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    AMBIGUOUS = "ambiguous"
    REJECTED = "rejected"


ALLOWED_PRIMITIVES: dict[ProgramStepKind, set[str]] = {
    ProgramStepKind.PARSE: {
        "parse_int",
        "parse_digits",
        "parse_bitstring",
        "parse_symbols",
        "parse_token_sequence",
        "parse_binary_int_expr",
    },
    ProgramStepKind.SELECT: {"identity", "lhs", "rhs", "operand_i", "digit_i", "bit_i", "symbol_i"},
    ProgramStepKind.TRANSFORM: {
        "add_const",
        "sub_const",
        "mul_const",
        "affine_small",
        "reverse",
        "digit_sum",
        "concat",
        "bit_not",
        "binary_op",
        "symbol_bijection",
    },
    ProgramStepKind.FORMAT: {"raw", "zero_pad", "bitstring", "encode_symbols"},
}


def stable_json_dumps(obj: Any) -> str:
    return json.dumps(_to_stable_obj(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(stable_json_dumps(obj).encode("utf-8")).hexdigest()


def ensure_non_empty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError(f"{field_name} must be a non-empty string.")
    return value


def ensure_probability(value: float, field_name: str) -> float:
    if isinstance(value, bool):
        raise SchemaValidationError(f"{field_name} must be a probability, not boolean.")
    if not isinstance(value, (int, float)):
        raise SchemaValidationError(f"{field_name} must be an int or float probability.")
    if value < 0.0 or value > 1.0:
        raise SchemaValidationError(f"{field_name} must be in [0, 1].")
    return float(value)


def coerce_enum(enum_cls: type[Enum], value: Any, field_name: str) -> Enum:
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(str(value))
    except ValueError as exc:
        raise SchemaValidationError(f"{field_name} has invalid value: {value!r}.") from exc


@dataclass(frozen=True)
class ImmutableProblemRecord:
    problem_id: str
    source: ProblemSource
    source_hash: str
    family_id: str
    primitive_family_id: str
    format_family_id: str
    prompt_wrapper_id: str
    split: SplitName
    created_from: str = ""
    parent_ids: tuple[str, ...] = ()
    contamination_flags: tuple[str, ...] = ()
    allowed_for_sft: bool = False
    allowed_for_dpo: bool = False
    allowed_for_grpo: bool = False
    allowed_for_eval: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "problem_id", ensure_non_empty(self.problem_id, "problem_id"))
        object.__setattr__(self, "source", coerce_enum(ProblemSource, self.source, "source"))
        object.__setattr__(self, "split", coerce_enum(SplitName, self.split, "split"))
        for name in ("source_hash", "family_id", "primitive_family_id", "format_family_id", "prompt_wrapper_id"):
            object.__setattr__(self, name, ensure_non_empty(getattr(self, name), name))
        object.__setattr__(self, "parent_ids", tuple(str(item) for item in self.parent_ids))
        object.__setattr__(self, "contamination_flags", tuple(str(item) for item in self.contamination_flags))
        if self.split in {SplitName.FORBIDDEN_HOLDOUT, SplitName.STRESS_ONLY}:
            if self.allowed_for_sft or self.allowed_for_dpo or self.allowed_for_grpo:
                raise SchemaValidationError(f"{self.split.value} cannot allow SFT/DPO/GRPO.")


@dataclass(frozen=True)
class ExamplePair:
    input_value: str
    output_value: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_value", ensure_non_empty(str(self.input_value), "input_value"))
        object.__setattr__(self, "output_value", ensure_non_empty(str(self.output_value), "output_value"))


@dataclass(frozen=True)
class TargetQuery:
    input_value: str
    expected_output: str | None = None
    certain: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_value", ensure_non_empty(str(self.input_value), "target.input_value"))
        if self.expected_output is not None:
            object.__setattr__(self, "expected_output", str(self.expected_output))


@dataclass(frozen=True)
class DomainSignature:
    kind: DomainKind
    raw_value: str = ""
    normalized_value: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", coerce_enum(DomainKind, self.kind, "domain.kind"))
        object.__setattr__(self, "raw_value", str(self.raw_value))
        object.__setattr__(self, "normalized_value", str(self.normalized_value))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class OperatorCandidate:
    operator_id: str
    name: str
    confidence: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "operator_id", ensure_non_empty(self.operator_id, "operator_id"))
        object.__setattr__(self, "name", ensure_non_empty(self.name, "name"))
        object.__setattr__(self, "confidence", ensure_probability(self.confidence, "confidence"))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class SymbolTable:
    symbols: Mapping[str, str] = field(default_factory=dict)
    normalized_symbols: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        symbols = dict(self.symbols)
        normalized = dict(self.normalized_symbols or symbols)
        if len(symbols) != len(set(symbols.keys())):
            raise SchemaValidationError("SymbolTable contains duplicate symbols.")
        values = list(normalized.values())
        if len(values) != len(set(values)):
            raise SchemaValidationError("SymbolTable normalized symbol collision.")
        if set(symbols.keys()) != set(normalized.keys()):
            raise SchemaValidationError("SymbolTable symbols and normalized_symbols keys must match.")
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "normalized_symbols", normalized)


@dataclass(frozen=True)
class NumericPolicy:
    domain: DomainKind
    allow_leading_zeros: bool = False
    decimal_places: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "domain", coerce_enum(DomainKind, self.domain, "domain"))
        if self.decimal_places is not None and self.decimal_places < 0:
            raise SchemaValidationError("decimal_places must be non-negative.")


@dataclass(frozen=True)
class FormatPolicyRef:
    answer_type: str
    preserve_leading_zeros: bool = False
    decimal_places: int | None = None
    preserve_case: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "answer_type", ensure_non_empty(str(self.answer_type), "answer_type"))
        if self.decimal_places is not None and self.decimal_places < 0:
            raise SchemaValidationError("decimal_places must be non-negative.")


@dataclass(frozen=True)
class CanonicalProblem:
    problem_id: str
    raw_prompt: str
    examples: tuple[ExamplePair, ...]
    target: TargetQuery
    input_domain: DomainSignature
    output_domain: DomainSignature
    symbol_table: SymbolTable = field(default_factory=SymbolTable)
    parse_confidence: float = 0.0
    round_trip_score: float = 0.0
    parse_hash: str = ""
    parser_warnings: tuple[str, ...] = ()
    parse_permission: ParsePermission = ParsePermission.NO_SFT

    def __post_init__(self) -> None:
        object.__setattr__(self, "problem_id", ensure_non_empty(self.problem_id, "problem_id"))
        object.__setattr__(self, "raw_prompt", ensure_non_empty(self.raw_prompt, "raw_prompt"))
        examples = tuple(self.examples)
        if not examples:
            raise SchemaValidationError("CanonicalProblem requires at least one example.")
        if not isinstance(self.target, TargetQuery):
            raise SchemaValidationError("CanonicalProblem target must be a TargetQuery.")
        object.__setattr__(self, "examples", examples)
        object.__setattr__(self, "parse_confidence", ensure_probability(self.parse_confidence, "parse_confidence"))
        object.__setattr__(self, "round_trip_score", ensure_probability(self.round_trip_score, "round_trip_score"))
        object.__setattr__(self, "parser_warnings", tuple(str(item) for item in self.parser_warnings))
        object.__setattr__(self, "parse_permission", coerce_enum(ParsePermission, self.parse_permission, "parse_permission"))
        if self.parse_confidence >= 0.92 and not _target_is_structurally_valid(self.target):
            raise SchemaValidationError("High-confidence CanonicalProblem requires a certain, non-placeholder target.")
        if not self.parse_hash:
            object.__setattr__(self, "parse_hash", stable_hash(_hash_fields(self, exclude={"parse_hash"})))


@dataclass(frozen=True)
class ProgramStep:
    kind: ProgramStepKind
    primitive: str
    args: Mapping[str, Any] = field(default_factory=dict)
    output_key: str = "value"

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", coerce_enum(ProgramStepKind, self.kind, "kind"))
        object.__setattr__(self, "primitive", ensure_non_empty(self.primitive, "primitive"))
        object.__setattr__(self, "output_key", ensure_non_empty(self.output_key, "output_key"))
        object.__setattr__(self, "args", dict(self.args))
        if self.primitive not in ALLOWED_PRIMITIVES[self.kind]:
            raise SchemaValidationError(f"Primitive {self.primitive!r} is not allowed for step kind {self.kind.value}.")


@dataclass(frozen=True)
class Program:
    program_id: str
    steps: tuple[ProgramStep, ...]
    output_key: str
    program_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "program_id", ensure_non_empty(self.program_id, "program_id"))
        steps = tuple(self.steps)
        if not steps:
            raise SchemaValidationError("Program must contain at least one step.")
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "output_key", ensure_non_empty(self.output_key, "output_key"))
        object.__setattr__(
            self,
            "program_hash",
            stable_hash(
                {
                    "program_id": self.program_id,
                    "steps": self.steps,
                    "output_key": self.output_key,
                }
            ),
        )


@dataclass(frozen=True)
class ExecutionTrace:
    input_value: str
    output_value: str | None = None
    expected_value: str | None = None
    passed: bool | None = None
    intermediate_values: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_value", str(self.input_value))
        if self.output_value is not None:
            object.__setattr__(self, "output_value", str(self.output_value))
        if self.expected_value is not None:
            object.__setattr__(self, "expected_value", str(self.expected_value))
        object.__setattr__(self, "intermediate_values", dict(self.intermediate_values))


@dataclass(frozen=True)
class CandidateProgram:
    program: Program
    example_executions: tuple[ExecutionTrace, ...] = ()
    target_execution: ExecutionTrace | None = None
    score: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "example_executions", tuple(self.example_executions))
        object.__setattr__(self, "score", ensure_probability(self.score, "score"))


@dataclass(frozen=True)
class RejectedProgram:
    program_id: str
    reason: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "program_id", ensure_non_empty(self.program_id, "program_id"))
        object.__setattr__(self, "reason", ensure_non_empty(self.reason, "reason"))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class AmbiguityReport:
    candidate_count: int
    target_outputs: tuple[str, ...] = ()
    unique_target_output: bool = False
    ambiguous: bool = True
    reason: str = ""

    def __post_init__(self) -> None:
        if self.candidate_count < 0:
            raise SchemaValidationError("candidate_count must be non-negative.")
        if self.candidate_count == 0 and (self.unique_target_output is True or self.ambiguous is False):
            raise SchemaValidationError("Zero-candidate AmbiguityReport must be ambiguous and non-unique.")
        if self.unique_target_output is True and self.ambiguous is True:
            raise SchemaValidationError("AmbiguityReport cannot be both unique and ambiguous.")
        object.__setattr__(self, "target_outputs", tuple(str(item) for item in self.target_outputs))


@dataclass(frozen=True)
class LeakageReport:
    has_leakage: bool = False
    leakage_types: tuple[str, ...] = ()
    details: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "leakage_types", tuple(str(item) for item in self.leakage_types))
        object.__setattr__(self, "details", tuple(str(item) for item in self.details))


@dataclass(frozen=True)
class FormatReport:
    valid: bool
    normalized_output: str | None = None
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.normalized_output is not None:
            object.__setattr__(self, "normalized_output", str(self.normalized_output))
        object.__setattr__(self, "errors", tuple(str(item) for item in self.errors))


@dataclass(frozen=True)
class ExecutableProof:
    problem_id: str
    parse_hash: str
    program_hash: str
    locked_program: Program
    rejected_programs: tuple[RejectedProgram, ...]
    example_executions: tuple[ExecutionTrace, ...]
    target_execution: ExecutionTrace | None
    ambiguity_report: AmbiguityReport
    leakage_report: LeakageReport
    format_report: FormatReport
    primary_verifier_pass: bool
    shadow_verifier_pass: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "problem_id", ensure_non_empty(self.problem_id, "problem_id"))
        object.__setattr__(self, "parse_hash", ensure_non_empty(self.parse_hash, "parse_hash"))
        object.__setattr__(self, "program_hash", ensure_non_empty(self.program_hash, "program_hash"))
        if not isinstance(self.locked_program, Program):
            raise SchemaValidationError("locked_program must be a Program.")
        if self.program_hash != self.locked_program.program_hash:
            raise SchemaValidationError("program_hash must match locked_program.program_hash.")
        if self.target_execution is None:
            raise SchemaValidationError("ExecutableProof requires target_execution.")
        object.__setattr__(self, "rejected_programs", tuple(self.rejected_programs))
        object.__setattr__(self, "example_executions", tuple(self.example_executions))

    @property
    def is_training_safe(self) -> bool:
        target = self.target_execution
        return (
            self.primary_verifier_pass is True
            and self.shadow_verifier_pass is True
            and bool(self.example_executions)
            and self.program_hash == self.locked_program.program_hash
            and self.ambiguity_report.candidate_count > 0
            and self.ambiguity_report.unique_target_output is True
            and self.ambiguity_report.ambiguous is False
            and self.leakage_report.has_leakage is False
            and self.format_report.valid is True
            and all(trace.passed is True for trace in self.example_executions)
            and target is not None
            and target.output_value is not None
            and str(target.output_value) != ""
            and target.error is None
        )


@dataclass(frozen=True)
class ArtifactManifest:
    stage: str
    input_manifest_hash: str | None
    output_manifest_hash: str
    schema_version: str
    row_count: int
    accepted_count: int
    rejected_count: int
    rejection_summary: Mapping[str, int] = field(default_factory=dict)
    code_git_hash: str = ""
    config_hash: str = ""
    created_at: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", ensure_non_empty(self.stage, "stage"))
        object.__setattr__(self, "output_manifest_hash", ensure_non_empty(self.output_manifest_hash, "output_manifest_hash"))
        object.__setattr__(self, "schema_version", ensure_non_empty(self.schema_version, "schema_version"))
        for name in ("row_count", "accepted_count", "rejected_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise SchemaValidationError(f"{name} must be a non-negative integer.")
        if self.accepted_count + self.rejected_count > self.row_count:
            raise SchemaValidationError("accepted_count + rejected_count cannot exceed row_count.")
        object.__setattr__(self, "rejection_summary", dict(self.rejection_summary))


@dataclass(frozen=True)
class QuarantineRecord:
    problem_id: str
    reason: str
    record_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "problem_id", ensure_non_empty(self.problem_id, "problem_id"))
        object.__setattr__(self, "reason", ensure_non_empty(self.reason, "reason"))
        object.__setattr__(self, "record_hash", ensure_non_empty(self.record_hash, "record_hash"))


def _hash_fields(instance: Any, *, exclude: set[str]) -> dict[str, Any]:
    return {item.name: getattr(instance, item.name) for item in fields(instance) if item.name not in exclude}


def _target_is_structurally_valid(target: TargetQuery) -> bool:
    if target.certain is not True:
        return False
    value = target.input_value.strip()
    if not value:
        return False
    return value.upper() not in {"UNKNOWN", "?", "TODO", "MISSING", "NONE", "NULL"}


def _to_stable_obj(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj):
        return {item.name: _to_stable_obj(getattr(obj, item.name)) for item in fields(obj)}
    if isinstance(obj, Mapping):
        return {str(key): _to_stable_obj(value) for key, value in sorted(obj.items(), key=lambda item: str(item[0]))}
    if isinstance(obj, (list, tuple)):
        return [_to_stable_obj(item) for item in obj]
    if isinstance(obj, set):
        return sorted(_to_stable_obj(item) for item in obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


__all__ = [
    "AmbiguityError",
    "AmbiguityReport",
    "ALLOWED_PRIMITIVES",
    "ArtifactManifest",
    "CandidateProgram",
    "CanonicalProblem",
    "CanonicalizationError",
    "DomainKind",
    "DomainSignature",
    "ExamplePair",
    "ExecutableProof",
    "ExecutionTrace",
    "FormatPolicyRef",
    "FormatReport",
    "ImmutableProblemRecord",
    "LeakageError",
    "LeakageReport",
    "ManifestError",
    "NumericPolicy",
    "OperatorCandidate",
    "ParsePermission",
    "ProblemSource",
    "Program",
    "ProgramExecutionError",
    "ProgramStep",
    "ProgramStepKind",
    "ProgramVerificationError",
    "QuarantineRecord",
    "RegistryError",
    "RejectedProgram",
    "SchemaValidationError",
    "SplitName",
    "SymbolTable",
    "TargetQuery",
    "VerificationStatus",
    "coerce_enum",
    "ensure_non_empty",
    "ensure_probability",
    "stable_hash",
    "stable_json_dumps",
]
