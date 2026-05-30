from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha1
import re
from typing import Any, Mapping, Sequence

from src.branches.branch_state import SymbolicEvidence
from src.common.schemas import FailureType, ProblemDomain
from src.state_graph.proof_obligations import ProofEvidenceKind, ProofObligation
from src.symbolic import algebra, geometry, number_theory


class ValidatorDomain(str, Enum):
    ALGEBRA = "algebra"
    NUMBER_THEORY = "number_theory"
    GEOMETRY = "geometry"


class ValidationTarget(str, Enum):
    BRANCH_STEP = "branch_step"
    CANDIDATE_ANSWER = "candidate_answer"
    CONSTRAINT_SLICE = "constraint_slice"
    RELATION_BUNDLE = "relation_bundle"
    STATEMENT = "statement"


class CompositeValidationStatus(str, Enum):
    SUCCESS = "success"
    CONTRADICTION = "contradiction"
    UNSUPPORTED = "unsupported"
    TIMEOUT = "timeout"
    MALFORMED_INPUT = "malformed_input"
    EXECUTION_FAILURE = "execution_failure"


@dataclass(frozen=True)
class ValidationDiagnostic:
    code: str
    message: str
    domain: ValidatorDomain | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationProvenance:
    domain: ValidatorDomain
    module_name: str
    function_name: str
    target: ValidationTarget


@dataclass(frozen=True)
class SubValidationRecord:
    provenance: ValidationProvenance
    status: CompositeValidationStatus
    summary: str
    score: float
    exact: bool
    diagnostics: tuple[ValidationDiagnostic, ...] = field(default_factory=tuple)
    evidence_payloads: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    derived_facts: tuple[str, ...] = field(default_factory=tuple)
    contradiction_found: bool = False
    failure_type: FailureType | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompositeValidationResult:
    status: CompositeValidationStatus
    target: ValidationTarget
    summary: str
    score: float
    exact: bool
    applied_domains: tuple[ValidatorDomain, ...]
    records: tuple[SubValidationRecord, ...] = field(default_factory=tuple)
    diagnostics: tuple[ValidationDiagnostic, ...] = field(default_factory=tuple)
    derived_constraints: tuple[str, ...] = field(default_factory=tuple)
    derived_invariants: tuple[str, ...] = field(default_factory=tuple)
    proof_obligations: tuple[ProofObligation, ...] = field(default_factory=tuple)
    discharged_obligation_ids: tuple[str, ...] = field(default_factory=tuple)
    contradicted_obligation_ids: tuple[str, ...] = field(default_factory=tuple)
    contradiction_found: bool = False
    failure_type: FailureType | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status is CompositeValidationStatus.SUCCESS

    def to_branch_symbolic_evidence(
        self,
        *,
        check_name: str = "symbolic_validation",
        supporting_step_ids: Sequence[str] = (),
    ) -> SymbolicEvidence:
        raw = (
            f"{self.target.value}|{self.summary}|"
            f"{tuple(domain.value for domain in self.applied_domains)}|"
            f"{sorted((self.metadata or {}).items())}"
        )
        evidence_id = f"sym_{sha1(raw.encode('utf-8')).hexdigest()[:16]}"
        return SymbolicEvidence(
            evidence_id=evidence_id,
            passed=self.passed,
            score=float(self.score),
            check_name=check_name,
            summary=self.summary,
            supporting_step_ids=tuple(supporting_step_ids),
            discharged_obligation_ids=self.discharged_obligation_ids,
            contradicted_obligation_ids=self.contradicted_obligation_ids,
        )


@dataclass(frozen=True)
class ValidatorDispatchPlan:
    target: ValidationTarget
    domains: tuple[ValidatorDomain, ...]
    reason: str


@dataclass(frozen=True)
class ValidationRequest:
    target: ValidationTarget
    statement: str | None = None
    candidate_answer: str | None = None
    constraints: tuple[str, ...] = ()
    relation_bundle: tuple[str, ...] = ()
    answer_symbol: str | None = None
    domain_hint: str | None = None
    validator_hint: str | None = None
    supporting_step_ids: tuple[str, ...] = ()
    proof_obligations: tuple[ProofObligation, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


_NT_HINT_RE = re.compile(
    r"\b(mod|congru|divides|gcd\(|lcm\(|even|odd|integer|divisor|multiple|prime|residue)\b|\|",
    re.IGNORECASE,
)
_GEOM_HINT_RE = re.compile(
    r"\b(collinear\(|parallel\(|perpendicular\(|midpoint\(|on_segment\(|equal_distance\(|"
    r"angle|triangle|circle|coordinate|point\b)",
    re.IGNORECASE,
)
_ALGEBRA_HINT_RE = re.compile(r"(<=|>=|!=|=|<|>)")


def _map_status(value: Any) -> CompositeValidationStatus:
    text = getattr(value, "value", value)
    return CompositeValidationStatus(str(text))


def _record_from_subresult(
    domain: ValidatorDomain,
    target: ValidationTarget,
    fn_name: str,
    result: Any,
) -> SubValidationRecord:
    diagnostics = tuple(
        ValidationDiagnostic(
            str(getattr(d, "code", "unknown")),
            getattr(d, "message", str(d)),
            domain,
            getattr(d, "payload", {}),
        )
        for d in getattr(result, "diagnostics", ())
    )
    evidences = tuple(
        {
            "label": getattr(e, "label", "evidence"),
            "expression": getattr(e, "expression", ""),
            "normalized_expression": getattr(
                e,
                "normalized_expression",
                getattr(e, "simplified_expression", ""),
            ),
            "exact_value": getattr(e, "exact_value", None),
            "note": getattr(e, "note", ""),
            "payload": getattr(e, "payload", {}),
        }
        for e in getattr(result, "evidence", ())
    )
    module_name = {
        ValidatorDomain.ALGEBRA: "src.symbolic.algebra",
        ValidatorDomain.NUMBER_THEORY: "src.symbolic.number_theory",
        ValidatorDomain.GEOMETRY: "src.symbolic.geometry",
    }[domain]
    return SubValidationRecord(
        provenance=ValidationProvenance(
            domain=domain,
            module_name=module_name,
            function_name=fn_name,
            target=target,
        ),
        status=_map_status(getattr(result, "status")),
        summary=getattr(result, "summary", ""),
        score=float(getattr(result, "score", 0.0)),
        exact=bool(getattr(result, "exact", False)),
        diagnostics=diagnostics,
        evidence_payloads=evidences,
        derived_facts=tuple(getattr(result, "derived_facts", ())),
        contradiction_found=bool(getattr(result, "contradiction_found", False)),
        failure_type=getattr(result, "failure_type", None),
        metadata=dict(getattr(result, "metadata", {}) or {}),
    )


def _summarize_records(status: CompositeValidationStatus, records: Sequence[SubValidationRecord]) -> str:
    if not records:
        return "No symbolic validator applied."
    if len(records) == 1:
        return records[0].summary

    success_count = sum(1 for record in records if record.status is CompositeValidationStatus.SUCCESS)
    unsupported_count = sum(1 for record in records if record.status is CompositeValidationStatus.UNSUPPORTED)
    contradiction_count = sum(
        1 for record in records if record.status is CompositeValidationStatus.CONTRADICTION
    )
    exact_count = sum(1 for record in records if record.exact)

    if status is CompositeValidationStatus.SUCCESS:
        return (
            f"Symbolic validation succeeded across {len(records)} sub-check(s); "
            f"{exact_count} were exact."
        )
    if status is CompositeValidationStatus.CONTRADICTION:
        return (
            f"Symbolic validation found contradiction evidence across {contradiction_count} "
            f"sub-check(s)."
        )
    if status is CompositeValidationStatus.UNSUPPORTED and success_count > 0:
        return (
            f"Symbolic validation produced partial support: {success_count} success, "
            f"{unsupported_count} unsupported, across {len(records)} sub-check(s)."
        )
    return f"Composite symbolic validation produced status {status.value} across {len(records)} sub-check(s)."


def _aggregate_records(
    target: ValidationTarget,
    records: Sequence[SubValidationRecord],
    *,
    request: ValidationRequest | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CompositeValidationResult:
    base_metadata = dict(metadata or {})
    if not records:
        return CompositeValidationResult(
            status=CompositeValidationStatus.UNSUPPORTED,
            target=target,
            summary="No symbolic validator applied.",
            score=0.15,
            exact=False,
            applied_domains=(),
            diagnostics=(ValidationDiagnostic("no_validator", "No validator applied for this request."),),
            proof_obligations=tuple(getattr(request, "proof_obligations", ()) or ()),
            failure_type=FailureType.COVERAGE_GAP,
            metadata=base_metadata,
        )

    statuses = [record.status for record in records]
    success_count = sum(1 for status in statuses if status is CompositeValidationStatus.SUCCESS)
    unsupported_count = sum(1 for status in statuses if status is CompositeValidationStatus.UNSUPPORTED)
    contradiction_count = sum(1 for status in statuses if status is CompositeValidationStatus.CONTRADICTION)
    timeout_count = sum(1 for status in statuses if status is CompositeValidationStatus.TIMEOUT)
    malformed_count = sum(1 for status in statuses if status is CompositeValidationStatus.MALFORMED_INPUT)
    execution_count = sum(1 for status in statuses if status is CompositeValidationStatus.EXECUTION_FAILURE)
    exact_success_count = sum(
        1
        for record in records
        if record.status is CompositeValidationStatus.SUCCESS and record.exact
    )

    if contradiction_count:
        status = CompositeValidationStatus.CONTRADICTION
        score = 0.0
        failure_type = FailureType.SYMBOLIC_MISMATCH
    elif execution_count:
        status = CompositeValidationStatus.EXECUTION_FAILURE
        score = 0.0
        failure_type = FailureType.SYMBOLIC_MISMATCH
    elif malformed_count:
        status = CompositeValidationStatus.MALFORMED_INPUT
        score = 0.05
        failure_type = FailureType.SYMBOLIC_MISMATCH
    elif records and all(item is CompositeValidationStatus.SUCCESS for item in statuses):
        status = CompositeValidationStatus.SUCCESS
        score = min(1.0, sum(record.score for record in records) / len(records))
        failure_type = None
    elif success_count:
        status = CompositeValidationStatus.UNSUPPORTED
        success_scores = [record.score for record in records if record.status is CompositeValidationStatus.SUCCESS]
        avg_success = sum(success_scores) / max(1, len(success_scores))
        score = min(0.42, max(0.18, avg_success * 0.40))
        failure_type = FailureType.COVERAGE_GAP
    elif timeout_count:
        status = CompositeValidationStatus.TIMEOUT
        score = 0.0
        failure_type = FailureType.TIMEOUT
    else:
        status = CompositeValidationStatus.UNSUPPORTED
        score = max((record.score for record in records), default=0.15)
        failure_type = FailureType.COVERAGE_GAP

    exact = bool(records) and all(
        record.exact
        for record in records
        if record.status in {CompositeValidationStatus.SUCCESS, CompositeValidationStatus.CONTRADICTION}
    )
    domains = tuple(dict.fromkeys(record.provenance.domain for record in records))
    diagnostics = tuple(diagnostic for record in records for diagnostic in record.diagnostics)
    derived = tuple(dict.fromkeys(fact for record in records for fact in record.derived_facts))
    summary = _summarize_records(status, records)

    base_metadata.update(
        {
            "symbolic_status": status.value,
            "symbolic_exact": exact,
            "record_count": len(records),
            "success_count": success_count,
            "unsupported_count": unsupported_count,
            "contradiction_count": contradiction_count,
            "timeout_count": timeout_count,
            "malformed_count": malformed_count,
            "execution_failure_count": execution_count,
            "exact_success_count": exact_success_count,
            "partial_support": status is CompositeValidationStatus.UNSUPPORTED and success_count > 0,
            "support_ratio": success_count / len(records),
            "exact_support_ratio": exact_success_count / len(records),
            "unsupported_ratio": unsupported_count / len(records),
            "applied_domains": [domain.value for domain in domains],
        }
    )
    proof_obligations, discharged_ids, contradicted_ids = _update_proof_obligations(
        request=request,
        status=status,
        summary=summary,
    )
    base_metadata.update(
        {
            "discharged_obligation_ids": list(discharged_ids),
            "contradicted_obligation_ids": list(contradicted_ids),
            "discharged_obligation_count": len(discharged_ids),
            "contradicted_obligation_count": len(contradicted_ids),
            "contradiction_found": any(record.contradiction_found for record in records),
        }
    )

    return CompositeValidationResult(
        status=status,
        target=target,
        summary=summary,
        score=score,
        exact=exact,
        applied_domains=domains,
        records=tuple(records),
        diagnostics=diagnostics,
        derived_constraints=derived,
        derived_invariants=(),
        proof_obligations=proof_obligations,
        discharged_obligation_ids=discharged_ids,
        contradicted_obligation_ids=contradicted_ids,
        contradiction_found=any(record.contradiction_found for record in records),
        failure_type=failure_type,
        metadata=base_metadata,
    )


def _update_proof_obligations(
    *,
    request: ValidationRequest | None,
    status: CompositeValidationStatus,
    summary: str,
) -> tuple[tuple[ProofObligation, ...], tuple[str, ...], tuple[str, ...]]:
    obligations = tuple(getattr(request, "proof_obligations", ()) or ())
    if not obligations:
        return (), (), ()

    discharged: list[str] = []
    contradicted: list[str] = []
    updated: list[ProofObligation] = []
    for obligation in obligations:
        if obligation.evidence_kind_required not in {
            ProofEvidenceKind.SYMBOLIC_CHECK,
            ProofEvidenceKind.CONSTRAINT,
            ProofEvidenceKind.GOAL,
            ProofEvidenceKind.OTHER,
        }:
            updated.append(obligation)
            continue

        if status is CompositeValidationStatus.SUCCESS and obligation.is_open():
            refreshed = obligation.discharge(
                discharged_by="symbolic_validation",
                notes=(summary,),
                metadata={
                    "validator_target": getattr(request, "target", None).value if request is not None else None,
                },
            )
            discharged.append(refreshed.obligation_id)
            updated.append(refreshed)
            continue

        if status is CompositeValidationStatus.CONTRADICTION and obligation.is_open():
            refreshed = obligation.contradict(
                source="symbolic_validation",
                summary=summary,
                failure_type=FailureType.SYMBOLIC_MISMATCH.value,
                metadata={
                    "validator_target": getattr(request, "target", None).value if request is not None else None,
                },
            )
            contradicted.append(refreshed.obligation_id)
            updated.append(refreshed)
            continue

        updated.append(obligation)

    return tuple(updated), tuple(discharged), tuple(contradicted)


def _hint_to_domain(value: str | None) -> ValidatorDomain | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {ProblemDomain.ALGEBRA.value, ValidatorDomain.ALGEBRA.value}:
        return ValidatorDomain.ALGEBRA
    if text in {ProblemDomain.NUMBER_THEORY.value, ValidatorDomain.NUMBER_THEORY.value, "nt"}:
        return ValidatorDomain.NUMBER_THEORY
    if text in {ProblemDomain.GEOMETRY.value, ValidatorDomain.GEOMETRY.value}:
        return ValidatorDomain.GEOMETRY
    return None


def build_dispatch_plan(request: ValidationRequest) -> ValidatorDispatchPlan:
    explicit = _hint_to_domain(request.validator_hint) or _hint_to_domain(request.domain_hint)
    if explicit is not None:
        return ValidatorDispatchPlan(
            request.target,
            (explicit,),
            f"explicit_hint:{explicit.value}",
        )

    texts = " ".join(
        filter(
            None,
            [
                request.statement,
                request.candidate_answer,
                *request.constraints,
                *request.relation_bundle,
            ],
        )
    )

    if _GEOM_HINT_RE.search(texts):
        return ValidatorDispatchPlan(
            request.target,
            (ValidatorDomain.GEOMETRY,),
            "geometry_tokens",
        )

    domains: list[ValidatorDomain] = []
    if _NT_HINT_RE.search(texts):
        domains.append(ValidatorDomain.NUMBER_THEORY)
    if _ALGEBRA_HINT_RE.search(texts) or request.target in {
        ValidationTarget.CANDIDATE_ANSWER,
        ValidationTarget.CONSTRAINT_SLICE,
        ValidationTarget.RELATION_BUNDLE,
    }:
        domains.append(ValidatorDomain.ALGEBRA)
    if not domains:
        domains.append(ValidatorDomain.ALGEBRA)

    reason = "mixed_symbolic" if len(domains) > 1 else f"default_{domains[0].value}"
    return ValidatorDispatchPlan(request.target, tuple(dict.fromkeys(domains)), reason)


def _run_algebra_records(request: ValidationRequest) -> list[SubValidationRecord]:
    records: list[SubValidationRecord] = []
    if request.target in {ValidationTarget.STATEMENT, ValidationTarget.BRANCH_STEP} and request.statement:
        records.append(
            _record_from_subresult(
                ValidatorDomain.ALGEBRA,
                request.target,
                "validate_algebra_statement",
                algebra.validate_algebra_statement(request.statement, request.constraints),
            )
        )
    elif request.target is ValidationTarget.CANDIDATE_ANSWER and request.candidate_answer is not None:
        records.append(
            _record_from_subresult(
                ValidatorDomain.ALGEBRA,
                request.target,
                "validate_candidate_answer",
                algebra.validate_candidate_answer(
                    request.candidate_answer,
                    request.constraints,
                    request.answer_symbol,
                ),
            )
        )
    elif request.target is ValidationTarget.CONSTRAINT_SLICE:
        records.append(
            _record_from_subresult(
                ValidatorDomain.ALGEBRA,
                request.target,
                "validate_constraint_slice",
                algebra.validate_constraint_slice(
                    request.constraints,
                    request.candidate_answer,
                    request.answer_symbol,
                ),
            )
        )
    elif request.target is ValidationTarget.RELATION_BUNDLE:
        for item in request.relation_bundle:
            records.append(
                _record_from_subresult(
                    ValidatorDomain.ALGEBRA,
                    request.target,
                    "validate_algebra_statement",
                    algebra.validate_algebra_statement(item, request.constraints),
                )
            )
    return records


def _run_number_theory_records(request: ValidationRequest) -> list[SubValidationRecord]:
    records: list[SubValidationRecord] = []
    if request.target in {ValidationTarget.STATEMENT, ValidationTarget.BRANCH_STEP} and request.statement:
        records.append(
            _record_from_subresult(
                ValidatorDomain.NUMBER_THEORY,
                request.target,
                "validate_number_theory_statement",
                number_theory.validate_number_theory_statement(request.statement, request.constraints),
            )
        )
    elif request.target is ValidationTarget.CANDIDATE_ANSWER and request.candidate_answer is not None:
        records.append(
            _record_from_subresult(
                ValidatorDomain.NUMBER_THEORY,
                request.target,
                "validate_candidate_answer",
                number_theory.validate_candidate_answer(
                    request.candidate_answer,
                    request.constraints,
                    request.answer_symbol,
                ),
            )
        )
    elif request.target is ValidationTarget.CONSTRAINT_SLICE:
        records.append(
            _record_from_subresult(
                ValidatorDomain.NUMBER_THEORY,
                request.target,
                "validate_constraint_slice",
                number_theory.validate_constraint_slice(
                    request.constraints,
                    request.candidate_answer,
                    request.answer_symbol,
                ),
            )
        )
    elif request.target is ValidationTarget.RELATION_BUNDLE:
        for item in request.relation_bundle:
            records.append(
                _record_from_subresult(
                    ValidatorDomain.NUMBER_THEORY,
                    request.target,
                    "validate_number_theory_statement",
                    number_theory.validate_number_theory_statement(item, request.constraints),
                )
            )
    return records


def _run_geometry_records(request: ValidationRequest) -> list[SubValidationRecord]:
    records: list[SubValidationRecord] = []
    if request.target in {ValidationTarget.STATEMENT, ValidationTarget.BRANCH_STEP} and request.statement:
        records.append(
            _record_from_subresult(
                ValidatorDomain.GEOMETRY,
                request.target,
                "validate_geometry_statement",
                geometry.validate_geometry_statement(request.statement, request.constraints),
            )
        )
    elif request.target is ValidationTarget.CANDIDATE_ANSWER and request.candidate_answer is not None:
        records.append(
            _record_from_subresult(
                ValidatorDomain.GEOMETRY,
                request.target,
                "validate_candidate_answer",
                geometry.validate_candidate_answer(
                    request.candidate_answer,
                    request.constraints,
                    request.answer_symbol,
                ),
            )
        )
    elif request.target is ValidationTarget.CONSTRAINT_SLICE:
        records.append(
            _record_from_subresult(
                ValidatorDomain.GEOMETRY,
                request.target,
                "validate_constraint_slice",
                geometry.validate_constraint_slice(request.constraints),
            )
        )
    elif request.target is ValidationTarget.RELATION_BUNDLE:
        for item in request.relation_bundle:
            records.append(
                _record_from_subresult(
                    ValidatorDomain.GEOMETRY,
                    request.target,
                    "validate_geometry_statement",
                    geometry.validate_geometry_statement(item, request.constraints),
                )
            )
    return records


def validate_symbolic_request(request: ValidationRequest) -> CompositeValidationResult:
    plan = build_dispatch_plan(request)
    if not plan.domains:
        return _aggregate_records(
            request.target,
            (),
            request=request,
            metadata={"dispatch_reason": plan.reason},
        )

    records: list[SubValidationRecord] = []
    for domain in plan.domains:
        if domain is ValidatorDomain.ALGEBRA:
            records.extend(_run_algebra_records(request))
        elif domain is ValidatorDomain.NUMBER_THEORY:
            records.extend(_run_number_theory_records(request))
        else:
            records.extend(_run_geometry_records(request))

    return _aggregate_records(
        request.target,
        records,
        request=request,
        metadata={"dispatch_reason": plan.reason},
    )


def validate_branch_step(
    statement: str,
    *,
    constraints: Sequence[str] = (),
    domain_hint: str | None = None,
    validator_hint: str | None = None,
    supporting_step_ids: Sequence[str] = (),
    proof_obligations: Sequence[ProofObligation] = (),
) -> CompositeValidationResult:
    return validate_symbolic_request(
        ValidationRequest(
            target=ValidationTarget.BRANCH_STEP,
            statement=statement,
            constraints=tuple(constraints),
            domain_hint=domain_hint,
            validator_hint=validator_hint,
            supporting_step_ids=tuple(supporting_step_ids),
            proof_obligations=tuple(proof_obligations),
        )
    )


def validate_candidate_answer(
    candidate_answer: str,
    *,
    constraints: Sequence[str] = (),
    answer_symbol: str | None = None,
    domain_hint: str | None = None,
    validator_hint: str | None = None,
    supporting_step_ids: Sequence[str] = (),
) -> CompositeValidationResult:
    return validate_symbolic_request(
        ValidationRequest(
            target=ValidationTarget.CANDIDATE_ANSWER,
            candidate_answer=candidate_answer,
            constraints=tuple(constraints),
            answer_symbol=answer_symbol,
            domain_hint=domain_hint,
            validator_hint=validator_hint,
            supporting_step_ids=tuple(supporting_step_ids),
        )
    )


def validate_constraint_slice(
    *,
    constraints: Sequence[str],
    candidate_answer: str | None = None,
    answer_symbol: str | None = None,
    domain_hint: str | None = None,
    validator_hint: str | None = None,
) -> CompositeValidationResult:
    return validate_symbolic_request(
        ValidationRequest(
            target=ValidationTarget.CONSTRAINT_SLICE,
            constraints=tuple(constraints),
            candidate_answer=candidate_answer,
            answer_symbol=answer_symbol,
            domain_hint=domain_hint,
            validator_hint=validator_hint,
        )
    )


def validate_relation_bundle(
    bundle: Sequence[str],
    *,
    constraints: Sequence[str] = (),
    domain_hint: str | None = None,
    validator_hint: str | None = None,
) -> CompositeValidationResult:
    return validate_symbolic_request(
        ValidationRequest(
            target=ValidationTarget.RELATION_BUNDLE,
            relation_bundle=tuple(bundle),
            constraints=tuple(constraints),
            domain_hint=domain_hint,
            validator_hint=validator_hint,
        )
    )


def validate_single_statement(
    statement: str,
    *,
    constraints: Sequence[str] = (),
    domain_hint: str | None = None,
    validator_hint: str | None = None,
) -> CompositeValidationResult:
    return validate_symbolic_request(
        ValidationRequest(
            target=ValidationTarget.STATEMENT,
            statement=statement,
            constraints=tuple(constraints),
            domain_hint=domain_hint,
            validator_hint=validator_hint,
        )
    )


VALIDATOR_REGISTRY: Mapping[str, Any] = {}


__all__ = [
    "ValidatorDomain",
    "ValidationTarget",
    "CompositeValidationStatus",
    "ValidationDiagnostic",
    "ValidationProvenance",
    "SubValidationRecord",
    "CompositeValidationResult",
    "ValidatorDispatchPlan",
    "ValidationRequest",
    "build_dispatch_plan",
    "validate_symbolic_request",
    "validate_branch_step",
    "validate_candidate_answer",
    "validate_constraint_slice",
    "validate_relation_bundle",
    "validate_single_statement",
    "VALIDATOR_REGISTRY",
]
