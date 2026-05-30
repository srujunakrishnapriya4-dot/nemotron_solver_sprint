# src/symbolic/sympy_runner.py
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from enum import Enum
from hashlib import sha1
import json
import time
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field

from src.state_graph.proof_obligations import ProofObligation
from src.symbolic import validators


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_assignment=True,
        str_strip_whitespace=True,
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )


class SymbolicStatus(str, Enum):
    SUCCESS = "success"
    CONTRADICTION = "contradiction"
    UNSUPPORTED = "unsupported"
    TIMEOUT = "timeout"
    MALFORMED_INPUT = "malformed_input"
    EXECUTION_FAILURE = "execution_failure"


class SymbolicTarget(str, Enum):
    BRANCH_STEP = "branch_step"
    CANDIDATE_ANSWER = "candidate_answer"
    CONSTRAINT_SLICE = "constraint_slice"
    RELATION_BUNDLE = "relation_bundle"
    STATEMENT = "statement"


class SymbolicValidatorKind(str, Enum):
    AUTO = "auto"
    ALGEBRA = "algebra"
    NUMBER_THEORY = "number_theory"
    GEOMETRY = "geometry"


class SymbolicDiagnosticCode(str, Enum):
    TIMEOUT = "timeout"
    EXECUTION_ERROR = "execution_error"
    REQUEST_ERROR = "request_error"
    SUCCESS = "success"
    CONTRADICTION = "contradiction"
    UNSUPPORTED = "unsupported"
    MALFORMED_INPUT = "malformed_input"


class SymbolicProvenance(StrictModel):
    problem_id: str | None = None
    branch_id: str | None = None
    step_id: str | None = None
    node_id: str | None = None
    operator_name: str | None = None
    source_module: str = "src.symbolic.sympy_runner"
    request_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class SymbolicDiagnostic(StrictModel):
    code: SymbolicDiagnosticCode
    message: str
    validator_kind: SymbolicValidatorKind
    payload: dict[str, Any] = Field(default_factory=dict)


class SymbolicEvidence(StrictModel):
    evidence_id: str
    status: SymbolicStatus
    validator_kind: SymbolicValidatorKind
    validator_name: str
    source_module: str
    check_name: str
    summary: str
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    exact_match: bool = False
    payload: dict[str, Any] = Field(default_factory=dict)


class SymbolicRunnerConfig(StrictModel):
    default_timeout_ms: int = Field(default=250, ge=1)
    deterministic: bool = True


class SymbolicRunRequest(StrictModel):
    target: SymbolicTarget
    statement: str | None = None
    candidate_answer: str | None = None
    constraints: tuple[str, ...] = Field(default_factory=tuple)
    relation_bundle: tuple[str, ...] = Field(default_factory=tuple)
    answer_symbol: str | None = None
    problem_id: str | None = None
    branch_id: str | None = None
    step_id: str | None = None
    node_id: str | None = None
    operator_name: str | None = None
    domain_hint: str | None = None
    validator_hint: SymbolicValidatorKind = SymbolicValidatorKind.AUTO
    proof_obligations: tuple[ProofObligation, ...] = Field(default_factory=tuple)
    time_budget_ms: int | None = Field(default=None, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    request_id: str = ""

    def model_post_init(self, __context: Any) -> None:
        if self.request_id:
            return
        payload = {
            "target": self.target.value,
            "statement": self.statement,
            "candidate_answer": self.candidate_answer,
            "constraints": list(self.constraints),
            "relation_bundle": list(self.relation_bundle),
            "answer_symbol": self.answer_symbol,
            "problem_id": self.problem_id,
            "branch_id": self.branch_id,
            "step_id": self.step_id,
        }
        object.__setattr__(self, "request_id", _stable_hash("symbolic_request", payload))


class SymbolicRunResult(StrictModel):
    request_id: str
    target: SymbolicTarget
    status: SymbolicStatus
    validator_kind: SymbolicValidatorKind
    validator_name: str
    summary: str
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    exact_match: bool = False
    diagnostics: tuple[SymbolicDiagnostic, ...] = Field(default_factory=tuple)
    evidence: tuple[SymbolicEvidence, ...] = Field(default_factory=tuple)
    provenance: SymbolicProvenance
    added_constraints: tuple[str, ...] = Field(default_factory=tuple)
    added_invariants: tuple[str, ...] = Field(default_factory=tuple)
    proof_obligations: tuple[ProofObligation, ...] = Field(default_factory=tuple)
    discharged_obligation_ids: tuple[str, ...] = Field(default_factory=tuple)
    contradicted_obligation_ids: tuple[str, ...] = Field(default_factory=tuple)
    duration_ms: int = Field(default=0, ge=0)
    time_budget_ms: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status is SymbolicStatus.SUCCESS

    @property
    def contradiction(self) -> bool:
        return self.status is SymbolicStatus.CONTRADICTION

    def to_branch_controller_hook_result(self):
        from src.branches.branch_controller import SymbolicHookResult

        return SymbolicHookResult(
            passed=self.passed,
            score=self.score,
            summary=self.summary,
            exact_match=self.exact_match,
            added_constraints=self.added_constraints,
            added_invariants=self.added_invariants,
            metadata={
                "symbolic_result": self.model_dump(mode="json"),
                "symbolic_status": self.status.value,
                "validator_kind": self.validator_kind.value,
                "proof_obligations": [
                    {
                        "obligation_id": item.obligation_id,
                        "status": item.status.value,
                        "claim": item.claim,
                        "evidence_kind_required": item.evidence_kind_required.value,
                    }
                    for item in self.proof_obligations
                ],
                "discharged_obligation_ids": list(self.discharged_obligation_ids),
                "contradicted_obligation_ids": list(self.contradicted_obligation_ids),
            },
        )

    def to_state_evidence_record(self):
        try:
            from src.state_graph.node import EvidenceRecord
        except Exception:
            return None
        return EvidenceRecord.create(
            kind="symbolic_check",
            source=self.validator_name,
            summary=self.summary,
            score=self.score,
            supports=list(self.added_constraints + self.added_invariants),
            refutes=["contradiction"] if self.contradiction else [],
            payload={
                "request_id": self.request_id,
                "status": self.status.value,
                "validator_kind": self.validator_kind.value,
                "diagnostics": [item.model_dump(mode="json") for item in self.diagnostics],
                "discharged_obligation_ids": list(self.discharged_obligation_ids),
                "contradicted_obligation_ids": list(self.contradicted_obligation_ids),
            },
        )


def _stable_hash(prefix: str, payload: Mapping[str, Any]) -> str:
    raw = f"{prefix}::{json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False)}"
    return f"{prefix}_{sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def _status_from_composite(
    status: validators.CompositeValidationStatus,
) -> SymbolicStatus:
    return SymbolicStatus(status.value)


def _validator_kind_from_domains(
    domains: tuple[validators.ValidatorDomain, ...],
) -> SymbolicValidatorKind:
    if not domains or len(domains) > 1:
        return SymbolicValidatorKind.AUTO
    domain = domains[0]
    return {
        validators.ValidatorDomain.ALGEBRA: SymbolicValidatorKind.ALGEBRA,
        validators.ValidatorDomain.NUMBER_THEORY: SymbolicValidatorKind.NUMBER_THEORY,
        validators.ValidatorDomain.GEOMETRY: SymbolicValidatorKind.GEOMETRY,
    }[domain]


def _diag_code_from_status(status: SymbolicStatus) -> SymbolicDiagnosticCode:
    return {
        SymbolicStatus.SUCCESS: SymbolicDiagnosticCode.SUCCESS,
        SymbolicStatus.CONTRADICTION: SymbolicDiagnosticCode.CONTRADICTION,
        SymbolicStatus.UNSUPPORTED: SymbolicDiagnosticCode.UNSUPPORTED,
        SymbolicStatus.TIMEOUT: SymbolicDiagnosticCode.TIMEOUT,
        SymbolicStatus.MALFORMED_INPUT: SymbolicDiagnosticCode.MALFORMED_INPUT,
        SymbolicStatus.EXECUTION_FAILURE: SymbolicDiagnosticCode.EXECUTION_ERROR,
    }[status]


def _constraint_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    summary = getattr(value, "summary", None)
    if callable(summary):
        try:
            rendered = summary()
        except Exception:
            rendered = None
        if isinstance(rendered, str) and rendered.strip():
            return rendered
    for attr in ("normalized_text", "normalized_expression", "normalized_lhs", "lhs", "summary_text"):
        rendered = getattr(value, attr, None)
        if isinstance(rendered, str) and rendered.strip():
            relation = getattr(value, "normalized_relation", None) or getattr(value, "relation", None)
            rhs = getattr(value, "normalized_rhs", None) or getattr(value, "rhs", None)
            if attr in {"normalized_lhs", "lhs"} and isinstance(relation, str) and relation.strip():
                parts = [rendered.strip(), relation.strip()]
                if isinstance(rhs, str) and rhs.strip():
                    parts.append(rhs.strip())
                return " ".join(parts)
            return rendered
    return str(value)


def _normalize_domain_hint(value: Any) -> str | None:
    raw = getattr(value, "value", value)
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


class SymbolicRunner:
    def __init__(self, *, config: SymbolicRunnerConfig | None = None) -> None:
        self.config = config or SymbolicRunnerConfig()

    def _build_validation_request(
        self,
        request: SymbolicRunRequest,
    ) -> validators.ValidationRequest:
        return validators.ValidationRequest(
            target=validators.ValidationTarget(request.target.value),
            statement=request.statement,
            candidate_answer=request.candidate_answer,
            constraints=tuple(request.constraints),
            relation_bundle=tuple(request.relation_bundle),
            answer_symbol=request.answer_symbol,
            domain_hint=request.domain_hint,
            validator_hint=(
                None
                if request.validator_hint is SymbolicValidatorKind.AUTO
                else request.validator_hint.value
            ),
            supporting_step_ids=((request.step_id,) if request.step_id else ()),
            proof_obligations=tuple(request.proof_obligations),
            metadata=dict(request.metadata),
        )

    def _run_with_timeout(self, fn, timeout_ms: int):
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="symbolic_runner") as executor:
            future = executor.submit(fn)
            return future.result(timeout=timeout_ms / 1000.0)

    def validate(self, request: SymbolicRunRequest) -> SymbolicRunResult:
        started = time.monotonic()
        provenance = SymbolicProvenance(
            problem_id=request.problem_id,
            branch_id=request.branch_id,
            step_id=request.step_id,
            node_id=request.node_id,
            operator_name=request.operator_name,
            request_id=request.request_id,
            metadata=dict(request.metadata),
        )
        timeout_ms = request.time_budget_ms or self.config.default_timeout_ms

        try:
            composite = self._run_with_timeout(
                lambda: validators.validate_symbolic_request(
                    self._build_validation_request(request)
                ),
                timeout_ms,
            )
        except FutureTimeoutError:
            duration_ms = int((time.monotonic() - started) * 1000)
            return SymbolicRunResult(
                request_id=request.request_id,
                target=request.target,
                status=SymbolicStatus.TIMEOUT,
                validator_kind=SymbolicValidatorKind.AUTO,
                validator_name="validators.timeout",
                summary="Symbolic validation timed out.",
                score=0.0,
                exact_match=False,
                diagnostics=(
                    SymbolicDiagnostic(
                        code=SymbolicDiagnosticCode.TIMEOUT,
                        message="Symbolic validation exceeded time budget.",
                        validator_kind=SymbolicValidatorKind.AUTO,
                        payload={"time_budget_ms": timeout_ms},
                    ),
                ),
                evidence=(),
                provenance=provenance,
                proof_obligations=tuple(request.proof_obligations),
                duration_ms=duration_ms,
                time_budget_ms=timeout_ms,
                metadata={"timeout": True},
            )
        except Exception as exc:
            duration_ms = int((time.monotonic() - started) * 1000)
            return SymbolicRunResult(
                request_id=request.request_id,
                target=request.target,
                status=SymbolicStatus.EXECUTION_FAILURE,
                validator_kind=SymbolicValidatorKind.AUTO,
                validator_name="validators.exception",
                summary=f"Symbolic validation failed before completion: {exc}",
                score=0.0,
                exact_match=False,
                diagnostics=(
                    SymbolicDiagnostic(
                        code=SymbolicDiagnosticCode.EXECUTION_ERROR,
                        message=str(exc),
                        validator_kind=SymbolicValidatorKind.AUTO,
                    ),
                ),
                evidence=(),
                provenance=provenance,
                proof_obligations=tuple(request.proof_obligations),
                duration_ms=duration_ms,
                time_budget_ms=timeout_ms,
                metadata={"exception": str(exc)},
            )

        duration_ms = int((time.monotonic() - started) * 1000)
        validator_kind = _validator_kind_from_domains(composite.applied_domains)

        evidence = tuple(
            SymbolicEvidence(
                evidence_id=f"symev_{i}_{request.request_id}",
                status=SymbolicStatus(record.status.value),
                validator_kind={
                    validators.ValidatorDomain.ALGEBRA: SymbolicValidatorKind.ALGEBRA,
                    validators.ValidatorDomain.NUMBER_THEORY: SymbolicValidatorKind.NUMBER_THEORY,
                    validators.ValidatorDomain.GEOMETRY: SymbolicValidatorKind.GEOMETRY,
                }[record.provenance.domain],
                validator_name=record.provenance.function_name,
                source_module=record.provenance.module_name,
                check_name=record.provenance.target.value,
                summary=record.summary,
                score=record.score,
                exact_match=record.exact,
                payload={
                    "diagnostics": [d.message for d in record.diagnostics],
                    "evidence_payloads": list(record.evidence_payloads),
                    "derived_facts": list(record.derived_facts),
                },
            )
            for i, record in enumerate(composite.records)
        )

        diagnostics: list[SymbolicDiagnostic] = []
        for diag in composite.diagnostics:
            if diag.code == "no_validator":
                code = SymbolicDiagnosticCode.REQUEST_ERROR
            elif diag.code in SymbolicDiagnosticCode._value2member_map_:
                code = SymbolicDiagnosticCode(diag.code)
            else:
                code = _diag_code_from_status(_status_from_composite(composite.status))
            diagnostics.append(
                SymbolicDiagnostic(
                    code=code,
                    message=diag.message,
                    validator_kind=validator_kind,
                    payload=dict(diag.payload),
                )
            )
        if not diagnostics:
            diagnostics = [
                SymbolicDiagnostic(
                    code=_diag_code_from_status(_status_from_composite(composite.status)),
                    message=composite.summary,
                    validator_kind=validator_kind,
                    payload={"applied_domains": [d.value for d in composite.applied_domains]},
                )
            ]

        return SymbolicRunResult(
            request_id=request.request_id,
            target=request.target,
            status=_status_from_composite(composite.status),
            validator_kind=validator_kind,
            validator_name="src.symbolic.validators",
            summary=composite.summary,
            score=composite.score,
            exact_match=composite.exact,
            diagnostics=tuple(diagnostics),
            evidence=evidence,
            provenance=provenance,
            added_constraints=composite.derived_constraints,
            added_invariants=composite.derived_invariants,
            proof_obligations=composite.proof_obligations,
            discharged_obligation_ids=composite.discharged_obligation_ids,
            contradicted_obligation_ids=composite.contradicted_obligation_ids,
            duration_ms=duration_ms,
            time_budget_ms=timeout_ms,
            metadata={
                "applied_domains": [d.value for d in composite.applied_domains],
                "contradiction_found": composite.contradiction_found,
                "failure_type": composite.failure_type.value if composite.failure_type else None,
                "discharged_obligation_ids": list(composite.discharged_obligation_ids),
                "contradicted_obligation_ids": list(composite.contradicted_obligation_ids),
                **dict(composite.metadata),
            },
        )

    def validate_branch_step(
        self,
        *,
        statement: str,
        constraints: tuple[str, ...] | list[str] = (),
        proof_obligations: tuple[ProofObligation, ...] = (),
        problem_id: str | None = None,
        branch_id: str | None = None,
        step_id: str | None = None,
        node_id: str | None = None,
        operator_name: str | None = None,
        domain_hint: str | None = None,
        validator_hint: SymbolicValidatorKind = SymbolicValidatorKind.AUTO,
        time_budget_ms: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SymbolicRunResult:
        return self.validate(
            SymbolicRunRequest(
                target=SymbolicTarget.BRANCH_STEP,
                statement=statement,
                constraints=tuple(constraints),
                problem_id=problem_id,
                branch_id=branch_id,
                step_id=step_id,
                node_id=node_id,
                operator_name=operator_name,
                domain_hint=domain_hint,
                validator_hint=validator_hint,
                proof_obligations=tuple(proof_obligations),
                time_budget_ms=time_budget_ms,
                metadata=dict(metadata or {}),
            )
        )

    def validate_candidate_answer(
        self,
        *,
        candidate_answer: str,
        constraints: tuple[str, ...] | list[str] = (),
        answer_symbol: str | None = None,
        problem_id: str | None = None,
        branch_id: str | None = None,
        step_id: str | None = None,
        node_id: str | None = None,
        domain_hint: str | None = None,
        validator_hint: SymbolicValidatorKind = SymbolicValidatorKind.AUTO,
        time_budget_ms: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SymbolicRunResult:
        return self.validate(
            SymbolicRunRequest(
                target=SymbolicTarget.CANDIDATE_ANSWER,
                candidate_answer=candidate_answer,
                constraints=tuple(constraints),
                answer_symbol=answer_symbol,
                problem_id=problem_id,
                branch_id=branch_id,
                step_id=step_id,
                node_id=node_id,
                domain_hint=domain_hint,
                validator_hint=validator_hint,
                time_budget_ms=time_budget_ms,
                metadata=dict(metadata or {}),
            )
        )

    def build_branch_controller_hook(self):
        def _hook(*, problem, route, branch, node, operator_name, generation):
            statement = getattr(generation, "reasoning", "") or getattr(generation, "summary", "")
            constraint_records = (
                getattr(node, "constraints", ())
                or getattr(node, "remaining_constraints", ())
                or ()
            )
            constraints = tuple(
                rendered
                for rendered in (_constraint_to_text(item) for item in constraint_records)
                if rendered
            )
            result = self.validate_branch_step(
                statement=statement,
                constraints=constraints,
                proof_obligations=tuple(getattr(node, "proof_obligations", ()) or ()),
                problem_id=getattr(problem, "problem_id", None),
                branch_id=getattr(branch, "branch_id", None),
                step_id=getattr(generation, "step_id", None),
                node_id=getattr(node, "node_id", None),
                operator_name=operator_name,
                domain_hint=_normalize_domain_hint(getattr(problem, "domain", None)),
                metadata={"generation": getattr(generation, "reasoning", None)},
            )
            return result.to_branch_controller_hook_result()

        return _hook


_DEFAULT_RUNNER: SymbolicRunner | None = None


def get_default_symbolic_runner() -> SymbolicRunner:
    global _DEFAULT_RUNNER
    if _DEFAULT_RUNNER is None:
        _DEFAULT_RUNNER = SymbolicRunner()
    return _DEFAULT_RUNNER


def validate_branch_step(**kwargs: Any) -> SymbolicRunResult:
    return get_default_symbolic_runner().validate_branch_step(**kwargs)


def validate_candidate_answer(**kwargs: Any) -> SymbolicRunResult:
    return get_default_symbolic_runner().validate_candidate_answer(**kwargs)


def build_branch_controller_symbolic_hook(*, runner: SymbolicRunner | None = None):
    return (runner or get_default_symbolic_runner()).build_branch_controller_hook()


__all__ = [
    "SymbolicStatus",
    "SymbolicTarget",
    "SymbolicValidatorKind",
    "SymbolicDiagnosticCode",
    "SymbolicProvenance",
    "SymbolicDiagnostic",
    "SymbolicEvidence",
    "SymbolicRunnerConfig",
    "SymbolicRunRequest",
    "SymbolicRunResult",
    "SymbolicRunner",
    "get_default_symbolic_runner",
    "validate_branch_step",
    "validate_candidate_answer",
    "build_branch_controller_symbolic_hook",
]
