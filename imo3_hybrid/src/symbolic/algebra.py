from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha1
import re
from typing import Any, Mapping, Sequence

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from src.branches.branch_state import SymbolicEvidence
from src.common.schemas import FailureType


_MAX_EXPR_CHARS = 256
_ALGEBRA_TRANSFORMS = standard_transformations + (
    convert_xor,
    implicit_multiplication_application,
)
_REL_RE = re.compile(r"(<=|>=|!=|=|<|>)")
_ASSIGN_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$")


class AlgebraCheckStatus(str, Enum):
    SUCCESS = "success"
    CONTRADICTION = "contradiction"
    UNSUPPORTED = "unsupported"
    TIMEOUT = "timeout"
    MALFORMED_INPUT = "malformed_input"
    EXECUTION_FAILURE = "execution_failure"


class AlgebraCheckKind(str, Enum):
    EQUALITY = "equality"
    EQUIVALENCE = "equivalence"
    INEQUALITY = "inequality"
    SUBSTITUTION = "substitution"
    CONSTRAINT_SLICE = "constraint_slice"
    CANDIDATE_ANSWER = "candidate_answer"
    CONTRADICTION_SCAN = "contradiction_scan"
    STATEMENT = "statement"


class AlgebraDiagnosticCode(str, Enum):
    PARSE_ERROR = "parse_error"
    UNSUPPORTED_FORM = "unsupported_form"
    NONALGEBRAIC_FORM = "nonalgebraic_form"
    CONTRADICTION = "contradiction"
    INSUFFICIENT_ASSIGNMENT = "insufficient_assignment"
    EXECUTION_ERROR = "execution_error"
    SUCCESS = "success"


@dataclass(frozen=True)
class AlgebraDiagnostic:
    code: AlgebraDiagnosticCode
    message: str
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AlgebraEvidence:
    label: str
    expression: str
    simplified_expression: str
    exact_value: str | None
    note: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AlgebraCheckResult:
    status: AlgebraCheckStatus
    kind: AlgebraCheckKind
    summary: str
    score: float
    exact: bool
    diagnostics: tuple[AlgebraDiagnostic, ...] = field(default_factory=tuple)
    evidence: tuple[AlgebraEvidence, ...] = field(default_factory=tuple)
    derived_facts: tuple[str, ...] = field(default_factory=tuple)
    contradiction_found: bool = False
    failure_type: FailureType | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status is AlgebraCheckStatus.SUCCESS

    def to_branch_symbolic_evidence(
        self,
        *,
        check_name: str | None = None,
        supporting_step_ids: Sequence[str] = (),
    ) -> SymbolicEvidence:
        return SymbolicEvidence(
            evidence_id=_stable_id(self.kind.value, self.summary, self.metadata),
            passed=self.passed,
            score=float(self.score),
            check_name=check_name or self.kind.value,
            summary=self.summary,
            supporting_step_ids=tuple(supporting_step_ids),
        )


def _stable_id(prefix: str, summary: str, metadata: Mapping[str, Any]) -> str:
    raw = f"{prefix}|{summary}|{sorted((metadata or {}).items())}"
    return f"{prefix}_{sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def _score(status: AlgebraCheckStatus, *, exact: bool) -> float:
    if status is AlgebraCheckStatus.SUCCESS:
        return 1.0 if exact else 0.8
    if status is AlgebraCheckStatus.CONTRADICTION:
        return 0.0
    if status is AlgebraCheckStatus.UNSUPPORTED:
        return 0.15
    if status is AlgebraCheckStatus.MALFORMED_INPUT:
        return 0.05
    return 0.0


def _result(
    *,
    status: AlgebraCheckStatus,
    kind: AlgebraCheckKind,
    summary: str,
    exact: bool,
    diagnostics: Sequence[AlgebraDiagnostic] = (),
    evidence: Sequence[AlgebraEvidence] = (),
    derived_facts: Sequence[str] = (),
    contradiction_found: bool = False,
    failure_type: FailureType | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> AlgebraCheckResult:
    return AlgebraCheckResult(
        status=status,
        kind=kind,
        summary=summary,
        score=_score(status, exact=exact),
        exact=exact,
        diagnostics=tuple(diagnostics),
        evidence=tuple(evidence),
        derived_facts=tuple(derived_facts),
        contradiction_found=contradiction_found,
        failure_type=failure_type,
        metadata=dict(metadata or {}),
    )


def _parse_expr_safe(text: str | int | float) -> sp.Expr:
    normalized = str(text).strip()
    if not normalized:
        raise ValueError("empty expression")
    if len(normalized) > _MAX_EXPR_CHARS:
        raise ValueError("expression too long for bounded algebra validation")
    return parse_expr(normalized, transformations=_ALGEBRA_TRANSFORMS, evaluate=True)


def _parse_relation(statement: str) -> tuple[str | None, str, str | None]:
    text = (statement or "").strip()
    if not text:
        raise ValueError("empty statement")
    match = _REL_RE.search(text)
    if not match:
        return None, text, None
    op = match.group(1)
    lhs = text[: match.start()].strip()
    rhs = text[match.end() :].strip()
    if not lhs or not rhs:
        raise ValueError(f"malformed relation: {statement}")
    return lhs, op, rhs


def _is_exact_constant(expr: sp.Expr) -> bool:
    return not expr.free_symbols and expr.is_number is True


def _extract_exact_assignments(constraints: Sequence[str]) -> dict[str, sp.Expr]:
    assignments: dict[str, sp.Expr] = {}
    for text in constraints:
        match = _ASSIGN_RE.match(text or "")
        if not match:
            continue
        lhs = match.group(1)
        rhs_text = match.group(2)
        rhs = _parse_expr_safe(rhs_text)
        if rhs.free_symbols:
            continue
        rhs = sp.simplify(rhs)
        previous = assignments.get(lhs)
        if previous is not None and sp.simplify(previous - rhs) != 0:
            raise ValueError(f"Contradictory assignments for {lhs}: {previous} vs {rhs}")
        assignments[lhs] = rhs
    return assignments


def _sub_map(assignments: Mapping[str, sp.Expr]) -> dict[sp.Symbol, sp.Expr]:
    return {sp.Symbol(name): value for name, value in assignments.items()}


def _simplify(expr: sp.Expr) -> sp.Expr:
    try:
        return sp.cancel(sp.factor(sp.together(sp.expand(expr))))
    except Exception:
        try:
            return sp.simplify(expr)
        except Exception:
            return expr


def _numeric_truth(expr: sp.Expr) -> bool | None:
    simplified = _simplify(expr)
    if simplified.free_symbols:
        return None
    if simplified in {sp.S.true, sp.S.false}:
        return bool(simplified)
    if getattr(simplified, "is_zero", None) is True:
        return True
    if getattr(simplified, "is_zero", None) is False:
        return False
    return None


def _difference_payload(lhs: sp.Expr, rhs: sp.Expr) -> tuple[sp.Expr, dict[str, Any]]:
    diff = _simplify(lhs - rhs)
    payload = {
        "difference": str(diff),
        "lhs_simplified": str(_simplify(lhs)),
        "rhs_simplified": str(_simplify(rhs)),
    }
    return diff, payload


def _truth_from_relation(
    lhs: sp.Expr,
    op: str,
    rhs: sp.Expr,
) -> tuple[AlgebraCheckStatus, str, bool, Mapping[str, Any]]:
    diff, payload = _difference_payload(lhs, rhs)

    if op == "=":
        if diff == 0:
            return AlgebraCheckStatus.SUCCESS, "Equality holds exactly.", True, payload
        if not diff.free_symbols:
            return (
                AlgebraCheckStatus.CONTRADICTION,
                "Equality is false under exact evaluation.",
                True,
                payload,
            )
        return (
            AlgebraCheckStatus.UNSUPPORTED,
            "Equality could not be resolved exactly.",
            False,
            payload,
        )

    if op == "!=":
        if diff == 0:
            return (
                AlgebraCheckStatus.CONTRADICTION,
                "Inequality x != y is false because both sides are equal.",
                True,
                payload,
            )
        if not diff.free_symbols:
            return AlgebraCheckStatus.SUCCESS, "Inequality holds exactly.", True, payload
        return (
            AlgebraCheckStatus.UNSUPPORTED,
            "Inequality could not be resolved exactly.",
            False,
            payload,
        )

    rel_expr = None
    if op == "<":
        rel_expr = lhs < rhs
    elif op == "<=":
        rel_expr = lhs <= rhs
    elif op == ">":
        rel_expr = lhs > rhs
    elif op == ">=":
        rel_expr = lhs >= rhs
    else:
        return (
            AlgebraCheckStatus.UNSUPPORTED,
            f"Unsupported algebraic relation operator: {op}",
            False,
            payload,
        )

    truth = _numeric_truth(rel_expr)
    if truth is True:
        return (
            AlgebraCheckStatus.SUCCESS,
            f"Inequality {lhs} {op} {rhs} holds under exact evaluation.",
            True,
            payload,
        )
    if truth is False:
        return (
            AlgebraCheckStatus.CONTRADICTION,
            f"Inequality {lhs} {op} {rhs} fails under exact evaluation.",
            True,
            payload,
        )
    return (
        AlgebraCheckStatus.UNSUPPORTED,
        "Inequality could not be resolved exactly.",
        False,
        payload,
    )


def _record_result(
    *,
    kind: AlgebraCheckKind,
    status: AlgebraCheckStatus,
    summary: str,
    exact: bool,
    expression: str,
    simplified_expression: str,
    exact_value: str | None,
    diagnostics: Sequence[AlgebraDiagnostic],
    derived_facts: Sequence[str] = (),
    contradiction_found: bool = False,
    failure_type: FailureType | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> AlgebraCheckResult:
    return _result(
        status=status,
        kind=kind,
        summary=summary,
        exact=exact,
        diagnostics=diagnostics,
        evidence=(
            AlgebraEvidence(
                label=kind.value,
                expression=expression,
                simplified_expression=simplified_expression,
                exact_value=exact_value,
                payload=dict(metadata or {}),
            ),
        ),
        derived_facts=derived_facts,
        contradiction_found=contradiction_found,
        failure_type=failure_type,
        metadata=metadata,
    )


def check_equality(
    lhs: str | int | float,
    rhs: str | int | float,
    constraints: Sequence[str] = (),
) -> AlgebraCheckResult:
    try:
        assignments = _extract_exact_assignments(constraints)
        left = _parse_expr_safe(lhs).subs(_sub_map(assignments))
        right = _parse_expr_safe(rhs).subs(_sub_map(assignments))
        status, summary, exact, payload = _truth_from_relation(left, "=", right)
        code = (
            AlgebraDiagnosticCode.SUCCESS
            if status is AlgebraCheckStatus.SUCCESS
            else AlgebraDiagnosticCode.CONTRADICTION
            if status is AlgebraCheckStatus.CONTRADICTION
            else AlgebraDiagnosticCode.UNSUPPORTED_FORM
        )
        return _record_result(
            kind=AlgebraCheckKind.EQUALITY,
            status=status,
            summary=summary,
            exact=exact,
            expression=f"{lhs} = {rhs}",
            simplified_expression=str(payload.get("difference", "")),
            exact_value=payload.get("difference"),
            diagnostics=(AlgebraDiagnostic(code, summary, payload),),
            derived_facts=((f"{lhs} = {rhs}",) if status is AlgebraCheckStatus.SUCCESS else ()),
            contradiction_found=status is AlgebraCheckStatus.CONTRADICTION,
            failure_type=FailureType.SYMBOLIC_MISMATCH if status is AlgebraCheckStatus.CONTRADICTION else None,
            metadata={"assignments": {k: str(v) for k, v in assignments.items()}, **payload},
        )
    except ValueError as exc:
        return _result(
            status=AlgebraCheckStatus.MALFORMED_INPUT,
            kind=AlgebraCheckKind.EQUALITY,
            summary=str(exc),
            exact=False,
            diagnostics=(AlgebraDiagnostic(AlgebraDiagnosticCode.PARSE_ERROR, str(exc)),),
            failure_type=FailureType.SYMBOLIC_MISMATCH,
        )
    except Exception as exc:
        return _result(
            status=AlgebraCheckStatus.EXECUTION_FAILURE,
            kind=AlgebraCheckKind.EQUALITY,
            summary=f"Algebra equality check failed: {exc}",
            exact=False,
            diagnostics=(AlgebraDiagnostic(AlgebraDiagnosticCode.EXECUTION_ERROR, str(exc)),),
            failure_type=FailureType.SYMBOLIC_MISMATCH,
        )


def check_equivalence(
    expr_a: str | int | float,
    expr_b: str | int | float,
    constraints: Sequence[str] = (),
) -> AlgebraCheckResult:
    result = check_equality(expr_a, expr_b, constraints)
    return AlgebraCheckResult(
        status=result.status,
        kind=AlgebraCheckKind.EQUIVALENCE,
        summary=result.summary,
        score=result.score,
        exact=result.exact,
        diagnostics=result.diagnostics,
        evidence=result.evidence,
        derived_facts=result.derived_facts,
        contradiction_found=result.contradiction_found,
        failure_type=result.failure_type,
        metadata=result.metadata,
    )


def check_substitution(
    symbol: str,
    value: str | int | float,
    constraints: Sequence[str],
) -> AlgebraCheckResult:
    result = validate_candidate_answer(
        candidate_answer=str(value),
        constraints=constraints,
        answer_symbol=symbol,
    )
    return AlgebraCheckResult(
        status=result.status,
        kind=AlgebraCheckKind.SUBSTITUTION,
        summary=result.summary,
        score=result.score,
        exact=result.exact,
        diagnostics=result.diagnostics,
        evidence=result.evidence,
        derived_facts=result.derived_facts,
        contradiction_found=result.contradiction_found,
        failure_type=result.failure_type,
        metadata=result.metadata,
    )


def detect_algebraic_contradictions(constraints: Sequence[str]) -> AlgebraCheckResult:
    diagnostics: list[AlgebraDiagnostic] = []
    evidence: list[AlgebraEvidence] = []
    try:
        assignments = _extract_exact_assignments(constraints)
    except ValueError as exc:
        return _result(
            status=AlgebraCheckStatus.CONTRADICTION,
            kind=AlgebraCheckKind.CONTRADICTION_SCAN,
            summary=str(exc),
            exact=True,
            diagnostics=(AlgebraDiagnostic(AlgebraDiagnosticCode.CONTRADICTION, str(exc)),),
            contradiction_found=True,
            failure_type=FailureType.SYMBOLIC_MISMATCH,
        )

    substitution_map = _sub_map(assignments)
    for text in constraints:
        try:
            lhs, op, rhs = _parse_relation(text)
        except ValueError:
            continue
        if lhs is None or rhs is None:
            continue
        try:
            left = _parse_expr_safe(lhs).subs(substitution_map)
            right = _parse_expr_safe(rhs).subs(substitution_map)
            status, summary, _, payload = _truth_from_relation(left, op, right)
        except Exception:
            continue
        evidence.append(
            AlgebraEvidence(
                label="constraint",
                expression=text,
                simplified_expression=str(payload.get("difference", "")),
                exact_value=payload.get("difference"),
                payload=payload,
            )
        )
        if status is AlgebraCheckStatus.CONTRADICTION:
            diagnostics.append(
                AlgebraDiagnostic(
                    AlgebraDiagnosticCode.CONTRADICTION,
                    summary,
                    {"constraint": text, **payload},
                )
            )

    if diagnostics:
        return _result(
            status=AlgebraCheckStatus.CONTRADICTION,
            kind=AlgebraCheckKind.CONTRADICTION_SCAN,
            summary=diagnostics[0].message,
            exact=True,
            diagnostics=tuple(diagnostics),
            evidence=tuple(evidence),
            contradiction_found=True,
            failure_type=FailureType.SYMBOLIC_MISMATCH,
            metadata={"assignments": {k: str(v) for k, v in assignments.items()}},
        )

    return _result(
        status=AlgebraCheckStatus.SUCCESS,
        kind=AlgebraCheckKind.CONTRADICTION_SCAN,
        summary="No exact algebraic contradiction detected in supported constraints.",
        exact=True,
        diagnostics=(AlgebraDiagnostic(AlgebraDiagnosticCode.SUCCESS, "No contradiction detected."),),
        evidence=tuple(evidence),
        metadata={"assignments": {k: str(v) for k, v in assignments.items()}},
    )


def validate_constraint_slice(
    constraints: Sequence[str],
    candidate_answer: str | None = None,
    answer_symbol: str | None = None,
) -> AlgebraCheckResult:
    contradiction = detect_algebraic_contradictions(constraints)
    if contradiction.status in {
        AlgebraCheckStatus.CONTRADICTION,
        AlgebraCheckStatus.EXECUTION_FAILURE,
        AlgebraCheckStatus.MALFORMED_INPUT,
    }:
        return contradiction

    if candidate_answer is not None and answer_symbol:
        answer_result = validate_candidate_answer(candidate_answer, constraints, answer_symbol)
        if answer_result.status in {
            AlgebraCheckStatus.CONTRADICTION,
            AlgebraCheckStatus.EXECUTION_FAILURE,
            AlgebraCheckStatus.MALFORMED_INPUT,
        }:
            return answer_result

    supported = 0
    exact_success = 0
    diagnostics: list[AlgebraDiagnostic] = list(contradiction.diagnostics)
    evidence: list[AlgebraEvidence] = list(contradiction.evidence)
    assignments = {}
    try:
        assignments = _extract_exact_assignments(constraints)
    except Exception:
        assignments = {}
    substitution_map = _sub_map(assignments)

    for text in constraints:
        try:
            lhs, op, rhs = _parse_relation(text)
        except ValueError as exc:
            diagnostics.append(
                AlgebraDiagnostic(
                    AlgebraDiagnosticCode.PARSE_ERROR,
                    str(exc),
                    {"constraint": text},
                )
            )
            continue
        if lhs is None or rhs is None:
            continue
        try:
            left = _parse_expr_safe(lhs).subs(substitution_map)
            right = _parse_expr_safe(rhs).subs(substitution_map)
            status, summary, exact, payload = _truth_from_relation(left, op, right)
        except Exception as exc:
            diagnostics.append(
                AlgebraDiagnostic(
                    AlgebraDiagnosticCode.EXECUTION_ERROR,
                    str(exc),
                    {"constraint": text},
                )
            )
            continue

        if status is not AlgebraCheckStatus.UNSUPPORTED:
            supported += 1
        if status is AlgebraCheckStatus.SUCCESS and exact:
            exact_success += 1

        diagnostics.append(
            AlgebraDiagnostic(
                AlgebraDiagnosticCode.SUCCESS
                if status is AlgebraCheckStatus.SUCCESS
                else AlgebraDiagnosticCode.CONTRADICTION
                if status is AlgebraCheckStatus.CONTRADICTION
                else AlgebraDiagnosticCode.UNSUPPORTED_FORM,
                summary,
                {"constraint": text, **payload},
            )
        )
        evidence.append(
            AlgebraEvidence(
                label="constraint",
                expression=text,
                simplified_expression=str(payload.get("difference", "")),
                exact_value=payload.get("difference"),
                payload=payload,
            )
        )

    if supported == 0:
        return _result(
            status=AlgebraCheckStatus.UNSUPPORTED,
            kind=AlgebraCheckKind.CONSTRAINT_SLICE,
            summary="Constraint slice contains no algebraically checkable supported relations.",
            exact=False,
            diagnostics=tuple(diagnostics),
            evidence=tuple(evidence),
            failure_type=FailureType.COVERAGE_GAP,
        )

    status = AlgebraCheckStatus.SUCCESS if exact_success == supported else AlgebraCheckStatus.UNSUPPORTED
    summary = (
        f"Validated {exact_success}/{supported} supported algebraic relations exactly."
        if status is AlgebraCheckStatus.SUCCESS
        else f"Partial algebraic support only: {exact_success}/{supported} supported relations were exact."
    )
    return _result(
        status=status,
        kind=AlgebraCheckKind.CONSTRAINT_SLICE,
        summary=summary,
        exact=(status is AlgebraCheckStatus.SUCCESS),
        diagnostics=tuple(diagnostics),
        evidence=tuple(evidence),
        failure_type=None if status is AlgebraCheckStatus.SUCCESS else FailureType.COVERAGE_GAP,
        metadata={
            "supported_relations": supported,
            "exact_success": exact_success,
            "partial_support": exact_success > 0 and exact_success < supported,
        },
    )


def validate_candidate_answer(
    candidate_answer: str,
    constraints: Sequence[str],
    answer_symbol: str | None,
) -> AlgebraCheckResult:
    if not answer_symbol:
        return _result(
            status=AlgebraCheckStatus.UNSUPPORTED,
            kind=AlgebraCheckKind.CANDIDATE_ANSWER,
            summary="Candidate-answer algebra validation requires an answer symbol.",
            exact=False,
            diagnostics=(
                AlgebraDiagnostic(
                    AlgebraDiagnosticCode.INSUFFICIENT_ASSIGNMENT,
                    "Missing answer symbol.",
                ),
            ),
            failure_type=FailureType.COVERAGE_GAP,
        )

    try:
        value_expr = _parse_expr_safe(candidate_answer)
        if value_expr.free_symbols:
            return _result(
                status=AlgebraCheckStatus.UNSUPPORTED,
                kind=AlgebraCheckKind.CANDIDATE_ANSWER,
                summary="Candidate answer is not a concrete algebraic value after parsing.",
                exact=False,
                diagnostics=(
                    AlgebraDiagnostic(
                        AlgebraDiagnosticCode.INSUFFICIENT_ASSIGNMENT,
                        "Candidate answer remains symbolic.",
                    ),
                ),
                failure_type=FailureType.COVERAGE_GAP,
            )
        augmented = list(constraints) + [f"{answer_symbol} = ({candidate_answer})"]
        result = validate_constraint_slice(augmented)
        metadata = dict(result.metadata)
        metadata["assignment"] = f"{answer_symbol} = ({candidate_answer})"
        return _result(
            status=result.status,
            kind=AlgebraCheckKind.CANDIDATE_ANSWER,
            summary=(
                result.summary
                if result.status is not AlgebraCheckStatus.SUCCESS
                else f"Candidate answer satisfies supported algebraic constraints for {answer_symbol}."
            ),
            exact=result.exact,
            diagnostics=result.diagnostics,
            evidence=result.evidence,
            derived_facts=result.derived_facts
            + ((f"{answer_symbol} = ({candidate_answer})",) if result.status is AlgebraCheckStatus.SUCCESS else ()),
            contradiction_found=result.contradiction_found,
            failure_type=result.failure_type,
            metadata=metadata,
        )
    except ValueError as exc:
        return _result(
            status=AlgebraCheckStatus.MALFORMED_INPUT,
            kind=AlgebraCheckKind.CANDIDATE_ANSWER,
            summary=str(exc),
            exact=False,
            diagnostics=(AlgebraDiagnostic(AlgebraDiagnosticCode.PARSE_ERROR, str(exc)),),
            failure_type=FailureType.SYMBOLIC_MISMATCH,
        )
    except Exception as exc:
        return _result(
            status=AlgebraCheckStatus.EXECUTION_FAILURE,
            kind=AlgebraCheckKind.CANDIDATE_ANSWER,
            summary=f"Candidate-answer algebra validation failed: {exc}",
            exact=False,
            diagnostics=(AlgebraDiagnostic(AlgebraDiagnosticCode.EXECUTION_ERROR, str(exc)),),
            failure_type=FailureType.SYMBOLIC_MISMATCH,
        )


def validate_algebra_statement(
    statement: str,
    constraints: Sequence[str] = (),
) -> AlgebraCheckResult:
    try:
        lhs, op, rhs = _parse_relation(statement)
        if lhs is None or rhs is None:
            expr = _parse_expr_safe(op)
            if expr.free_symbols:
                return _result(
                    status=AlgebraCheckStatus.UNSUPPORTED,
                    kind=AlgebraCheckKind.STATEMENT,
                    summary="Free symbolic expression without a relation is outside supported algebra validation scope.",
                    exact=False,
                    diagnostics=(
                        AlgebraDiagnostic(
                            AlgebraDiagnosticCode.UNSUPPORTED_FORM,
                            "Expression-only algebra validation requires a closed expression.",
                        ),
                    ),
                    failure_type=FailureType.COVERAGE_GAP,
                )
            exact_val = str(_simplify(expr))
            return _result(
                status=AlgebraCheckStatus.SUCCESS,
                kind=AlgebraCheckKind.STATEMENT,
                summary="Closed algebraic expression evaluated exactly.",
                exact=True,
                diagnostics=(AlgebraDiagnostic(AlgebraDiagnosticCode.SUCCESS, "Closed expression evaluated."),),
                evidence=(AlgebraEvidence("expression", statement, exact_val, exact_val),),
            )

        assignments = _extract_exact_assignments(constraints)
        left = _parse_expr_safe(lhs).subs(_sub_map(assignments))
        right = _parse_expr_safe(rhs).subs(_sub_map(assignments))
        kind = AlgebraCheckKind.INEQUALITY if op != "=" else AlgebraCheckKind.STATEMENT
        status, summary, exact, payload = _truth_from_relation(left, op, right)
        code = (
            AlgebraDiagnosticCode.SUCCESS
            if status is AlgebraCheckStatus.SUCCESS
            else AlgebraDiagnosticCode.CONTRADICTION
            if status is AlgebraCheckStatus.CONTRADICTION
            else AlgebraDiagnosticCode.UNSUPPORTED_FORM
        )
        return _record_result(
            kind=kind,
            status=status,
            summary=summary,
            exact=exact,
            expression=statement,
            simplified_expression=str(payload.get("difference", "")),
            exact_value=payload.get("difference"),
            diagnostics=(AlgebraDiagnostic(code, summary, payload),),
            contradiction_found=status is AlgebraCheckStatus.CONTRADICTION,
            failure_type=FailureType.SYMBOLIC_MISMATCH if status is AlgebraCheckStatus.CONTRADICTION else None,
            metadata={"assignments": {k: str(v) for k, v in assignments.items()}, **payload},
        )
    except ValueError as exc:
        return _result(
            status=AlgebraCheckStatus.MALFORMED_INPUT,
            kind=AlgebraCheckKind.STATEMENT,
            summary=str(exc),
            exact=False,
            diagnostics=(AlgebraDiagnostic(AlgebraDiagnosticCode.PARSE_ERROR, str(exc)),),
            failure_type=FailureType.SYMBOLIC_MISMATCH,
        )
    except Exception as exc:
        return _result(
            status=AlgebraCheckStatus.EXECUTION_FAILURE,
            kind=AlgebraCheckKind.STATEMENT,
            summary=f"Algebra validation failed: {exc}",
            exact=False,
            diagnostics=(AlgebraDiagnostic(AlgebraDiagnosticCode.EXECUTION_ERROR, str(exc)),),
            failure_type=FailureType.SYMBOLIC_MISMATCH,
        )


__all__ = [
    "AlgebraCheckStatus",
    "AlgebraCheckKind",
    "AlgebraDiagnosticCode",
    "AlgebraDiagnostic",
    "AlgebraEvidence",
    "AlgebraCheckResult",
    "check_equality",
    "check_equivalence",
    "check_substitution",
    "detect_algebraic_contradictions",
    "validate_constraint_slice",
    "validate_candidate_answer",
    "validate_algebra_statement",
]
po