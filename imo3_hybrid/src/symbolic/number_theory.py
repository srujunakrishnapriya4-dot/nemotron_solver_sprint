from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha1
import math
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
_PARSE_TRANSFORMS = standard_transformations + (
    convert_xor,
    implicit_multiplication_application,
)
_CONGRUENCE_RE = re.compile(
    r"^\s*(.+?)\s*(?:\u2261|==?)\s*(.+?)\s*\(\s*mod\s+(.+?)\s*\)\s*$",
    re.IGNORECASE,
)
_DIVIDES_RE = re.compile(
    r"^\s*(.+?)\s*(?:\||divides)\s*(.+?)\s*$",
    re.IGNORECASE,
)
_GCD_RE = re.compile(
    r"^\s*gcd\((.+?),(.+?)\)\s*=\s*(.+?)\s*$",
    re.IGNORECASE,
)
_LCM_RE = re.compile(
    r"^\s*lcm\((.+?),(.+?)\)\s*=\s*(.+?)\s*$",
    re.IGNORECASE,
)
_PARITY_RE = re.compile(
    r"^\s*(.+?)\s+is\s+(even|odd)\s*$",
    re.IGNORECASE,
)
_INTEGRAL_RE = re.compile(
    r"^\s*(.+?)\s+is\s+integer\s*$",
    re.IGNORECASE,
)
_ASSIGN_RE = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$",
)


class NTCheckStatus(str, Enum):
    SUCCESS = "success"
    CONTRADICTION = "contradiction"
    UNSUPPORTED = "unsupported"
    TIMEOUT = "timeout"
    MALFORMED_INPUT = "malformed_input"
    EXECUTION_FAILURE = "execution_failure"


class NTCheckKind(str, Enum):
    DIVISIBILITY = "divisibility"
    CONGRUENCE = "congruence"
    PARITY = "parity"
    INTEGRALITY = "integrality"
    GCD_LCM = "gcd_lcm"
    RESIDUE_CLASS = "residue_class"
    CANDIDATE_ANSWER = "candidate_answer"
    CONSTRAINT_SLICE = "constraint_slice"
    CONTRADICTION_SCAN = "contradiction_scan"
    STATEMENT = "statement"


class NTDiagnosticCode(str, Enum):
    PARSE_ERROR = "parse_error"
    NON_INTEGER_DOMAIN = "non_integer_domain"
    UNSUPPORTED_FORM = "unsupported_form"
    CONTRADICTION = "contradiction"
    EXECUTION_ERROR = "execution_error"
    SUCCESS = "success"
    MISSING_ASSIGNMENT = "missing_assignment"


class ParityValue(str, Enum):
    EVEN = "even"
    ODD = "odd"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class NTDiagnostic:
    code: NTDiagnosticCode
    message: str
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NTEvidence:
    label: str
    expression: str
    normalized_expression: str
    exact_value: str | None
    note: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NTCheckResult:
    status: NTCheckStatus
    kind: NTCheckKind
    summary: str
    score: float
    exact: bool
    diagnostics: tuple[NTDiagnostic, ...] = field(default_factory=tuple)
    evidence: tuple[NTEvidence, ...] = field(default_factory=tuple)
    derived_facts: tuple[str, ...] = field(default_factory=tuple)
    contradiction_found: bool = False
    failure_type: FailureType | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status is NTCheckStatus.SUCCESS

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


@dataclass(frozen=True)
class NTConstraintSliceResult:
    checks: tuple[NTCheckResult, ...]
    combined: NTCheckResult


@dataclass(frozen=True)
class NTCandidateAnswerResult:
    combined: NTCheckResult


def _stable_id(prefix: str, summary: str, metadata: Mapping[str, Any]) -> str:
    raw = f"{prefix}|{summary}|{sorted((metadata or {}).items())}"
    return f"{prefix}_{sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def _score(status: NTCheckStatus, *, exact: bool) -> float:
    if status is NTCheckStatus.SUCCESS:
        return 1.0 if exact else 0.8
    if status is NTCheckStatus.UNSUPPORTED:
        return 0.15
    if status is NTCheckStatus.MALFORMED_INPUT:
        return 0.05
    return 0.0


def _result(
    *,
    status: NTCheckStatus,
    kind: NTCheckKind,
    summary: str,
    exact: bool,
    diagnostics: Sequence[NTDiagnostic] = (),
    evidence: Sequence[NTEvidence] = (),
    derived_facts: Sequence[str] = (),
    contradiction_found: bool = False,
    failure_type: FailureType | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> NTCheckResult:
    return NTCheckResult(
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


def _parse_expr_safe(text: str | int) -> sp.Expr:
    normalized = str(text).strip()
    if not normalized:
        raise ValueError("empty expression")
    if len(normalized) > _MAX_EXPR_CHARS:
        raise ValueError("expression too long for bounded number-theory validation")
    return parse_expr(normalized, transformations=_PARSE_TRANSFORMS, evaluate=True)


def _simplify_safe(expr: sp.Expr) -> sp.Expr:
    try:
        return sp.simplify(expr)
    except Exception:
        return expr


def _is_integer_expr(expr: sp.Expr) -> bool:
    if expr.is_integer is True:
        return True
    if expr.free_symbols:
        return False
    simplified = _simplify_safe(expr)
    return simplified.is_integer is True


def _as_exact_int(expr: sp.Expr) -> int | None:
    if not _is_integer_expr(expr):
        return None
    if expr.free_symbols:
        return None
    simplified = _simplify_safe(expr)
    try:
        return int(simplified)
    except Exception:
        return None


def _extract_integer_assignments(constraints: Sequence[str]) -> dict[str, int]:
    assignments: dict[str, int] = {}
    for text in constraints:
        match = _ASSIGN_RE.match(text or "")
        if not match:
            continue
        name = match.group(1)
        rhs = _parse_expr_safe(match.group(2))
        value = _as_exact_int(rhs)
        if value is None:
            continue
        if name in assignments and assignments[name] != value:
            raise ValueError(
                f"Contradictory integer assignments for {name}: {assignments[name]} vs {value}"
            )
        assignments[name] = value
    return assignments


def _subs_map(assignments: Mapping[str, int]) -> dict[sp.Symbol, int]:
    return {sp.Symbol(name): value for name, value in assignments.items()}


def _expr_key(expr: sp.Expr) -> str:
    return str(_simplify_safe(expr))


def _unsupported_non_integer(
    kind: NTCheckKind,
    summary: str,
    *,
    payload: Mapping[str, Any] | None = None,
) -> NTCheckResult:
    return _result(
        status=NTCheckStatus.UNSUPPORTED,
        kind=kind,
        summary=summary,
        exact=False,
        diagnostics=(
            NTDiagnostic(
                NTDiagnosticCode.NON_INTEGER_DOMAIN,
                summary,
                payload or {},
            ),
        ),
        failure_type=FailureType.COVERAGE_GAP,
        metadata=payload,
    )


def _evaluate_integer_expr(
    text: str | int,
    assignments: Mapping[str, int],
) -> tuple[sp.Expr, int | None]:
    reduced = _parse_expr_safe(text).subs(_subs_map(assignments))
    return reduced, _as_exact_int(reduced)


def normalize_modulus(value: int) -> int | None:
    if int(value) == 0:
        return None
    return abs(int(value))


def normalize_residue(value: int, modulus: int) -> int:
    mod = normalize_modulus(modulus)
    if mod is None:
        raise ValueError("modulus must be non-zero")
    return int(value) % mod


def modular_equivalent_int(lhs: int, rhs: int, modulus: int) -> bool:
    mod = normalize_modulus(modulus)
    if mod is None:
        raise ValueError("modulus must be non-zero")
    return normalize_residue(lhs, mod) == normalize_residue(rhs, mod)


def gcd_int(a: int, b: int) -> int:
    return math.gcd(int(a), int(b))


def lcm_int(a: int, b: int) -> int:
    aa = int(a)
    bb = int(b)
    if aa == 0 or bb == 0:
        return 0
    return abs(aa * bb) // math.gcd(aa, bb)


def parity_of_int(value: int) -> ParityValue:
    return ParityValue.EVEN if int(value) % 2 == 0 else ParityValue.ODD


def prime_valuation(n: int, p: int) -> int | None:
    nn = int(n)
    pp = abs(int(p))
    if pp < 2 or nn == 0:
        return None
    count = 0
    value = abs(nn)
    while value % pp == 0:
        value //= pp
        count += 1
    return count


def factorization_payload(n: int) -> Mapping[str, int]:
    value = int(n)
    if value == 0:
        return {"0": 1}
    factors = sp.factorint(abs(value))
    payload = {str(prime): int(power) for prime, power in sorted(factors.items())}
    if value < 0:
        payload["-1"] = 1
    return payload


def residue_class_feasible_int(value: int, modulus: int, allowed_residues: Sequence[int]) -> bool:
    mod = normalize_modulus(modulus)
    if mod is None:
        raise ValueError("modulus must be non-zero")
    allowed = {normalize_residue(item, mod) for item in allowed_residues}
    return normalize_residue(value, mod) in allowed


def check_divisibility(
    divisor: str | int,
    dividend: str | int,
    constraints: Sequence[str] = (),
) -> NTCheckResult:
    try:
        assignments = _extract_integer_assignments(constraints)
        reduced_divisor, divisor_int = _evaluate_integer_expr(divisor, assignments)
        reduced_dividend, dividend_int = _evaluate_integer_expr(dividend, assignments)
        if divisor_int is None or dividend_int is None:
            return _unsupported_non_integer(
                NTCheckKind.DIVISIBILITY,
                "Divisibility requires exactly reducible integer operands.",
                payload={
                    "divisor": str(reduced_divisor),
                    "dividend": str(reduced_dividend),
                },
            )
        if divisor_int == 0:
            return _result(
                status=NTCheckStatus.MALFORMED_INPUT,
                kind=NTCheckKind.DIVISIBILITY,
                summary="Divisibility by zero is malformed.",
                exact=False,
                diagnostics=(NTDiagnostic(NTDiagnosticCode.PARSE_ERROR, "Divisor reduced to zero."),),
                failure_type=FailureType.SYMBOLIC_MISMATCH,
            )

        remainder = dividend_int % divisor_int
        ok = remainder == 0
        summary = (
            f"Divisibility holds exactly: {divisor_int} | {dividend_int}."
            if ok
            else f"Divisibility fails exactly: {divisor_int} does not divide {dividend_int}."
        )
        return _result(
            status=NTCheckStatus.SUCCESS if ok else NTCheckStatus.CONTRADICTION,
            kind=NTCheckKind.DIVISIBILITY,
            summary=summary,
            exact=True,
            diagnostics=(
                NTDiagnostic(
                    NTDiagnosticCode.SUCCESS if ok else NTDiagnosticCode.CONTRADICTION,
                    "Exact divisibility evaluated.",
                    {
                        "divisor": divisor_int,
                        "dividend": dividend_int,
                        "remainder": remainder,
                    },
                ),
            ),
            evidence=(
                NTEvidence(
                    label="divisibility",
                    expression=f"{divisor} | {dividend}",
                    normalized_expression=f"{divisor_int} | {dividend_int}",
                    exact_value=str(remainder),
                    payload={"remainder": remainder},
                ),
            ),
            derived_facts=(f"{divisor_int} | {dividend_int}",) if ok else (),
            contradiction_found=not ok,
            failure_type=FailureType.SYMBOLIC_MISMATCH if not ok else None,
            metadata={
                "assignments": dict(assignments),
                "divisor": divisor_int,
                "dividend": dividend_int,
                "remainder": remainder,
                "divisor_factorization": factorization_payload(divisor_int),
                "dividend_factorization": factorization_payload(dividend_int),
            },
        )
    except ValueError as exc:
        return _result(
            status=NTCheckStatus.MALFORMED_INPUT,
            kind=NTCheckKind.DIVISIBILITY,
            summary=str(exc),
            exact=False,
            diagnostics=(NTDiagnostic(NTDiagnosticCode.PARSE_ERROR, str(exc)),),
            failure_type=FailureType.SYMBOLIC_MISMATCH,
        )
    except Exception as exc:
        return _result(
            status=NTCheckStatus.EXECUTION_FAILURE,
            kind=NTCheckKind.DIVISIBILITY,
            summary=f"Divisibility check failed: {exc}",
            exact=False,
            diagnostics=(NTDiagnostic(NTDiagnosticCode.EXECUTION_ERROR, str(exc)),),
            failure_type=FailureType.SYMBOLIC_MISMATCH,
        )


def check_congruence(
    lhs: str | int,
    rhs: str | int,
    modulus: str | int,
    constraints: Sequence[str] = (),
) -> NTCheckResult:
    try:
        assignments = _extract_integer_assignments(constraints)
        reduced_lhs, lhs_int = _evaluate_integer_expr(lhs, assignments)
        reduced_rhs, rhs_int = _evaluate_integer_expr(rhs, assignments)
        reduced_modulus, modulus_int = _evaluate_integer_expr(modulus, assignments)

        if lhs_int is None or rhs_int is None or modulus_int is None:
            return _unsupported_non_integer(
                NTCheckKind.CONGRUENCE,
                "Congruence requires exactly reducible integer operands and modulus.",
                payload={
                    "lhs": str(reduced_lhs),
                    "rhs": str(reduced_rhs),
                    "modulus": str(reduced_modulus),
                },
            )

        normalized_modulus = normalize_modulus(modulus_int)
        if normalized_modulus is None:
            return _result(
                status=NTCheckStatus.MALFORMED_INPUT,
                kind=NTCheckKind.CONGRUENCE,
                summary="Congruence modulo zero is malformed.",
                exact=False,
                diagnostics=(NTDiagnostic(NTDiagnosticCode.PARSE_ERROR, "Modulus reduced to zero."),),
                failure_type=FailureType.SYMBOLIC_MISMATCH,
            )

        lhs_residue = normalize_residue(lhs_int, normalized_modulus)
        rhs_residue = normalize_residue(rhs_int, normalized_modulus)
        ok = lhs_residue == rhs_residue
        summary = (
            f"Congruence holds exactly: {lhs_int} == {rhs_int} (mod {normalized_modulus})."
            if ok
            else f"Congruence fails exactly: residues {lhs_residue} and {rhs_residue} differ mod {normalized_modulus}."
        )
        return _result(
            status=NTCheckStatus.SUCCESS if ok else NTCheckStatus.CONTRADICTION,
            kind=NTCheckKind.CONGRUENCE,
            summary=summary,
            exact=True,
            diagnostics=(
                NTDiagnostic(
                    NTDiagnosticCode.SUCCESS if ok else NTDiagnosticCode.CONTRADICTION,
                    "Exact congruence evaluated.",
                    {
                        "lhs": lhs_int,
                        "rhs": rhs_int,
                        "modulus": normalized_modulus,
                        "lhs_residue": lhs_residue,
                        "rhs_residue": rhs_residue,
                    },
                ),
            ),
            evidence=(
                NTEvidence(
                    label="congruence",
                    expression=f"{lhs} == {rhs} (mod {modulus})",
                    normalized_expression=f"{lhs_int} == {rhs_int} (mod {normalized_modulus})",
                    exact_value=str(lhs_residue),
                    payload={
                        "lhs_residue": lhs_residue,
                        "rhs_residue": rhs_residue,
                    },
                ),
            ),
            derived_facts=(f"{lhs_int} == {rhs_int} (mod {normalized_modulus})",) if ok else (),
            contradiction_found=not ok,
            failure_type=FailureType.SYMBOLIC_MISMATCH if not ok else None,
            metadata={
                "assignments": dict(assignments),
                "lhs": lhs_int,
                "rhs": rhs_int,
                "modulus": normalized_modulus,
            },
        )
    except ValueError as exc:
        return _result(
            status=NTCheckStatus.MALFORMED_INPUT,
            kind=NTCheckKind.CONGRUENCE,
            summary=str(exc),
            exact=False,
            diagnostics=(NTDiagnostic(NTDiagnosticCode.PARSE_ERROR, str(exc)),),
            failure_type=FailureType.SYMBOLIC_MISMATCH,
        )
    except Exception as exc:
        return _result(
            status=NTCheckStatus.EXECUTION_FAILURE,
            kind=NTCheckKind.CONGRUENCE,
            summary=f"Congruence check failed: {exc}",
            exact=False,
            diagnostics=(NTDiagnostic(NTDiagnosticCode.EXECUTION_ERROR, str(exc)),),
            failure_type=FailureType.SYMBOLIC_MISMATCH,
        )


def check_parity(
    expr: str | int,
    expected: str | ParityValue | None = None,
    constraints: Sequence[str] = (),
) -> NTCheckResult:
    try:
        assignments = _extract_integer_assignments(constraints)
        reduced, value = _evaluate_integer_expr(expr, assignments)
        if value is None:
            return _unsupported_non_integer(
                NTCheckKind.PARITY,
                "Parity requires an exactly reducible integer expression.",
                payload={"expression": str(reduced)},
            )

        actual = parity_of_int(value)
        if expected is None:
            return _result(
                status=NTCheckStatus.SUCCESS,
                kind=NTCheckKind.PARITY,
                summary=f"Parity determined exactly: {actual.value}.",
                exact=True,
                diagnostics=(
                    NTDiagnostic(
                        NTDiagnosticCode.SUCCESS,
                        "Parity determined exactly.",
                        {"parity": actual.value},
                    ),
                ),
                evidence=(
                    NTEvidence(
                        label="parity",
                        expression=str(expr),
                        normalized_expression=str(value),
                        exact_value=actual.value,
                    ),
                ),
                derived_facts=(f"{expr} is {actual.value}",),
                metadata={"parity": actual.value, "assignments": dict(assignments)},
            )

        if isinstance(expected, ParityValue):
            expected_parity = expected
        else:
            lowered = str(expected).strip().lower()
            if lowered not in {ParityValue.EVEN.value, ParityValue.ODD.value}:
                raise ValueError(f"Unsupported parity expectation: {expected}")
            expected_parity = ParityValue(lowered)

        ok = actual is expected_parity
        return _result(
            status=NTCheckStatus.SUCCESS if ok else NTCheckStatus.CONTRADICTION,
            kind=NTCheckKind.PARITY,
            summary=f"Expected parity {expected_parity.value}; exact evaluation gives {actual.value}.",
            exact=True,
            diagnostics=(
                NTDiagnostic(
                    NTDiagnosticCode.SUCCESS if ok else NTDiagnosticCode.CONTRADICTION,
                    "Parity expectation checked exactly.",
                    {"expected": expected_parity.value, "actual": actual.value},
                ),
            ),
            evidence=(
                NTEvidence(
                    label="parity",
                    expression=f"{expr} is {expected_parity.value}",
                    normalized_expression=str(value),
                    exact_value=actual.value,
                ),
            ),
            derived_facts=(f"{expr} is {actual.value}",) if ok else (),
            contradiction_found=not ok,
            failure_type=FailureType.SYMBOLIC_MISMATCH if not ok else None,
            metadata={
                "parity": actual.value,
                "expected_parity": expected_parity.value,
                "assignments": dict(assignments),
            },
        )
    except ValueError as exc:
        return _result(
            status=NTCheckStatus.MALFORMED_INPUT,
            kind=NTCheckKind.PARITY,
            summary=str(exc),
            exact=False,
            diagnostics=(NTDiagnostic(NTDiagnosticCode.PARSE_ERROR, str(exc)),),
            failure_type=FailureType.SYMBOLIC_MISMATCH,
        )
    except Exception as exc:
        return _result(
            status=NTCheckStatus.EXECUTION_FAILURE,
            kind=NTCheckKind.PARITY,
            summary=f"Parity check failed: {exc}",
            exact=False,
            diagnostics=(NTDiagnostic(NTDiagnosticCode.EXECUTION_ERROR, str(exc)),),
            failure_type=FailureType.SYMBOLIC_MISMATCH,
        )


def check_integrality(
    expr: str | int,
    constraints: Sequence[str] = (),
) -> NTCheckResult:
    try:
        assignments = _extract_integer_assignments(constraints)
        reduced = _parse_expr_safe(expr).subs(_subs_map(assignments))
        value = _as_exact_int(reduced)
        if value is not None:
            return _result(
                status=NTCheckStatus.SUCCESS,
                kind=NTCheckKind.INTEGRALITY,
                summary="Expression is exactly integral.",
                exact=True,
                diagnostics=(
                    NTDiagnostic(
                        NTDiagnosticCode.SUCCESS,
                        "Integrality established exactly.",
                        {"value": value},
                    ),
                ),
                evidence=(
                    NTEvidence(
                        label="integrality",
                        expression=str(expr),
                        normalized_expression=str(value),
                        exact_value="integer",
                    ),
                ),
                derived_facts=(f"{expr} is integer",),
                metadata={"assignments": dict(assignments), "value": value},
            )

        if reduced.free_symbols:
            return _result(
                status=NTCheckStatus.UNSUPPORTED,
                kind=NTCheckKind.INTEGRALITY,
                summary="Integrality cannot be decided exactly with remaining free symbols.",
                exact=False,
                diagnostics=(
                    NTDiagnostic(
                        NTDiagnosticCode.MISSING_ASSIGNMENT,
                        "Unresolved free symbols remain.",
                        {"free_symbols": sorted(str(item) for item in reduced.free_symbols)},
                    ),
                ),
                failure_type=FailureType.COVERAGE_GAP,
                metadata={"assignments": dict(assignments)},
            )

        return _result(
            status=NTCheckStatus.CONTRADICTION,
            kind=NTCheckKind.INTEGRALITY,
            summary="Expression evaluates to a non-integer value.",
            exact=True,
            diagnostics=(
                NTDiagnostic(
                    NTDiagnosticCode.CONTRADICTION,
                    "Integrality failed exactly.",
                    {"evaluated_expression": str(_simplify_safe(reduced))},
                ),
            ),
            evidence=(
                NTEvidence(
                    label="integrality",
                    expression=str(expr),
                    normalized_expression=str(_simplify_safe(reduced)),
                    exact_value=None,
                ),
            ),
            contradiction_found=True,
            failure_type=FailureType.SYMBOLIC_MISMATCH,
            metadata={"assignments": dict(assignments)},
        )
    except ValueError as exc:
        return _result(
            status=NTCheckStatus.MALFORMED_INPUT,
            kind=NTCheckKind.INTEGRALITY,
            summary=str(exc),
            exact=False,
            diagnostics=(NTDiagnostic(NTDiagnosticCode.PARSE_ERROR, str(exc)),),
            failure_type=FailureType.SYMBOLIC_MISMATCH,
        )
    except Exception as exc:
        return _result(
            status=NTCheckStatus.EXECUTION_FAILURE,
            kind=NTCheckKind.INTEGRALITY,
            summary=f"Integrality check failed: {exc}",
            exact=False,
            diagnostics=(NTDiagnostic(NTDiagnosticCode.EXECUTION_ERROR, str(exc)),),
            failure_type=FailureType.SYMBOLIC_MISMATCH,
        )


def check_gcd_lcm_relation(
    a: str | int,
    b: str | int,
    *,
    gcd_expected: str | int | None = None,
    lcm_expected: str | int | None = None,
    constraints: Sequence[str] = (),
) -> NTCheckResult:
    try:
        assignments = _extract_integer_assignments(constraints)
        reduced_a, a_int = _evaluate_integer_expr(a, assignments)
        reduced_b, b_int = _evaluate_integer_expr(b, assignments)
        if a_int is None or b_int is None:
            return _unsupported_non_integer(
                NTCheckKind.GCD_LCM,
                "gcd/lcm checks require exactly reducible integer operands.",
                payload={"a": str(reduced_a), "b": str(reduced_b)},
            )

        gcd_value = gcd_int(a_int, b_int)
        lcm_value = lcm_int(a_int, b_int)
        expected_gcd_int: int | None = None
        expected_lcm_int: int | None = None

        if gcd_expected is not None:
            reduced_gcd, expected_gcd_int = _evaluate_integer_expr(gcd_expected, assignments)
            if expected_gcd_int is None:
                return _unsupported_non_integer(
                    NTCheckKind.GCD_LCM,
                    "Expected gcd must reduce exactly to an integer.",
                    payload={"gcd_expected": str(reduced_gcd)},
                )

        if lcm_expected is not None:
            reduced_lcm, expected_lcm_int = _evaluate_integer_expr(lcm_expected, assignments)
            if expected_lcm_int is None:
                return _unsupported_non_integer(
                    NTCheckKind.GCD_LCM,
                    "Expected lcm must reduce exactly to an integer.",
                    payload={"lcm_expected": str(reduced_lcm)},
                )

        checks: list[bool] = []
        summary_parts: list[str] = []
        if expected_gcd_int is not None:
            checks.append(gcd_value == expected_gcd_int)
            summary_parts.append(f"gcd={gcd_value}, expected {expected_gcd_int}")
        if expected_lcm_int is not None:
            checks.append(lcm_value == expected_lcm_int)
            summary_parts.append(f"lcm={lcm_value}, expected {expected_lcm_int}")
        if not checks:
            summary_parts.append(f"gcd={gcd_value}, lcm={lcm_value}")

        ok = all(checks) if checks else True
        return _result(
            status=NTCheckStatus.SUCCESS if ok else NTCheckStatus.CONTRADICTION,
            kind=NTCheckKind.GCD_LCM,
            summary="gcd/lcm evaluated exactly: " + "; ".join(summary_parts) + ".",
            exact=True,
            diagnostics=(
                NTDiagnostic(
                    NTDiagnosticCode.SUCCESS if ok else NTDiagnosticCode.CONTRADICTION,
                    "gcd/lcm evaluated exactly.",
                    {
                        "a": a_int,
                        "b": b_int,
                        "gcd": gcd_value,
                        "lcm": lcm_value,
                        "gcd_expected": expected_gcd_int,
                        "lcm_expected": expected_lcm_int,
                    },
                ),
            ),
            evidence=(
                NTEvidence(
                    label="gcd",
                    expression=f"gcd({a}, {b})",
                    normalized_expression=f"gcd({a_int}, {b_int})",
                    exact_value=str(gcd_value),
                    payload={"factorization_a": factorization_payload(a_int)},
                ),
                NTEvidence(
                    label="lcm",
                    expression=f"lcm({a}, {b})",
                    normalized_expression=f"lcm({a_int}, {b_int})",
                    exact_value=str(lcm_value),
                    payload={"factorization_b": factorization_payload(b_int)},
                ),
            ),
            derived_facts=(
                f"gcd({a_int}, {b_int}) = {gcd_value}",
                f"lcm({a_int}, {b_int}) = {lcm_value}",
            ) if ok else (),
            contradiction_found=not ok,
            failure_type=FailureType.SYMBOLIC_MISMATCH if not ok else None,
            metadata={
                "assignments": dict(assignments),
                "a": a_int,
                "b": b_int,
                "gcd": gcd_value,
                "lcm": lcm_value,
            },
        )
    except ValueError as exc:
        return _result(
            status=NTCheckStatus.MALFORMED_INPUT,
            kind=NTCheckKind.GCD_LCM,
            summary=str(exc),
            exact=False,
            diagnostics=(NTDiagnostic(NTDiagnosticCode.PARSE_ERROR, str(exc)),),
            failure_type=FailureType.SYMBOLIC_MISMATCH,
        )
    except Exception as exc:
        return _result(
            status=NTCheckStatus.EXECUTION_FAILURE,
            kind=NTCheckKind.GCD_LCM,
            summary=f"gcd/lcm check failed: {exc}",
            exact=False,
            diagnostics=(NTDiagnostic(NTDiagnosticCode.EXECUTION_ERROR, str(exc)),),
            failure_type=FailureType.SYMBOLIC_MISMATCH,
        )


def validate_residue_class(
    expr: str | int,
    *,
    modulus: str | int,
    allowed_residues: Sequence[int] | Sequence[str],
    constraints: Sequence[str] = (),
) -> NTCheckResult:
    try:
        assignments = _extract_integer_assignments(constraints)
        reduced_expr, expr_int = _evaluate_integer_expr(expr, assignments)
        reduced_modulus, modulus_int = _evaluate_integer_expr(modulus, assignments)

        if modulus_int is None:
            return _unsupported_non_integer(
                NTCheckKind.RESIDUE_CLASS,
                "Residue-class feasibility requires an exact integer modulus.",
                payload={"modulus": str(reduced_modulus)},
            )

        normalized_modulus = normalize_modulus(modulus_int)
        if normalized_modulus is None:
            return _result(
                status=NTCheckStatus.MALFORMED_INPUT,
                kind=NTCheckKind.RESIDUE_CLASS,
                summary="Residue classes modulo zero are malformed.",
                exact=False,
                diagnostics=(NTDiagnostic(NTDiagnosticCode.PARSE_ERROR, "Modulus reduced to zero."),),
                failure_type=FailureType.SYMBOLIC_MISMATCH,
            )

        normalized_allowed: list[int] = []
        for item in allowed_residues:
            _, residue_int = _evaluate_integer_expr(item, assignments)
            if residue_int is None:
                raise ValueError(f"Allowed residue is not an exact integer: {item}")
            normalized_allowed.append(normalize_residue(residue_int, normalized_modulus))
        normalized_allowed = sorted(set(normalized_allowed))

        if expr_int is None:
            if reduced_expr.free_symbols:
                return _result(
                    status=NTCheckStatus.UNSUPPORTED,
                    kind=NTCheckKind.RESIDUE_CLASS,
                    summary="Residue-class feasibility is unsupported with remaining free symbols.",
                    exact=False,
                    diagnostics=(
                        NTDiagnostic(
                            NTDiagnosticCode.MISSING_ASSIGNMENT,
                            "Unresolved free symbols remain.",
                            {"free_symbols": sorted(str(item) for item in reduced_expr.free_symbols)},
                        ),
                    ),
                    failure_type=FailureType.COVERAGE_GAP,
                    metadata={
                        "allowed_residues": normalized_allowed,
                        "modulus": normalized_modulus,
                    },
                )
            return _unsupported_non_integer(
                NTCheckKind.RESIDUE_CLASS,
                "Residue-class feasibility requires an exact integer expression.",
                payload={"expr": str(reduced_expr)},
            )

        actual_residue = normalize_residue(expr_int, normalized_modulus)
        ok = actual_residue in normalized_allowed
        summary = (
            f"Residue-class check passed: {expr_int} mod {normalized_modulus} = {actual_residue}."
            if ok
            else f"Residue-class check failed: residue {actual_residue} not in allowed set {normalized_allowed}."
        )
        return _result(
            status=NTCheckStatus.SUCCESS if ok else NTCheckStatus.CONTRADICTION,
            kind=NTCheckKind.RESIDUE_CLASS,
            summary=summary,
            exact=True,
            diagnostics=(
                NTDiagnostic(
                    NTDiagnosticCode.SUCCESS if ok else NTDiagnosticCode.CONTRADICTION,
                    "Residue-class feasibility evaluated exactly.",
                    {
                        "expr": expr_int,
                        "modulus": normalized_modulus,
                        "actual_residue": actual_residue,
                        "allowed_residues": normalized_allowed,
                    },
                ),
            ),
            evidence=(
                NTEvidence(
                    label="residue_class",
                    expression=str(expr),
                    normalized_expression=str(expr_int),
                    exact_value=str(actual_residue),
                    payload={
                        "modulus": normalized_modulus,
                        "allowed_residues": normalized_allowed,
                    },
                ),
            ),
            derived_facts=(f"{expr_int} == {actual_residue} (mod {normalized_modulus})",) if ok else (),
            contradiction_found=not ok,
            failure_type=FailureType.SYMBOLIC_MISMATCH if not ok else None,
            metadata={
                "assignments": dict(assignments),
                "expr": expr_int,
                "modulus": normalized_modulus,
                "actual_residue": actual_residue,
                "allowed_residues": normalized_allowed,
            },
        )
    except ValueError as exc:
        return _result(
            status=NTCheckStatus.MALFORMED_INPUT,
            kind=NTCheckKind.RESIDUE_CLASS,
            summary=str(exc),
            exact=False,
            diagnostics=(NTDiagnostic(NTDiagnosticCode.PARSE_ERROR, str(exc)),),
            failure_type=FailureType.SYMBOLIC_MISMATCH,
        )
    except Exception as exc:
        return _result(
            status=NTCheckStatus.EXECUTION_FAILURE,
            kind=NTCheckKind.RESIDUE_CLASS,
            summary=f"Residue-class check failed: {exc}",
            exact=False,
            diagnostics=(NTDiagnostic(NTDiagnosticCode.EXECUTION_ERROR, str(exc)),),
            failure_type=FailureType.SYMBOLIC_MISMATCH,
        )


def detect_integer_contradictions(constraints: Sequence[str]) -> NTCheckResult:
    diagnostics: list[NTDiagnostic] = []
    evidence: list[NTEvidence] = []
    derived: list[str] = []

    try:
        assignments = _extract_integer_assignments(constraints)
    except ValueError as exc:
        return _result(
            status=NTCheckStatus.CONTRADICTION,
            kind=NTCheckKind.CONTRADICTION_SCAN,
            summary=str(exc),
            exact=True,
            diagnostics=(NTDiagnostic(NTDiagnosticCode.CONTRADICTION, str(exc)),),
            contradiction_found=True,
            failure_type=FailureType.SYMBOLIC_MISMATCH,
        )
    except Exception as exc:
        return _result(
            status=NTCheckStatus.EXECUTION_FAILURE,
            kind=NTCheckKind.CONTRADICTION_SCAN,
            summary=f"Contradiction scan failed while parsing assignments: {exc}",
            exact=False,
            diagnostics=(NTDiagnostic(NTDiagnosticCode.EXECUTION_ERROR, str(exc)),),
            failure_type=FailureType.SYMBOLIC_MISMATCH,
        )

    parity_claims: dict[str, ParityValue] = {}
    congruence_claims: dict[tuple[str, int], int] = {}

    for text in constraints:
        statement = (text or "").strip()
        if not statement:
            continue

        parity_match = _PARITY_RE.match(statement)
        if parity_match:
            try:
                reduced_expr = _parse_expr_safe(parity_match.group(1)).subs(_subs_map(assignments))
                expr_key = _expr_key(reduced_expr)
                parity_value = ParityValue(parity_match.group(2).strip().lower())
                previous = parity_claims.get(expr_key)
                if previous is not None and previous is not parity_value:
                    return _result(
                        status=NTCheckStatus.CONTRADICTION,
                        kind=NTCheckKind.CONTRADICTION_SCAN,
                        summary=f"Conflicting parity claims for {expr_key}: {previous.value} vs {parity_value.value}.",
                        exact=True,
                        diagnostics=(
                            NTDiagnostic(
                                NTDiagnosticCode.CONTRADICTION,
                                "Parity contradiction detected across constraints.",
                                {"expression": expr_key},
                            ),
                        ),
                        contradiction_found=True,
                        failure_type=FailureType.SYMBOLIC_MISMATCH,
                    )
                parity_claims[expr_key] = parity_value
            except Exception:
                pass

        congruence_match = _CONGRUENCE_RE.match(statement)
        if congruence_match:
            try:
                reduced_lhs = _parse_expr_safe(congruence_match.group(1)).subs(_subs_map(assignments))
                _, rhs_int = _evaluate_integer_expr(congruence_match.group(2), assignments)
                _, modulus_int = _evaluate_integer_expr(congruence_match.group(3), assignments)
                if rhs_int is not None and modulus_int is not None:
                    normalized_modulus = normalize_modulus(modulus_int)
                    if normalized_modulus is not None:
                        key = (_expr_key(reduced_lhs), normalized_modulus)
                        residue = normalize_residue(rhs_int, normalized_modulus)
                        previous = congruence_claims.get(key)
                        if previous is not None and previous != residue:
                            return _result(
                                status=NTCheckStatus.CONTRADICTION,
                                kind=NTCheckKind.CONTRADICTION_SCAN,
                                summary=(
                                    f"Conflicting congruence residues for {key[0]} mod {normalized_modulus}: "
                                    f"{previous} vs {residue}."
                                ),
                                exact=True,
                                diagnostics=(
                                    NTDiagnostic(
                                        NTDiagnosticCode.CONTRADICTION,
                                        "Congruence contradiction detected across constraints.",
                                        {
                                            "expression": key[0],
                                            "modulus": normalized_modulus,
                                        },
                                    ),
                                ),
                                contradiction_found=True,
                                failure_type=FailureType.SYMBOLIC_MISMATCH,
                            )
                        congruence_claims[key] = residue
            except Exception:
                pass

        sub_result = validate_number_theory_statement(statement, constraints=constraints)
        if sub_result.status is NTCheckStatus.CONTRADICTION:
            return _result(
                status=NTCheckStatus.CONTRADICTION,
                kind=NTCheckKind.CONTRADICTION_SCAN,
                summary=sub_result.summary,
                exact=sub_result.exact,
                diagnostics=sub_result.diagnostics,
                evidence=sub_result.evidence,
                contradiction_found=True,
                failure_type=FailureType.SYMBOLIC_MISMATCH,
                metadata={"source_statement": statement},
            )

        diagnostics.extend(sub_result.diagnostics)
        evidence.extend(sub_result.evidence)
        derived.extend(sub_result.derived_facts)

    return _result(
        status=NTCheckStatus.SUCCESS,
        kind=NTCheckKind.CONTRADICTION_SCAN,
        summary="No exact number-theoretic contradiction detected in supported statements.",
        exact=True,
        diagnostics=tuple(diagnostics),
        evidence=tuple(evidence),
        derived_facts=tuple(dict.fromkeys(derived)),
        metadata={"assignments": dict(assignments)},
    )


def validate_constraint_slice(
    constraints: Sequence[str],
    candidate_answer: str | None = None,
    answer_symbol: str | None = None,
) -> NTCheckResult:
    contradiction = detect_integer_contradictions(constraints)
    if contradiction.status in {
        NTCheckStatus.CONTRADICTION,
        NTCheckStatus.EXECUTION_FAILURE,
        NTCheckStatus.MALFORMED_INPUT,
    }:
        return contradiction

    if candidate_answer is not None and answer_symbol:
        candidate_result = validate_candidate_answer(
            candidate_answer,
            constraints,
            answer_symbol,
        )
        if candidate_result.status in {
            NTCheckStatus.CONTRADICTION,
            NTCheckStatus.EXECUTION_FAILURE,
            NTCheckStatus.MALFORMED_INPUT,
        }:
            return candidate_result

    checks: list[NTCheckResult] = []
    supported = 0
    exact_success = 0
    diagnostics: list[NTDiagnostic] = list(contradiction.diagnostics)
    evidence: list[NTEvidence] = list(contradiction.evidence)

    for text in constraints:
        result = validate_number_theory_statement(text, constraints=constraints)
        if result.status is NTCheckStatus.UNSUPPORTED:
            continue
        checks.append(result)
        supported += 1
        if result.status is NTCheckStatus.SUCCESS and result.exact:
            exact_success += 1
        diagnostics.extend(result.diagnostics)
        evidence.extend(result.evidence)

    if supported == 0:
        return _result(
            status=NTCheckStatus.UNSUPPORTED,
            kind=NTCheckKind.CONSTRAINT_SLICE,
            summary="Constraint slice contains no supported number-theoretic statements.",
            exact=False,
            diagnostics=tuple(diagnostics) + (
                NTDiagnostic(
                    NTDiagnosticCode.UNSUPPORTED_FORM,
                    "No supported number-theoretic constraints found.",
                ),
            ),
            evidence=tuple(evidence),
            failure_type=FailureType.COVERAGE_GAP,
        )

    if any(item.status is NTCheckStatus.CONTRADICTION for item in checks):
        first = next(item for item in checks if item.status is NTCheckStatus.CONTRADICTION)
        return _result(
            status=NTCheckStatus.CONTRADICTION,
            kind=NTCheckKind.CONSTRAINT_SLICE,
            summary=first.summary,
            exact=all(item.exact for item in checks),
            diagnostics=tuple(diagnostics),
            evidence=tuple(evidence),
            derived_facts=tuple(dict.fromkeys(fact for item in checks for fact in item.derived_facts)),
            contradiction_found=True,
            failure_type=FailureType.SYMBOLIC_MISMATCH,
            metadata={"supported_checks": supported},
        )

    status = NTCheckStatus.SUCCESS if exact_success == supported else NTCheckStatus.UNSUPPORTED
    summary = (
        f"Validated {exact_success}/{supported} supported number-theoretic constraints exactly."
        if status is NTCheckStatus.SUCCESS
        else f"Partial number-theory support only: {exact_success}/{supported} supported checks were exact."
    )
    return _result(
        status=status,
        kind=NTCheckKind.CONSTRAINT_SLICE,
        summary=summary,
        exact=(status is NTCheckStatus.SUCCESS),
        diagnostics=tuple(diagnostics),
        evidence=tuple(evidence),
        derived_facts=tuple(dict.fromkeys(fact for item in checks for fact in item.derived_facts)),
        failure_type=None if status is NTCheckStatus.SUCCESS else FailureType.COVERAGE_GAP,
        metadata={
            "supported_checks": supported,
            "exact_successes": exact_success,
            "partial_support": exact_success > 0 and exact_success < supported,
        },
    )


def validate_candidate_answer(
    candidate_answer: str,
    constraints: Sequence[str],
    answer_symbol: str | None,
) -> NTCheckResult:
    if not answer_symbol:
        return _result(
            status=NTCheckStatus.UNSUPPORTED,
            kind=NTCheckKind.CANDIDATE_ANSWER,
            summary="Candidate-answer NT validation requires an answer symbol.",
            exact=False,
            diagnostics=(
                NTDiagnostic(
                    NTDiagnosticCode.MISSING_ASSIGNMENT,
                    "Missing answer symbol.",
                ),
            ),
            failure_type=FailureType.COVERAGE_GAP,
        )

    try:
        value_expr = _parse_expr_safe(candidate_answer)
    except Exception as exc:
        return _result(
            status=NTCheckStatus.MALFORMED_INPUT,
            kind=NTCheckKind.CANDIDATE_ANSWER,
            summary=f"Candidate answer could not be parsed: {exc}",
            exact=False,
            diagnostics=(NTDiagnostic(NTDiagnosticCode.PARSE_ERROR, str(exc)),),
            failure_type=FailureType.SYMBOLIC_MISMATCH,
        )

    if value_expr.free_symbols:
        return _result(
            status=NTCheckStatus.UNSUPPORTED,
            kind=NTCheckKind.CANDIDATE_ANSWER,
            summary="Candidate answer remains symbolic after parsing.",
            exact=False,
            diagnostics=(
                NTDiagnostic(
                    NTDiagnosticCode.MISSING_ASSIGNMENT,
                    "Candidate answer is not a concrete value.",
                ),
            ),
            failure_type=FailureType.COVERAGE_GAP,
        )

    candidate_int = _as_exact_int(value_expr)
    if candidate_int is None:
        return _unsupported_non_integer(
            NTCheckKind.CANDIDATE_ANSWER,
            "Candidate answer must reduce exactly to an integer for number-theory validation.",
            payload={"candidate_answer": candidate_answer},
        )

    augmented = list(constraints) + [f"{answer_symbol} = {candidate_int}"]
    result = validate_constraint_slice(augmented)
    summary = (
        result.summary
        if result.status is not NTCheckStatus.SUCCESS
        else f"Candidate answer {candidate_int} is consistent with the supported number-theoretic constraint slice."
    )
    metadata = dict(result.metadata)
    metadata["assignment"] = f"{answer_symbol} = {candidate_int}"

    return _result(
        status=result.status,
        kind=NTCheckKind.CANDIDATE_ANSWER,
        summary=summary,
        exact=result.exact,
        diagnostics=result.diagnostics,
        evidence=result.evidence,
        derived_facts=result.derived_facts,
        contradiction_found=result.contradiction_found,
        failure_type=result.failure_type,
        metadata=metadata,
    )


def validate_number_theory_statement(
    statement: str,
    constraints: Sequence[str] = (),
) -> NTCheckResult:
    text = (statement or "").strip()
    if not text:
        return _result(
            status=NTCheckStatus.MALFORMED_INPUT,
            kind=NTCheckKind.STATEMENT,
            summary="Empty number-theory statement.",
            exact=False,
            diagnostics=(NTDiagnostic(NTDiagnosticCode.PARSE_ERROR, "Empty statement."),),
            failure_type=FailureType.SYMBOLIC_MISMATCH,
        )

    congruence_match = _CONGRUENCE_RE.match(text)
    if congruence_match:
        return check_congruence(
            congruence_match.group(1),
            congruence_match.group(2),
            congruence_match.group(3),
            constraints,
        )

    divides_match = _DIVIDES_RE.match(text)
    if divides_match and "mod" not in text.lower():
        return check_divisibility(
            divides_match.group(1),
            divides_match.group(2),
            constraints,
        )

    parity_match = _PARITY_RE.match(text)
    if parity_match:
        return check_parity(
            parity_match.group(1),
            parity_match.group(2),
            constraints,
        )

    integral_match = _INTEGRAL_RE.match(text)
    if integral_match:
        return check_integrality(integral_match.group(1), constraints)

    gcd_match = _GCD_RE.match(text)
    if gcd_match:
        return check_gcd_lcm_relation(
            gcd_match.group(1),
            gcd_match.group(2),
            gcd_expected=gcd_match.group(3),
            constraints=constraints,
        )

    lcm_match = _LCM_RE.match(text)
    if lcm_match:
        return check_gcd_lcm_relation(
            lcm_match.group(1),
            lcm_match.group(2),
            lcm_expected=lcm_match.group(3),
            constraints=constraints,
        )

    return _result(
        status=NTCheckStatus.UNSUPPORTED,
        kind=NTCheckKind.STATEMENT,
        summary="Statement is outside the supported number-theoretic validation subset.",
        exact=False,
        diagnostics=(
            NTDiagnostic(
                NTDiagnosticCode.UNSUPPORTED_FORM,
                "Unsupported NT form.",
                {"statement": text},
            ),
        ),
        failure_type=FailureType.COVERAGE_GAP,
        metadata={"statement": text},
    )


__all__ = [
    "NTCheckStatus",
    "NTCheckKind",
    "NTDiagnosticCode",
    "ParityValue",
    "NTDiagnostic",
    "NTEvidence",
    "NTCheckResult",
    "NTConstraintSliceResult",
    "NTCandidateAnswerResult",
    "normalize_modulus",
    "normalize_residue",
    "modular_equivalent_int",
    "gcd_int",
    "lcm_int",
    "parity_of_int",
    "prime_valuation",
    "factorization_payload",
    "residue_class_feasible_int",
    "check_divisibility",
    "check_congruence",
    "check_parity",
    "check_integrality",
    "check_gcd_lcm_relation",
    "validate_residue_class",
    "detect_integer_contradictions",
    "validate_constraint_slice",
    "validate_candidate_answer",
    "validate_number_theory_statement",
]
