# src/symbolic/geometry.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
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


_TRANSFORMS = standard_transformations + (
    convert_xor,
    implicit_multiplication_application,
)
_POINT_DEF_RE = re.compile(
    r"\b([A-Za-z][A-Za-z0-9_]*)\s*=\s*\(\s*([^,]+?)\s*,\s*([^)]+?)\s*\)"
)


class GeometryCheckStatus(str, Enum):
    SUCCESS = "success"
    CONTRADICTION = "contradiction"
    UNSUPPORTED = "unsupported"
    TIMEOUT = "timeout"
    MALFORMED_INPUT = "malformed_input"
    EXECUTION_FAILURE = "execution_failure"


class GeometryCheckKind(str, Enum):
    COLLINEAR = "collinear"
    PARALLEL = "parallel"
    PERPENDICULAR = "perpendicular"
    DISTANCE_EQUALITY = "distance_equality"
    MIDPOINT = "midpoint"
    ON_SEGMENT = "on_segment"
    CONSTRAINT_SLICE = "constraint_slice"
    CANDIDATE_ANSWER = "candidate_answer"
    STATEMENT = "statement"


class GeometryDiagnosticCode(str, Enum):
    PARSE_ERROR = "parse_error"
    UNSUPPORTED_SYNTHETIC = "unsupported_synthetic"
    MISSING_COORDINATES = "missing_coordinates"
    CONTRADICTION = "contradiction"
    EXECUTION_ERROR = "execution_error"
    SUCCESS = "success"


@dataclass(frozen=True)
class GeometryDiagnostic:
    code: GeometryDiagnosticCode
    message: str
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GeometryEvidence:
    label: str
    expression: str
    simplified_expression: str
    exact_value: str | None
    note: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GeometryPoint:
    name: str
    x: sp.Expr
    y: sp.Expr


@dataclass(frozen=True)
class GeometryCheckResult:
    status: GeometryCheckStatus
    kind: GeometryCheckKind
    summary: str
    score: float
    exact: bool
    diagnostics: tuple[GeometryDiagnostic, ...] = field(default_factory=tuple)
    evidence: tuple[GeometryEvidence, ...] = field(default_factory=tuple)
    derived_facts: tuple[str, ...] = field(default_factory=tuple)
    contradiction_found: bool = False
    failure_type: FailureType | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status is GeometryCheckStatus.SUCCESS

    def to_branch_symbolic_evidence(
        self,
        *,
        check_name: str | None = None,
        supporting_step_ids: Sequence[str] = (),
    ) -> SymbolicEvidence:
        return SymbolicEvidence(
            evidence_id=f"geom_{abs(hash((self.kind.value, self.summary))) % 10**12:012d}",
            passed=self.passed,
            score=float(self.score),
            check_name=check_name or self.kind.value,
            summary=self.summary,
            supporting_step_ids=tuple(supporting_step_ids),
        )


def _parse_expr_safe(text: str) -> sp.Expr:
    return parse_expr((text or "").strip(), transformations=_TRANSFORMS, evaluate=True)


def _extract_points(constraints: Sequence[str]) -> dict[str, GeometryPoint]:
    points: dict[str, GeometryPoint] = {}
    for text in constraints:
        for m in _POINT_DEF_RE.finditer(text or ""):
            name = m.group(1)
            points[name] = GeometryPoint(
                name,
                _parse_expr_safe(m.group(2)),
                _parse_expr_safe(m.group(3)),
            )
    return points


def _vec(a: GeometryPoint, b: GeometryPoint) -> tuple[sp.Expr, sp.Expr]:
    return (sp.simplify(b.x - a.x), sp.simplify(b.y - a.y))


def _dot(u: tuple[sp.Expr, sp.Expr], v: tuple[sp.Expr, sp.Expr]) -> sp.Expr:
    return sp.simplify(u[0] * v[0] + u[1] * v[1])


def _det(u: tuple[sp.Expr, sp.Expr], v: tuple[sp.Expr, sp.Expr]) -> sp.Expr:
    return sp.simplify(u[0] * v[1] - u[1] * v[0])


def _dist2(a: GeometryPoint, b: GeometryPoint) -> sp.Expr:
    return sp.simplify((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def _zero_status(
    value: sp.Expr,
    success_summary: str,
    failure_summary: str,
    *,
    kind: GeometryCheckKind,
    expression: str,
) -> GeometryCheckResult:
    simplified = sp.simplify(value)
    if simplified == 0:
        return GeometryCheckResult(
            GeometryCheckStatus.SUCCESS,
            kind,
            success_summary,
            1.0,
            True,
            diagnostics=(GeometryDiagnostic(GeometryDiagnosticCode.SUCCESS, success_summary),),
            evidence=(GeometryEvidence(kind.value, expression, str(simplified), "0"),),
            derived_facts=(expression,),
        )
    if simplified.free_symbols:
        return GeometryCheckResult(
            GeometryCheckStatus.UNSUPPORTED,
            kind,
            "Geometric relation did not reduce to an exact coordinate/algebraic fact.",
            0.15,
            False,
            diagnostics=(GeometryDiagnostic(GeometryDiagnosticCode.UNSUPPORTED_SYNTHETIC, "Relation remained symbolic after coordinate reduction."),),
            evidence=(GeometryEvidence(kind.value, expression, str(simplified), None),),
            failure_type=FailureType.COVERAGE_GAP,
        )
    return GeometryCheckResult(
        GeometryCheckStatus.CONTRADICTION,
        kind,
        failure_summary,
        0.0,
        True,
        diagnostics=(GeometryDiagnostic(GeometryDiagnosticCode.CONTRADICTION, failure_summary, {"value": str(simplified)}),),
        evidence=(GeometryEvidence(kind.value, expression, str(simplified), str(simplified)),),
        contradiction_found=True,
        failure_type=FailureType.SYMBOLIC_MISMATCH,
    )


def validate_geometry_statement(
    statement: str,
    constraints: Sequence[str] = (),
) -> GeometryCheckResult:
    points = _extract_points(constraints)
    text = (statement or "").strip()
    if not text:
        return GeometryCheckResult(
            GeometryCheckStatus.MALFORMED_INPUT,
            GeometryCheckKind.STATEMENT,
            "Empty geometry statement.",
            0.05,
            False,
            diagnostics=(GeometryDiagnostic(GeometryDiagnosticCode.PARSE_ERROR, "Empty statement."),),
            failure_type=FailureType.SYMBOLIC_MISMATCH,
        )
    try:
        m = re.match(r"^collinear\(([^,]+),([^,]+),([^\)]+)\)$", text, re.IGNORECASE)
        if m:
            a, b, c = [points.get(x.strip()) for x in m.groups()]
            if None in {a, b, c}:
                return GeometryCheckResult(
                    GeometryCheckStatus.UNSUPPORTED,
                    GeometryCheckKind.COLLINEAR,
                    "Collinearity validation requires coordinate definitions for all referenced points.",
                    0.15,
                    False,
                    diagnostics=(GeometryDiagnostic(GeometryDiagnosticCode.MISSING_COORDINATES, "Missing coordinates for at least one point."),),
                    failure_type=FailureType.COVERAGE_GAP,
                )
            return _zero_status(
                _det(_vec(a, b), _vec(a, c)),
                f"Points {a.name}, {b.name}, {c.name} are collinear.",
                f"Points {a.name}, {b.name}, {c.name} are not collinear.",
                kind=GeometryCheckKind.COLLINEAR,
                expression=text,
            )

        m = re.match(r"^parallel\(([^,]+),([^,]+),([^,]+),([^\)]+)\)$", text, re.IGNORECASE)
        if m:
            a, b, c, d = [points.get(x.strip()) for x in m.groups()]
            if None in {a, b, c, d}:
                return GeometryCheckResult(
                    GeometryCheckStatus.UNSUPPORTED,
                    GeometryCheckKind.PARALLEL,
                    "Parallelism validation requires coordinate definitions for all referenced points.",
                    0.15,
                    False,
                    diagnostics=(GeometryDiagnostic(GeometryDiagnosticCode.MISSING_COORDINATES, "Missing coordinates for at least one point."),),
                    failure_type=FailureType.COVERAGE_GAP,
                )
            return _zero_status(
                _det(_vec(a, b), _vec(c, d)),
                f"Segments {a.name}{b.name} and {c.name}{d.name} are parallel.",
                f"Segments {a.name}{b.name} and {c.name}{d.name} are not parallel.",
                kind=GeometryCheckKind.PARALLEL,
                expression=text,
            )

        m = re.match(r"^perpendicular\(([^,]+),([^,]+),([^,]+),([^\)]+)\)$", text, re.IGNORECASE)
        if m:
            a, b, c, d = [points.get(x.strip()) for x in m.groups()]
            if None in {a, b, c, d}:
                return GeometryCheckResult(
                    GeometryCheckStatus.UNSUPPORTED,
                    GeometryCheckKind.PERPENDICULAR,
                    "Perpendicularity validation requires coordinate definitions for all referenced points.",
                    0.15,
                    False,
                    diagnostics=(GeometryDiagnostic(GeometryDiagnosticCode.MISSING_COORDINATES, "Missing coordinates for at least one point."),),
                    failure_type=FailureType.COVERAGE_GAP,
                )
            return _zero_status(
                _dot(_vec(a, b), _vec(c, d)),
                f"Segments {a.name}{b.name} and {c.name}{d.name} are perpendicular.",
                f"Segments {a.name}{b.name} and {c.name}{d.name} are not perpendicular.",
                kind=GeometryCheckKind.PERPENDICULAR,
                expression=text,
            )

        m = re.match(r"^equal_distance\(([^,]+),([^,]+),([^,]+),([^\)]+)\)$", text, re.IGNORECASE)
        if m:
            a, b, c, d = [points.get(x.strip()) for x in m.groups()]
            if None in {a, b, c, d}:
                return GeometryCheckResult(
                    GeometryCheckStatus.UNSUPPORTED,
                    GeometryCheckKind.DISTANCE_EQUALITY,
                    "Distance-equality validation requires coordinate definitions for all referenced points.",
                    0.15,
                    False,
                    diagnostics=(GeometryDiagnostic(GeometryDiagnosticCode.MISSING_COORDINATES, "Missing coordinates for at least one point."),),
                    failure_type=FailureType.COVERAGE_GAP,
                )
            return _zero_status(
                sp.simplify(_dist2(a, b) - _dist2(c, d)),
                f"Distances {a.name}{b.name} and {c.name}{d.name} are equal.",
                f"Distances {a.name}{b.name} and {c.name}{d.name} are not equal.",
                kind=GeometryCheckKind.DISTANCE_EQUALITY,
                expression=text,
            )

        m = re.match(r"^midpoint\(([^,]+),([^,]+),([^\)]+)\)$", text, re.IGNORECASE)
        if m:
            mpt, a, b = [points.get(x.strip()) for x in m.groups()]
            if None in {mpt, a, b}:
                return GeometryCheckResult(
                    GeometryCheckStatus.UNSUPPORTED,
                    GeometryCheckKind.MIDPOINT,
                    "Midpoint validation requires coordinate definitions for all referenced points.",
                    0.15,
                    False,
                    diagnostics=(GeometryDiagnostic(GeometryDiagnosticCode.MISSING_COORDINATES, "Missing coordinates for at least one point."),),
                    failure_type=FailureType.COVERAGE_GAP,
                )
            expr = sp.simplify((2 * mpt.x - a.x - b.x) ** 2 + (2 * mpt.y - a.y - b.y) ** 2)
            return _zero_status(
                expr,
                f"{mpt.name} is the midpoint of {a.name}{b.name}.",
                f"{mpt.name} is not the midpoint of {a.name}{b.name}.",
                kind=GeometryCheckKind.MIDPOINT,
                expression=text,
            )

        m = re.match(r"^on_segment\(([^,]+),([^,]+),([^\)]+)\)$", text, re.IGNORECASE)
        if m:
            p, a, b = [points.get(x.strip()) for x in m.groups()]
            if None in {p, a, b}:
                return GeometryCheckResult(
                    GeometryCheckStatus.UNSUPPORTED,
                    GeometryCheckKind.ON_SEGMENT,
                    "On-segment validation requires coordinate definitions for all referenced points.",
                    0.15,
                    False,
                    diagnostics=(GeometryDiagnostic(GeometryDiagnosticCode.MISSING_COORDINATES, "Missing coordinates for at least one point."),),
                    failure_type=FailureType.COVERAGE_GAP,
                )
            col = _det(_vec(a, b), _vec(a, p))
            if sp.simplify(col) != 0:
                return GeometryCheckResult(
                    GeometryCheckStatus.CONTRADICTION,
                    GeometryCheckKind.ON_SEGMENT,
                    f"{p.name} is not on segment {a.name}{b.name} because the points are not collinear.",
                    0.0,
                    True,
                    diagnostics=(GeometryDiagnostic(GeometryDiagnosticCode.CONTRADICTION, "Point is not collinear with segment endpoints."),),
                    contradiction_found=True,
                    failure_type=FailureType.SYMBOLIC_MISMATCH,
                )
            bounds = [
                sp.simplify((p.x - a.x) * (p.x - b.x)),
                sp.simplify((p.y - a.y) * (p.y - b.y)),
            ]
            if all(not item.free_symbols and float(item) <= 0 for item in bounds):
                return GeometryCheckResult(
                    GeometryCheckStatus.SUCCESS,
                    GeometryCheckKind.ON_SEGMENT,
                    f"{p.name} lies on segment {a.name}{b.name}.",
                    1.0,
                    True,
                    diagnostics=(GeometryDiagnostic(GeometryDiagnosticCode.SUCCESS, "Point lies on segment."),),
                    evidence=(GeometryEvidence("on_segment", text, str(bounds[0]), str(bounds[0])),),
                    derived_facts=(text,),
                )
            if any(item.free_symbols for item in bounds):
                return GeometryCheckResult(
                    GeometryCheckStatus.UNSUPPORTED,
                    GeometryCheckKind.ON_SEGMENT,
                    "On-segment bounds remained symbolic after coordinate reduction.",
                    0.15,
                    False,
                    diagnostics=(GeometryDiagnostic(GeometryDiagnosticCode.UNSUPPORTED_SYNTHETIC, "Bounds remained symbolic."),),
                    failure_type=FailureType.COVERAGE_GAP,
                )
            return GeometryCheckResult(
                GeometryCheckStatus.CONTRADICTION,
                GeometryCheckKind.ON_SEGMENT,
                f"{p.name} is collinear with {a.name}{b.name} but lies outside the segment bounds.",
                0.0,
                True,
                diagnostics=(GeometryDiagnostic(GeometryDiagnosticCode.CONTRADICTION, "Point lies outside segment bounds."),),
                contradiction_found=True,
                failure_type=FailureType.SYMBOLIC_MISMATCH,
            )

        return GeometryCheckResult(
            GeometryCheckStatus.UNSUPPORTED,
            GeometryCheckKind.STATEMENT,
            "Geometry validator supports only coordinate-style claims such as collinear(...), parallel(...), perpendicular(...), equal_distance(...), midpoint(...), and on_segment(...).",
            0.15,
            False,
            diagnostics=(GeometryDiagnostic(GeometryDiagnosticCode.UNSUPPORTED_SYNTHETIC, "Synthetic or unsupported geometry form."),),
            failure_type=FailureType.COVERAGE_GAP,
        )
    except ValueError as exc:
        return GeometryCheckResult(
            GeometryCheckStatus.MALFORMED_INPUT,
            GeometryCheckKind.STATEMENT,
            str(exc),
            0.05,
            False,
            diagnostics=(GeometryDiagnostic(GeometryDiagnosticCode.PARSE_ERROR, str(exc)),),
            failure_type=FailureType.SYMBOLIC_MISMATCH,
        )
    except Exception as exc:
        return GeometryCheckResult(
            GeometryCheckStatus.EXECUTION_FAILURE,
            GeometryCheckKind.STATEMENT,
            f"Geometry validation failed: {exc}",
            0.0,
            False,
            diagnostics=(GeometryDiagnostic(GeometryDiagnosticCode.EXECUTION_ERROR, str(exc)),),
            failure_type=FailureType.SYMBOLIC_MISMATCH,
        )


def validate_constraint_slice(constraints: Sequence[str]) -> GeometryCheckResult:
    results = [
        validate_geometry_statement(text, constraints=constraints)
        for text in constraints
        if any(
            tok in (text or "")
            for tok in (
                "collinear(",
                "parallel(",
                "perpendicular(",
                "equal_distance(",
                "midpoint(",
                "on_segment(",
            )
        )
    ]
    if not results:
        return GeometryCheckResult(
            GeometryCheckStatus.UNSUPPORTED,
            GeometryCheckKind.CONSTRAINT_SLICE,
            "Constraint slice contains no supported coordinate-style geometry statements.",
            0.15,
            False,
            diagnostics=(GeometryDiagnostic(GeometryDiagnosticCode.UNSUPPORTED_SYNTHETIC, "No supported geometry constraints found."),),
            failure_type=FailureType.COVERAGE_GAP,
        )
    if any(r.status is GeometryCheckStatus.CONTRADICTION for r in results):
        first = next(r for r in results if r.status is GeometryCheckStatus.CONTRADICTION)
        return GeometryCheckResult(
            GeometryCheckStatus.CONTRADICTION,
            GeometryCheckKind.CONSTRAINT_SLICE,
            first.summary,
            0.0,
            all(r.exact for r in results),
            diagnostics=tuple(d for r in results for d in r.diagnostics),
            evidence=tuple(e for r in results for e in r.evidence),
            contradiction_found=True,
            failure_type=FailureType.SYMBOLIC_MISMATCH,
        )
    successes = sum(r.status is GeometryCheckStatus.SUCCESS for r in results)
    return GeometryCheckResult(
        GeometryCheckStatus.SUCCESS if successes == len(results) else GeometryCheckStatus.UNSUPPORTED,
        GeometryCheckKind.CONSTRAINT_SLICE,
        f"Validated {successes}/{len(results)} supported geometry constraints.",
        1.0 if successes == len(results) else 0.4,
        all(r.exact for r in results),
        diagnostics=tuple(d for r in results for d in r.diagnostics),
        evidence=tuple(e for r in results for e in r.evidence),
        derived_facts=tuple(f for r in results for f in r.derived_facts),
    )


def validate_candidate_answer(
    candidate_answer: str,
    constraints: Sequence[str],
    answer_symbol: str | None,
) -> GeometryCheckResult:
    return GeometryCheckResult(
        GeometryCheckStatus.UNSUPPORTED,
        GeometryCheckKind.CANDIDATE_ANSWER,
        "Geometry candidate-answer validation is unsupported unless geometry has already been reduced to non-geometric algebraic/NT constraints upstream.",
        0.15,
        False,
        diagnostics=(GeometryDiagnostic(GeometryDiagnosticCode.UNSUPPORTED_SYNTHETIC, "Geometry does not validate free candidate answers directly."),),
        failure_type=FailureType.COVERAGE_GAP,
        metadata={"answer_symbol": answer_symbol, "candidate_answer": candidate_answer},
    )


__all__ = [
    "GeometryCheckStatus",
    "GeometryCheckKind",
    "GeometryDiagnosticCode",
    "GeometryDiagnostic",
    "GeometryEvidence",
    "GeometryPoint",
    "GeometryCheckResult",
    "validate_geometry_statement",
    "validate_constraint_slice",
    "validate_candidate_answer",
]
