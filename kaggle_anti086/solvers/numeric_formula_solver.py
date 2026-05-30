from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import re

from kaggle_anti086.solvers.answer_normalizer import normalize_answer
from kaggle_anti086.solvers.base import BaseSolver
from kaggle_anti086.solvers.types import SolverCandidate, SolverResult, validate_candidate
from kaggle_anti086.solvers.unit_conversion_solver import (
    NUMBER,
    extract_numeric_pairs,
    extract_query_value,
    infer_decimal_places,
)


@dataclass(frozen=True)
class FormulaCandidate:
    subfamily: str
    coefficients: tuple[float, ...]
    precision: int
    confidence: float

    def predict(self, x: float) -> float:
        if self.subfamily == "gravity_distance":
            (g,) = self.coefficients
            return 0.5 * g * x * x
        if self.subfamily == "linear_scale":
            (a,) = self.coefficients
            return a * x
        if self.subfamily == "linear_offset":
            a, b = self.coefficients
            return a * x + b
        if self.subfamily == "quadratic_scale":
            (a,) = self.coefficients
            return a * x * x
        if self.subfamily == "quadratic_offset":
            a, b = self.coefficients
            return a * x * x + b
        raise ValueError(f"unknown formula: {self.subfamily}")


def fit_numeric_formula_candidates(examples: list[tuple[float, float]], family_hint: str | None = None) -> list[FormulaCandidate]:
    if len(examples) < 2:
        return []
    precision = infer_decimal_places([str(y) for _, y in examples])
    candidates: list[FormulaCandidate] = []
    if family_hint == "gravity_numeric":
        gravity = _fit_gravity(examples, precision)
        if gravity:
            candidates.append(gravity)
    linear_scale = _fit_scale(examples, precision, "linear_scale", lambda x: x)
    if linear_scale:
        candidates.append(linear_scale)
    linear_offset = _fit_two_parameter(examples, precision, "linear_offset", lambda x: x)
    if linear_offset:
        candidates.append(linear_offset)
    quadratic_scale = _fit_scale(examples, precision, "quadratic_scale", lambda x: x * x)
    if quadratic_scale:
        candidates.append(quadratic_scale)
    quadratic_offset = _fit_two_parameter(examples, precision, "quadratic_offset", lambda x: x * x)
    if quadratic_offset:
        candidates.append(quadratic_offset)
    return _dedupe([candidate for candidate in candidates if _fits_all(candidate, examples)])


class NumericFormulaSolver(BaseSolver):
    @property
    def name(self) -> str:
        return "numeric_formula_solver"

    @property
    def supported_families(self) -> tuple[str, ...]:
        return ("numeric_formula", "gravity_numeric")

    def solve(self, row: dict) -> SolverResult:
        family = str(row.get("family", "numeric_formula"))
        if family not in self.supported_families:
            return SolverResult(self.name, family, [], True, "unsupported_family", {})
        prompt = str(row.get("prompt", ""))
        examples = extract_numeric_pairs(prompt)
        query = extract_query_value(prompt, examples)
        if len(examples) < 2:
            return SolverResult(self.name, family, [], True, "insufficient_examples", {"example_count": len(examples)})
        if query is None:
            return SolverResult(self.name, family, [], True, "missing_query", {"example_count": len(examples)})
        fits = fit_numeric_formula_candidates(examples, family_hint=family)
        if not fits:
            return SolverResult(self.name, family, [], True, "inconsistent_examples", {"example_count": len(examples)})
        predictions = [_format_prediction(candidate.predict(query), candidate.precision) for candidate in fits]
        if len(set(predictions)) > 1:
            return SolverResult(
                self.name,
                family,
                [],
                True,
                "ambiguous_formula_disagreement",
                {"predictions": predictions, "candidate_count": len(fits)},
            )
        best = _rank_formula(fits)[0]
        candidate = SolverCandidate(
            answer=normalize_answer(predictions[0], expected_type="numeric").normalized,
            source=self.name,
            family=family,
            subfamily=best.subfamily,
            confidence=best.confidence,
            example_consistency=1.0,
            verified=True,
            risk="low",
            metadata={"query": query, "example_count": len(examples), "candidate_count": len(fits)},
        )
        validate_candidate(candidate)
        return SolverResult(self.name, family, [candidate], False, "", {"candidate_count": len(fits)})


def _fit_gravity(examples: list[tuple[float, float]], precision: int) -> FormulaCandidate | None:
    if any(abs(t) < 1e-12 for t, _ in examples):
        return None
    gs = [2.0 * d / (t * t) for t, d in examples]
    g = sum(gs) / len(gs)
    candidate = FormulaCandidate("gravity_distance", (g,), precision, 0.97)
    return candidate if _fits_all(candidate, examples) else None


def _fit_scale(examples: list[tuple[float, float]], precision: int, subfamily: str, feature) -> FormulaCandidate | None:
    values = [feature(x) for x, _ in examples]
    if any(abs(value) < 1e-12 for value in values):
        return None
    a = sum(y / value for value, (_, y) in zip(values, examples)) / len(examples)
    confidence = 0.94 if subfamily == "linear_scale" else 0.9
    candidate = FormulaCandidate(subfamily, (a,), precision, confidence)
    return candidate if _fits_all(candidate, examples) else None


def _fit_two_parameter(examples: list[tuple[float, float]], precision: int, subfamily: str, feature) -> FormulaCandidate | None:
    values = [(feature(x), y) for x, y in examples]
    if len({round(v, 12) for v, _ in values}) < 2:
        return None
    n = len(values)
    sx = sum(v for v, _ in values)
    sy = sum(y for _, y in values)
    sxx = sum(v * v for v, _ in values)
    sxy = sum(v * y for v, y in values)
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return None
    a = (n * sxy - sx * sy) / denom
    b = (sy - a * sx) / n
    confidence = 0.92 if subfamily == "linear_offset" else 0.88
    candidate = FormulaCandidate(subfamily, (a, b), precision, confidence)
    return candidate if _fits_all(candidate, examples) else None


def _fits_all(candidate: FormulaCandidate, examples: list[tuple[float, float]]) -> bool:
    tolerance = _tolerance(candidate.precision)
    return all(abs(float(_format_prediction(candidate.predict(x), candidate.precision)) - y) <= tolerance for x, y in examples)


def _tolerance(precision: int) -> float:
    if precision <= 0:
        return 1e-6
    return 1.5 * 10 ** (-precision)


def _format_prediction(value: float, precision: int) -> str:
    quant = Decimal(1).scaleb(-precision)
    decimal = Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP)
    if precision == 0:
        return str(decimal.quantize(Decimal(1)))
    return format(decimal, f".{precision}f").rstrip("0").rstrip(".")


def _dedupe(candidates: list[FormulaCandidate]) -> list[FormulaCandidate]:
    result: list[FormulaCandidate] = []
    seen: set[tuple[str, tuple[int, ...], int]] = set()
    for candidate in candidates:
        key = (candidate.subfamily, tuple(round(v, 10) for v in candidate.coefficients), candidate.precision)
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result


def _rank_formula(candidates: list[FormulaCandidate]) -> list[FormulaCandidate]:
    order = {
        "gravity_distance": 0,
        "linear_scale": 1,
        "linear_offset": 2,
        "quadratic_scale": 3,
        "quadratic_offset": 4,
    }
    return sorted(candidates, key=lambda candidate: (order.get(candidate.subfamily, 99), -candidate.confidence))
