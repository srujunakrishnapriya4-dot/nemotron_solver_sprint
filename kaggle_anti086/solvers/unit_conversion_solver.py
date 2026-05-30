from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import math

from kaggle_anti086.solvers.answer_normalizer import normalize_answer
from kaggle_anti086.solvers.base import BaseSolver
from kaggle_anti086.solvers.numeric_parsing import (
    RawNumeric,
    RawNumericPair,
    extract_raw_numeric_pairs,
    extract_raw_query_value,
    format_decimal,
    infer_output_precision_from_raw_pairs,
    pair_spans,
)
from kaggle_anti086.solvers.types import SolverCandidate, SolverResult, validate_candidate


MAX_COEFFICIENT_MAGNITUDE = Decimal("1e12")


@dataclass(frozen=True)
class FitCandidate:
    formula_name: str
    a: Decimal
    b: Decimal
    precision: int
    confidence: float
    max_abs_error: Decimal
    rounded_match_count: int
    example_count: int

    @property
    def subfamily(self) -> str:
        if self.formula_name == "scale":
            return f"multiplicative_scale_round_{self.precision}"
        if self.formula_name == "offset":
            return f"linear_offset_round_{self.precision}"
        if self.formula_name == "linear":
            return f"linear_scale_offset_round_{self.precision}"
        return f"inverse_scale_round_{self.precision}"

    def predict(self, x: Decimal) -> Decimal:
        return self.a * x + self.b

    def metadata(self) -> dict:
        return {
            "formula_name": self.formula_name,
            "a": str(self.a),
            "b": str(self.b),
            "precision": self.precision,
            "max_abs_error": str(self.max_abs_error),
            "rounded_match_count": self.rounded_match_count,
            "example_count": self.example_count,
        }


def extract_numeric_pairs(prompt: str) -> list[tuple[float, float]]:
    return [(float(pair.x.value), float(pair.y.value)) for pair in extract_raw_numeric_pairs(prompt)]


def extract_query_value(prompt: str, examples: list[tuple[float, float]]) -> float | None:
    query = extract_raw_query_value(prompt, pair_spans(prompt))
    return None if query is None else float(query.value)


def infer_decimal_places(values: list[str]) -> int:
    places = []
    for value in values:
        text = str(value)
        places.append(len(text.split(".", 1)[1]) if "." in text else 0)
    return max(places) if places else 0


def fit_unit_candidates(examples: list[tuple[float, float]] | list[RawNumericPair]) -> list[FitCandidate]:
    pairs = _coerce_pairs(examples)
    if len(pairs) < 2:
        return []
    precision = infer_output_precision_from_raw_pairs(pairs)
    candidates: list[tuple[str, Decimal, Decimal, float]] = []
    xs = [pair.x.value for pair in pairs]
    ys = [pair.y.value for pair in pairs]
    if all(x != 0 for x in xs):
        scale = sum((y / x for x, y in zip(xs, ys)), Decimal(0)) / Decimal(len(pairs))
        candidates.append(("scale", scale, Decimal(0), 0.94))
        if scale != 0:
            candidates.append(("inverse", scale, Decimal(0), 0.88))
    offsets = [y - x for x, y in zip(xs, ys)]
    offset = sum(offsets, Decimal(0)) / Decimal(len(offsets))
    candidates.append(("offset", Decimal(1), offset, 0.92))
    line = _fit_line(pairs)
    if line is not None:
        a, b = line
        candidates.append(("linear", a, b, 0.9))

    verified: list[FitCandidate] = []
    for formula_name, a, b, confidence in candidates:
        if _absurd(a) or _absurd(b) or not a.is_finite() or not b.is_finite():
            continue
        candidate = _verify_candidate(formula_name, a, b, precision, confidence, pairs)
        if candidate:
            verified.append(candidate)
    return _dedupe_candidates(verified)


class UnitConversionSolver(BaseSolver):
    @property
    def name(self) -> str:
        return "unit_conversion_solver"

    @property
    def supported_families(self) -> tuple[str, ...]:
        return ("unit_conversion",)

    def solve(self, row: dict) -> SolverResult:
        family = str(row.get("family", "unit_conversion"))
        if family != "unit_conversion":
            return SolverResult(self.name, family, [], True, "unsupported_family", {})
        prompt = str(row.get("prompt", ""))
        pairs = extract_raw_numeric_pairs(prompt)
        query = extract_raw_query_value(prompt, pair_spans(prompt))
        if len(pairs) < 2:
            return SolverResult(self.name, family, [], True, "insufficient_examples", {"example_count": len(pairs)})
        if query is None:
            return SolverResult(self.name, family, [], True, "missing_query", {"example_count": len(pairs)})
        fits = fit_unit_candidates(pairs)
        if not fits:
            return SolverResult(self.name, family, [], True, "inconsistent_examples", {"example_count": len(pairs)})
        predictions = [format_decimal(candidate.predict(query.value), candidate.precision) for candidate in fits]
        if len(set(predictions)) > 1:
            return SolverResult(
                self.name,
                family,
                [],
                True,
                "ambiguous_fit_disagreement",
                {"predictions": predictions, "candidates": [candidate.metadata() for candidate in fits]},
            )
        best = sorted(fits, key=_candidate_rank)[0]
        answer = predictions[0]
        metadata = best.metadata() | {"query": query.raw, "candidate_count": len(fits)}
        candidate = SolverCandidate(
            answer=answer,
            source=self.name,
            family=family,
            subfamily=best.subfamily,
            confidence=best.confidence,
            example_consistency=1.0,
            verified=True,
            risk="low",
            metadata=metadata,
        )
        validate_candidate(candidate)
        return SolverResult(self.name, family, [candidate], False, "", {"candidate_count": len(fits), "candidates": metadata})


def _coerce_pairs(examples: list[tuple[float, float]] | list[RawNumericPair]) -> list[RawNumericPair]:
    if not examples:
        return []
    if isinstance(examples[0], RawNumericPair):
        return list(examples)  # type: ignore[arg-type]
    pairs: list[RawNumericPair] = []
    for x, y in examples:  # type: ignore[assignment]
        pairs.append(RawNumericPair(RawNumeric(str(x), Decimal(str(x)), infer_decimal_places([str(x)])), RawNumeric(str(y), Decimal(str(y)), infer_decimal_places([str(y)]))))
    return pairs


def _fit_line(pairs: list[RawNumericPair]) -> tuple[Decimal, Decimal] | None:
    if len(pairs) < 3:
        return None
    n = Decimal(len(pairs))
    sx = sum((pair.x.value for pair in pairs), Decimal(0))
    sy = sum((pair.y.value for pair in pairs), Decimal(0))
    sxx = sum((pair.x.value * pair.x.value for pair in pairs), Decimal(0))
    sxy = sum((pair.x.value * pair.y.value for pair in pairs), Decimal(0))
    denom = n * sxx - sx * sx
    if denom == 0:
        return None
    a = (n * sxy - sx * sy) / denom
    b = (sy - a * sx) / n
    return a, b


def _verify_candidate(formula_name: str, a: Decimal, b: Decimal, precision: int, confidence: float, pairs: list[RawNumericPair]) -> FitCandidate | None:
    tolerance = _tolerance(precision)
    rounded_matches = 0
    max_error = Decimal(0)
    for pair in pairs:
        predicted = a * pair.x.value + b
        rounded_pred = format_decimal(predicted, precision)
        rounded_gold = format_decimal(pair.y.value, precision)
        error = abs(predicted - pair.y.value)
        max_error = max(max_error, error)
        if rounded_pred == rounded_gold and error <= tolerance:
            rounded_matches += 1
    if rounded_matches != len(pairs):
        return None
    return FitCandidate(formula_name, a, b, precision, confidence, max_error, rounded_matches, len(pairs))


def _tolerance(precision: int) -> Decimal:
    if precision <= 0:
        return Decimal("0.5")
    return Decimal("0.5").scaleb(-precision) + Decimal("1e-18")


def _absurd(value: Decimal) -> bool:
    return abs(value) > MAX_COEFFICIENT_MAGNITUDE or math.isnan(float(value)) or math.isinf(float(value))


def _candidate_rank(candidate: FitCandidate) -> tuple[int, str]:
    order = {"scale": 0, "offset": 1, "linear": 2, "inverse": 3}
    return (order.get(candidate.formula_name, 99), candidate.formula_name)


def _dedupe_candidates(candidates: list[FitCandidate]) -> list[FitCandidate]:
    seen: set[tuple[str, str, str, int]] = set()
    result: list[FitCandidate] = []
    for candidate in candidates:
        key = (candidate.formula_name, str(candidate.a.normalize()), str(candidate.b.normalize()), candidate.precision)
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result
