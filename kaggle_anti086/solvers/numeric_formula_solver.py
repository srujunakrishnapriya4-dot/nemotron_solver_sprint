from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from kaggle_anti086.solvers.base import BaseSolver
from kaggle_anti086.solvers.numeric_parsing import (
    RawNumericPair,
    extract_raw_numeric_pairs,
    extract_raw_query_value,
    format_decimal,
    infer_output_precision_from_raw_pairs,
    pair_spans,
)
from kaggle_anti086.solvers.types import SolverCandidate, SolverResult, validate_candidate


@dataclass(frozen=True)
class FormulaCandidate:
    subfamily: str
    coefficients: tuple[Decimal, ...]
    precision: int
    confidence: float
    max_abs_error: Decimal

    def predict(self, x: Decimal) -> Decimal:
        if self.subfamily == "gravity_distance":
            (g,) = self.coefficients
            return Decimal("0.5") * g * x * x
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

    def metadata(self) -> dict:
        return {
            "subfamily": self.subfamily,
            "coefficients": [str(value) for value in self.coefficients],
            "precision": self.precision,
            "max_abs_error": str(self.max_abs_error),
        }


def extract_numeric_pairs(prompt: str) -> list[tuple[float, float]]:
    return [(float(pair.x.value), float(pair.y.value)) for pair in extract_raw_numeric_pairs(prompt)]


def extract_query_value(prompt: str, examples) -> float | None:
    query = extract_raw_query_value(prompt, pair_spans(prompt))
    return None if query is None else float(query.value)


def fit_numeric_formula_candidates(examples: list[tuple[float, float]] | list[RawNumericPair], family_hint: str | None = None) -> list[FormulaCandidate]:
    pairs = _coerce_pairs(examples)
    if len(pairs) < 2:
        return []
    precision = infer_output_precision_from_raw_pairs(pairs)
    candidates: list[FormulaCandidate] = []
    if family_hint == "gravity_numeric":
        gravity = _fit_gravity(pairs, precision)
        return [gravity] if gravity else []
    for fitter in (_fit_linear_scale, _fit_linear_offset, _fit_quadratic_scale, _fit_quadratic_offset):
        candidate = fitter(pairs, precision)
        if candidate:
            candidates.append(candidate)
    return _dedupe(candidates)


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
        pairs = extract_raw_numeric_pairs(prompt)
        query = extract_raw_query_value(prompt, pair_spans(prompt))
        if len(pairs) < 2:
            return SolverResult(self.name, family, [], True, "insufficient_examples", {"example_count": len(pairs)})
        if query is None:
            return SolverResult(self.name, family, [], True, "missing_query", {"example_count": len(pairs)})
        fits = fit_numeric_formula_candidates(pairs, family_hint=family)
        if not fits:
            reason = "gravity_formula_not_supported" if family == "gravity_numeric" else "inconsistent_examples"
            return SolverResult(self.name, family, [], True, reason, {"example_count": len(pairs)})
        ranked = _rank_formula(fits)
        best = ranked[0]
        best_prediction = format_decimal(best.predict(query.value), best.precision)
        predictions = [format_decimal(candidate.predict(query.value), candidate.precision) for candidate in fits]

        # Day 9 hardening:
        # Some integer affine prompts also admit a weak rounded quadratic fit.
        # Do not abstain if the top-ranked candidate is an exact fit on examples.
        # Prefer exact affine/scale candidates over rounded-valid alternatives.
        if len(set(predictions)) > 1 and best.max_abs_error != 0:
            return SolverResult(
                self.name,
                family,
                [],
                True,
                "ambiguous_formula_disagreement",
                {
                    "predictions": predictions,
                    "best_prediction": best_prediction,
                    "best_candidate": best.metadata(),
                    "candidates": [candidate.metadata() for candidate in fits],
                },
            )

        metadata = {
            "query": query.raw,
            "candidate_count": len(fits),
            "candidate_predictions": predictions,
            "selected_prediction": best_prediction,
            "selected_candidate": best.metadata(),
            "candidates": [candidate.metadata() for candidate in fits],
        }
        candidate = SolverCandidate(
            answer=best_prediction,
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
        return SolverResult(self.name, family, [candidate], False, "", {"candidate_count": len(fits)})


def _coerce_pairs(examples: list[tuple[float, float]] | list[RawNumericPair]) -> list[RawNumericPair]:
    if not examples:
        return []
    if isinstance(examples[0], RawNumericPair):
        return list(examples)  # type: ignore[arg-type]
    from kaggle_anti086.solvers.numeric_parsing import RawNumeric

    pairs: list[RawNumericPair] = []
    for x, y in examples:  # type: ignore[assignment]
        pairs.append(RawNumericPair(RawNumeric(str(x), Decimal(str(x)), _places(str(x))), RawNumeric(str(y), Decimal(str(y)), _places(str(y)))))
    return pairs


def _fit_gravity(pairs: list[RawNumericPair], precision: int) -> FormulaCandidate | None:
    if any(pair.x.value == 0 for pair in pairs):
        return None
    gs = [Decimal(2) * pair.y.value / (pair.x.value * pair.x.value) for pair in pairs]
    g = sum(gs, Decimal(0)) / Decimal(len(gs))
    return _candidate_if_valid("gravity_distance", (g,), precision, Decimal("0.97"), pairs)


def _fit_linear_scale(pairs: list[RawNumericPair], precision: int) -> FormulaCandidate | None:
    if any(pair.x.value == 0 for pair in pairs):
        return None
    scales = [pair.y.value / pair.x.value for pair in pairs]
    a = sum(scales, Decimal(0)) / Decimal(len(scales))
    return _candidate_if_valid("linear_scale", (a,), precision, Decimal("0.94"), pairs)


def _fit_linear_offset(pairs: list[RawNumericPair], precision: int) -> FormulaCandidate | None:
    line = _fit_two_parameter([(pair.x.value, pair.y.value) for pair in pairs])
    if line is None:
        return None
    return _candidate_if_valid("linear_offset", line, precision, Decimal("0.92"), pairs)


def _fit_quadratic_scale(pairs: list[RawNumericPair], precision: int) -> FormulaCandidate | None:
    features = [pair.x.value * pair.x.value for pair in pairs]
    if any(feature == 0 for feature in features):
        return None
    scales = [pair.y.value / feature for feature, pair in zip(features, pairs)]
    a = sum(scales, Decimal(0)) / Decimal(len(scales))
    return _candidate_if_valid("quadratic_scale", (a,), precision, Decimal("0.9"), pairs)


def _fit_quadratic_offset(pairs: list[RawNumericPair], precision: int) -> FormulaCandidate | None:
    line = _fit_two_parameter([(pair.x.value * pair.x.value, pair.y.value) for pair in pairs])
    if line is None:
        return None
    return _candidate_if_valid("quadratic_offset", line, precision, Decimal("0.88"), pairs)


def _fit_two_parameter(points: list[tuple[Decimal, Decimal]]) -> tuple[Decimal, Decimal] | None:
    if len({x for x, _ in points}) < 2:
        return None
    n = Decimal(len(points))
    sx = sum((x for x, _ in points), Decimal(0))
    sy = sum((y for _, y in points), Decimal(0))
    sxx = sum((x * x for x, _ in points), Decimal(0))
    sxy = sum((x * y for x, y in points), Decimal(0))
    denom = n * sxx - sx * sx
    if denom == 0:
        return None
    a = (n * sxy - sx * sy) / denom
    b = (sy - a * sx) / n
    return a, b


def _candidate_if_valid(subfamily: str, coefficients: tuple[Decimal, ...], precision: int, confidence: Decimal, pairs: list[RawNumericPair]) -> FormulaCandidate | None:
    tolerance = _tolerance(precision)
    probe = FormulaCandidate(subfamily, coefficients, precision, float(confidence), Decimal(0))
    max_error = Decimal(0)
    for pair in pairs:
        predicted = probe.predict(pair.x.value)
        max_error = max(max_error, abs(predicted - pair.y.value))
        if format_decimal(predicted, precision) != format_decimal(pair.y.value, precision):
            return None
        if abs(predicted - pair.y.value) > tolerance:
            return None
    return FormulaCandidate(subfamily, coefficients, precision, float(confidence), max_error)


def _tolerance(precision: int) -> Decimal:
    if precision <= 0:
        return Decimal("0.5")
    return Decimal("0.5").scaleb(-precision) + Decimal("1e-18")


def _dedupe(candidates: list[FormulaCandidate]) -> list[FormulaCandidate]:
    result: list[FormulaCandidate] = []
    seen: set[tuple[str, tuple[str, ...], int]] = set()
    for candidate in candidates:
        key = (candidate.subfamily, tuple(str(value.normalize()) for value in candidate.coefficients), candidate.precision)
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
    # Prefer exact fits over rounded-valid approximate fits, then simpler formulas.
    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.max_abs_error,
            order.get(candidate.subfamily, 99),
            -candidate.confidence,
        ),
    )


def _places(text: str) -> int:
    return len(text.split(".", 1)[1]) if "." in text else 0
