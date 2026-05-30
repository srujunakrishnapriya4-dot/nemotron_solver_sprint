from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import math
import re

from kaggle_anti086.solvers.answer_normalizer import normalize_answer
from kaggle_anti086.solvers.base import BaseSolver
from kaggle_anti086.solvers.types import SolverCandidate, SolverResult, validate_candidate


NUMBER = r"[-+]?\d+(?:\.\d+)?"
PAIR_RE = re.compile(
    rf"({NUMBER})\s*(?:->|=|converts?\s+to|becomes|maps?\s+to)\s*({NUMBER})",
    re.IGNORECASE,
)
QUERY_RE = re.compile(
    rf"(?:query|input|convert|what\s+is|for)\s*[:#]?\s*({NUMBER})(?=[^0-9.\-+]*(?:\?|output|result|convert|$))",
    re.IGNORECASE,
)
FINAL_QUERY_RE = re.compile(rf"({NUMBER})\s*(?:->|=|converts?\s+to|output\s+is)\s*\?", re.IGNORECASE)


@dataclass(frozen=True)
class FitCandidate:
    subfamily: str
    a: float
    b: float
    precision: int
    confidence: float

    def predict(self, x: float) -> float:
        return self.a * x + self.b


def extract_numeric_pairs(prompt: str) -> list[tuple[float, float]]:
    return [(float(a), float(b)) for a, b in PAIR_RE.findall(prompt or "")]


def extract_query_value(prompt: str, examples: list[tuple[float, float]]) -> float | None:
    text = prompt or ""
    pair_spans = [match.span() for match in PAIR_RE.finditer(text)]
    for pattern in (FINAL_QUERY_RE, QUERY_RE):
        for match in reversed(list(pattern.finditer(text))):
            if _inside_any(match.span(1), pair_spans):
                continue
            return float(match.group(1))
    if examples:
        last_pair_end = max((match.end() for match in PAIR_RE.finditer(text)), default=0)
        tail = text[last_pair_end:]
        if "?" in tail:
            nums = re.findall(NUMBER, tail)
            if nums:
                return float(nums[-1])
    return None


def infer_decimal_places(values: list[str]) -> int:
    places = []
    for value in values:
        if "." in value:
            places.append(len(value.split(".", 1)[1].rstrip()))
        else:
            places.append(0)
    return max(places) if places else 0


def fit_unit_candidates(examples: list[tuple[float, float]]) -> list[FitCandidate]:
    if len(examples) < 2:
        return []
    precision = infer_decimal_places([_raw_number(v) for _, v in examples])
    candidates: list[FitCandidate] = []
    xs = [x for x, _ in examples]
    ys = [y for _, y in examples]
    if all(abs(x) > 1e-12 for x in xs):
        scale = sum(y / x for x, y in examples) / len(examples)
        candidates.append(FitCandidate(f"multiplicative_scale_round_{precision}", scale, 0.0, precision, 0.94))
    offsets = [y - x for x, y in examples]
    offset = sum(offsets) / len(offsets)
    candidates.append(FitCandidate(f"linear_offset_round_{precision}", 1.0, offset, precision, 0.92))
    if len(set(xs)) >= 2:
        a, b = _fit_line(examples)
        candidates.append(FitCandidate(f"linear_scale_offset_round_{precision}", a, b, precision, 0.9))
    verified = [candidate for candidate in candidates if _fits_all(candidate, examples)]
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
        examples = extract_numeric_pairs(prompt)
        query = extract_query_value(prompt, examples)
        if len(examples) < 2:
            return SolverResult(self.name, family, [], True, "insufficient_examples", {"example_count": len(examples)})
        if query is None:
            return SolverResult(self.name, family, [], True, "missing_query", {"example_count": len(examples)})
        fits = fit_unit_candidates(examples)
        if not fits:
            return SolverResult(self.name, family, [], True, "inconsistent_examples", {"example_count": len(examples)})
        predictions = [_format_prediction(candidate.predict(query), candidate.precision) for candidate in fits]
        if len(set(predictions)) > 1:
            return SolverResult(
                self.name,
                family,
                [],
                True,
                "ambiguous_fit_disagreement",
                {"predictions": predictions, "candidate_count": len(fits)},
            )
        best = sorted(fits, key=lambda item: (-item.confidence, item.subfamily))[0]
        answer = predictions[0]
        candidate = SolverCandidate(
            answer=normalize_answer(answer, expected_type="numeric").normalized,
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


def _fit_line(examples: list[tuple[float, float]]) -> tuple[float, float]:
    n = len(examples)
    sx = sum(x for x, _ in examples)
    sy = sum(y for _, y in examples)
    sxx = sum(x * x for x, _ in examples)
    sxy = sum(x * y for x, y in examples)
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return 0.0, 0.0
    a = (n * sxy - sx * sy) / denom
    b = (sy - a * sx) / n
    return a, b


def _fits_all(candidate: FitCandidate, examples: list[tuple[float, float]]) -> bool:
    tolerance = _tolerance(candidate.precision)
    return all(abs(float(_format_prediction(candidate.predict(x), candidate.precision)) - y) <= tolerance for x, y in examples)


def _tolerance(precision: int) -> float:
    if precision == 2:
        return 0.015
    if precision == 3:
        return 0.0015
    if precision > 0:
        return 1.5 * 10 ** (-precision)
    return 1e-6


def _format_prediction(value: float, precision: int) -> str:
    quant = Decimal(1).scaleb(-precision)
    decimal = Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP)
    if precision == 0:
        return str(decimal.quantize(Decimal(1)))
    return format(decimal, f".{precision}f").rstrip("0").rstrip(".")


def _raw_number(value: float) -> str:
    if math.isclose(value, round(value)):
        return str(int(round(value)))
    return repr(value)


def _dedupe_candidates(candidates: list[FitCandidate]) -> list[FitCandidate]:
    seen: set[tuple[int, int, int]] = set()
    result: list[FitCandidate] = []
    for candidate in candidates:
        key = (round(candidate.a, 10), round(candidate.b, 10), candidate.precision)
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result


def _inside_any(span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(parent_start <= start and end <= parent_end for parent_start, parent_end in spans)
