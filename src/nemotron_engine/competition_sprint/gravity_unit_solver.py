from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
import re

from .hard_family_router import RuleHypothesis


NUM = re.compile(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", re.I)


def solve_gravity_unit_problem(examples: tuple[tuple[str, str], ...], target_input: str) -> RuleHypothesis:
    pairs = []
    for x, y in examples:
        xs = NUM.findall(x)
        ys = NUM.findall(y)
        if not xs or not ys:
            return _reject("non_numeric")
        pairs.append((Decimal(xs[-1]), Decimal(ys[-1]), _precision(ys[-1])))
    target_nums = NUM.findall(target_input)
    if not target_nums:
        return _reject("non_numeric_target")
    precision = max(p for _, _, p in pairs)
    candidates = []
    if all(x != 0 for x, _, _ in pairs):
        ratio = pairs[0][1] / pairs[0][0]
        if all(_round(x * ratio, p) == y for x, y, p in pairs):
            candidates.append(("ratio", Decimal(target_nums[-1]) * ratio))
    if len({x for x, _, _ in pairs}) >= 2:
        x1, y1, _ = pairs[0]
        x2, y2, _ = pairs[1]
        if x1 != x2:
            a = (y2 - y1) / (x2 - x1)
            b = y1 - a * x1
            if all(_round(a * x + b, p) == y for x, y, p in pairs):
                candidates.append(("affine", a * Decimal(target_nums[-1]) + b))
    if not candidates:
        return _reject("no_rounding_consistent_model")
    outputs = {_format_decimal(_round(value, precision), precision) for _, value in candidates}
    if len(outputs) != 1:
        return RuleHypothesis("gravity_unit_solver", "disagreement", None, False, "ratio_affine_disagreement", {"outputs": sorted(outputs)})
    return RuleHypothesis("gravity_unit_solver", candidates[0][0], outputs.pop(), True, "verified_unique", {"precision": precision, "candidate_count": len(candidates)})


def _round(value: Decimal, precision: int) -> Decimal:
    return value.quantize(Decimal("1") if precision == 0 else Decimal("1").scaleb(-precision), rounding=ROUND_HALF_UP)


def _precision(text: str) -> int:
    return len(text.split(".", 1)[1]) if "." in text else 0


def _format_decimal(value: Decimal, precision: int) -> str:
    return f"{value:.{precision}f}" if precision else str(int(value))


def _reject(reason: str) -> RuleHypothesis:
    return RuleHypothesis("gravity_unit_solver", "reject", None, False, reason, {})
