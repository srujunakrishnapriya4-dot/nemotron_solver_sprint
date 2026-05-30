from __future__ import annotations

from dataclasses import dataclass
import re

from kaggle_anti086.solvers.base import BaseSolver
from kaggle_anti086.solvers.types import SolverCandidate, SolverResult, validate_candidate


BIT_RE = r"[01]{4,16}"
PAIR_RE = re.compile(rf"\b({BIT_RE})\b\s*(?:->|=>|=|maps?\s+to)\s*\b({BIT_RE})\b", re.IGNORECASE)
QUERY_PATTERNS = (
    re.compile(rf"\b({BIT_RE})\b\s*(?:->|=>|=|maps?\s+to)\s*\?", re.IGNORECASE),
    re.compile(rf"\b(?:input|query|solve|target)\s*[:#]?\s*\b({BIT_RE})\b", re.IGNORECASE),
)


@dataclass(frozen=True)
class BitProblem:
    examples: list[tuple[str, str]]
    query: str | None
    width: int | None
    reason: str = ""


@dataclass(frozen=True)
class BitRule:
    operation: str
    subfamily: str
    complexity: int
    confidence: float
    risk: str
    params: dict

    def apply(self, bits: str) -> str:
        width = len(bits)
        value = int(bits, 2)
        mask = (1 << width) - 1
        if self.operation == "identity":
            return bits
        if self.operation == "not":
            return _format_bits((~value) & mask, width)
        if self.operation == "reverse":
            return bits[::-1]
        if self.operation == "swap_nibbles":
            return bits[4:] + bits[:4]
        if self.operation == "rotate_left":
            return _rol(bits, self.params["k"])
        if self.operation == "rotate_right":
            return _ror(bits, self.params["k"])
        if self.operation == "xor_mask":
            return _format_bits(value ^ self.params["mask_int"], width)
        if self.operation == "and_mask":
            return _format_bits(value & self.params["mask_int"], width)
        if self.operation == "or_mask":
            return _format_bits(value | self.params["mask_int"], width)
        if self.operation == "reverse_then_xor":
            rev = int(bits[::-1], 2)
            return _format_bits(rev ^ self.params["mask_int"], width)
        if self.operation == "rotate_then_xor":
            rotated = int(_rol(bits, self.params["k"]), 2)
            return _format_bits(rotated ^ self.params["mask_int"], width)
        if self.operation == "xor_then_rotate":
            xored = _format_bits(value ^ self.params["mask_int"], width)
            return _rol(xored, self.params["k"])
        if self.operation == "not_then_rotate":
            inverted = _format_bits((~value) & mask, width)
            return _rol(inverted, self.params["k"])
        if self.operation == "conditional_parity_flip":
            if bits.count("1") % 2:
                return bits[:-1] + ("0" if bits[-1] == "1" else "1")
            return bits
        if self.operation == "swap_positions":
            chars = list(bits)
            i, j = self.params["positions"]
            chars[i], chars[j] = chars[j], chars[i]
            return "".join(chars)
        if self.operation == "permutation":
            return _apply_permutation(bits, self.params["permutation"])
        if self.operation == "permutation_then_xor":
            permuted = int(_apply_permutation(bits, self.params["permutation"]), 2)
            return _format_bits(permuted ^ self.params["mask_int"], width)
        if self.operation == "xor_then_permutation":
            xored = _format_bits(value ^ self.params["mask_int"], width)
            return _apply_permutation(xored, self.params["permutation"])
        raise ValueError(f"unknown bit operation: {self.operation}")


class BitTransformSolver(BaseSolver):
    @property
    def name(self) -> str:
        return "bit_transform_solver"

    @property
    def supported_families(self) -> tuple[str, ...]:
        return ("bit_manipulation",)

    def solve(self, row: dict) -> SolverResult:
        family = str(row.get("family", "bit_manipulation"))
        if family != "bit_manipulation":
            return SolverResult(self.name, family, [], True, "unsupported_family", {})
        problem = parse_bit_problem(str(row.get("prompt", "")))
        if problem.reason:
            return SolverResult(self.name, family, [], True, problem.reason, {"example_count": len(problem.examples)})
        assert problem.query is not None and problem.width is not None
        rules = fit_bit_rules(problem.examples, problem.width)
        if not rules:
            return SolverResult(self.name, family, [], True, "no_matching_transform", {"example_count": len(problem.examples), "width": problem.width})
        ranked_rules = sorted(rules, key=lambda rule: (rule.complexity, -rule.confidence, rule.operation))
        best_complexity = ranked_rules[0].complexity
        decisive_rules = [rule for rule in ranked_rules if rule.complexity == best_complexity]
        predictions = [rule.apply(problem.query) for rule in decisive_rules]
        if len(set(predictions)) > 1:
            return SolverResult(
                self.name,
                family,
                [],
                True,
                "ambiguous_transform_disagreement",
                {"predictions": predictions, "rules": [_rule_metadata(rule, problem) for rule in decisive_rules[:20]]},
            )
        best = decisive_rules[0]
        candidate = SolverCandidate(
            answer=predictions[0],
            source=self.name,
            family=family,
            subfamily=best.subfamily,
            confidence=best.confidence,
            example_consistency=1.0,
            verified=True,
            risk=best.risk,
            metadata=_rule_metadata(best, problem),
        )
        validate_candidate(candidate)
        return SolverResult(self.name, family, [candidate], False, "", {"candidate_count": len(rules)})


def parse_bit_problem(prompt: str) -> BitProblem:
    text = prompt or ""
    examples = [(match.group(1), match.group(2)) for match in PAIR_RE.finditer(text)]
    if len(examples) < 2:
        return BitProblem(examples, None, None, "insufficient_examples")
    widths = {len(inp) for inp, _ in examples} | {len(out) for _, out in examples}
    if len(widths) != 1:
        return BitProblem(examples, None, None, "mixed_bit_widths")
    width = widths.pop()
    if width not in {4, 8, 16}:
        return BitProblem(examples, None, None, "unsupported_width")
    spans = [match.span() for match in PAIR_RE.finditer(text)]
    query = None
    for pattern in QUERY_PATTERNS:
        for match in reversed(list(pattern.finditer(text))):
            if _inside_any(match.span(1), spans):
                continue
            query = match.group(1)
            break
        if query:
            break
    if query is None:
        return BitProblem(examples, None, width, "missing_query")
    if len(query) != width:
        return BitProblem(examples, query, width, "query_width_mismatch")
    return BitProblem(examples, query, width)


def fit_bit_rules(examples: list[tuple[str, str]], width: int) -> list[BitRule]:
    rules: list[BitRule] = []
    for rule in _candidate_rules(examples, width):
        if all(rule.apply(inp) == out for inp, out in examples):
            rules.append(rule)
    return _dedupe_rules(rules)


def _candidate_rules(examples: list[tuple[str, str]], width: int) -> list[BitRule]:
    rules = [
        BitRule("identity", "identity", 0, 0.99, "low", {}),
        BitRule("not", "not", 1, 0.97, "low", {}),
        BitRule("reverse", "reverse", 1, 0.97, "low", {}),
        BitRule("conditional_parity_flip", "conditional_parity_flip", 4, 0.85, "medium", {}),
    ]
    if width == 8:
        rules.append(BitRule("swap_nibbles", "swap_nibbles", 1, 0.96, "low", {}))
    for k in range(1, width):
        rules.append(BitRule("rotate_left", f"rotate_left_{k}", 1, 0.96, "low", {"k": k}))
        rules.append(BitRule("rotate_right", f"rotate_right_{k}", 1, 0.96, "low", {"k": k}))
        mask = _consistent_mask(examples, lambda bits, kk=k: _rol(bits, kk))
        if mask is not None:
            rules.append(BitRule("rotate_then_xor", f"rotate_left_{k}_then_xor", 3, 0.91, "low", {"k": k, "mask": _format_bits(mask, width), "mask_int": mask}))
        mask = _consistent_mask(examples, lambda bits: bits)
        if mask is not None:
            rules.append(BitRule("xor_mask", "xor_mask", 2, 0.94, "low", {"mask": _format_bits(mask, width), "mask_int": mask}))
    xor_mask = _consistent_mask(examples, lambda bits: bits)
    if xor_mask is not None:
        rules.append(BitRule("xor_mask", "xor_mask", 2, 0.94, "low", {"mask": _format_bits(xor_mask, width), "mask_int": xor_mask}))
        for k in range(1, width):
            rules.append(BitRule("xor_then_rotate", f"xor_then_rotate_left_{k}", 3, 0.9, "low", {"k": k, "mask": _format_bits(xor_mask, width), "mask_int": xor_mask}))
    rev_mask = _consistent_mask(examples, lambda bits: bits[::-1])
    if rev_mask is not None:
        rules.append(BitRule("reverse_then_xor", "reverse_then_xor", 3, 0.91, "low", {"mask": _format_bits(rev_mask, width), "mask_int": rev_mask}))
    for op_name, combiner in (("and_mask", lambda inp, out: int(out, 2) if (int(inp, 2) & int(out, 2)) == int(out, 2) else None), ("or_mask", lambda inp, out: int(out, 2) if (int(inp, 2) | int(out, 2)) == int(out, 2) else None)):
        mask = _same_value([combiner(inp, out) for inp, out in examples])
        if mask is not None:
            rules.append(BitRule(op_name, op_name, 2, 0.9, "medium", {"mask": _format_bits(mask, width), "mask_int": mask}))
    for k in range(1, width):
        rules.append(BitRule("not_then_rotate", f"not_then_rotate_left_{k}", 3, 0.89, "low", {"k": k}))
    for i in range(width):
        for j in range(i + 1, width):
            rules.append(BitRule("swap_positions", f"swap_positions_{i}_{j}", 2, 0.9, "medium", {"positions": (i, j)}))
    rules.extend(_infer_permutation_rules(examples, width))
    return rules


def _infer_permutation_rules(examples: list[tuple[str, str]], width: int) -> list[BitRule]:
    if len(examples) < 3 or width not in {4, 8}:
        return []
    choices: list[list[tuple[int, int]]] = []
    for out_pos in range(width):
        position_choices: list[tuple[int, int]] = []
        for in_pos in range(width):
            for xor_bit in (0, 1):
                if all(((int(inp[in_pos]) ^ xor_bit) == int(out[out_pos])) for inp, out in examples):
                    position_choices.append((in_pos, xor_bit))
        if not position_choices:
            return []
        choices.append(position_choices)
    candidates: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def backtrack(out_pos: int, used: set[int], perm: list[int], xor_bits: list[int]) -> None:
        if len(candidates) >= 32:
            return
        if out_pos == width:
            candidates.append((tuple(perm), tuple(xor_bits)))
            return
        for in_pos, xor_bit in choices[out_pos]:
            if in_pos in used:
                continue
            used.add(in_pos)
            perm.append(in_pos)
            xor_bits.append(xor_bit)
            backtrack(out_pos + 1, used, perm, xor_bits)
            xor_bits.pop()
            perm.pop()
            used.remove(in_pos)

    backtrack(0, set(), [], [])
    rules: list[BitRule] = []
    identity = tuple(range(width))
    for perm, xor_bits in candidates:
        mask_int = int("".join(str(bit) for bit in xor_bits), 2)
        if perm == identity and mask_int == 0:
            continue
        if mask_int == 0:
            rules.append(BitRule("permutation", "bit_position_permutation", 5, 0.82, "medium", {"permutation": perm}))
        else:
            rules.append(
                BitRule(
                    "permutation_then_xor",
                    "bit_position_permutation_then_xor",
                    6,
                    0.8,
                    "medium",
                    {"permutation": perm, "mask": _format_bits(mask_int, width), "mask_int": mask_int},
                )
            )
            rules.append(
                BitRule(
                    "xor_then_permutation",
                    "xor_then_bit_position_permutation",
                    6,
                    0.8,
                    "medium",
                    {"permutation": perm, "mask": _format_bits(mask_int, width), "mask_int": mask_int},
                )
            )
    return rules


def _consistent_mask(examples: list[tuple[str, str]], transform) -> int | None:
    masks = [int(transform(inp), 2) ^ int(out, 2) for inp, out in examples]
    return _same_value(masks)


def _same_value(values: list[int | None]) -> int | None:
    if not values or any(value is None for value in values):
        return None
    first = values[0]
    return first if all(value == first for value in values) else None


def _rol(bits: str, k: int) -> str:
    k %= len(bits)
    return bits[k:] + bits[:k]


def _ror(bits: str, k: int) -> str:
    k %= len(bits)
    return bits[-k:] + bits[:-k]


def _format_bits(value: int, width: int) -> str:
    return format(value & ((1 << width) - 1), f"0{width}b")


def _rule_metadata(rule: BitRule, problem: BitProblem) -> dict:
    metadata = dict(rule.params)
    metadata.update(
        {
            "operation": rule.subfamily,
            "width": problem.width,
            "matched_examples": len(problem.examples),
            "example_count": len(problem.examples),
            "complexity": rule.complexity,
            "ambiguity_count": 0,
        }
    )
    return metadata


def _apply_permutation(bits: str, permutation: tuple[int, ...] | list[int]) -> str:
    return "".join(bits[index] for index in permutation)


def _dedupe_rules(rules: list[BitRule]) -> list[BitRule]:
    seen = set()
    result = []
    for rule in rules:
        key = (rule.operation, tuple(sorted(rule.params.items())))
        if key in seen:
            continue
        seen.add(key)
        result.append(rule)
    return result


def _inside_any(span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(parent_start <= start and end <= parent_end for parent_start, parent_end in spans)
