from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations, combinations_with_replacement
from typing import Callable

from nemotron_engine.core.schemas import stable_hash

from .hard_family_router import RuleHypothesis


@dataclass(frozen=True)
class BitExpr:
    expr_id: str
    values: tuple[int, ...]
    target_value: int
    depth: int
    operations: tuple[str, ...]


def solve_bit_composition_problem(examples: tuple[tuple[str, str], ...], target_input: str, *, budget: int = 40000) -> RuleHypothesis:
    parsed = _parse_single_stream(examples, target_input)
    if isinstance(parsed, RuleHypothesis):
        return parsed
    xs, ys, target, width = parsed
    if len(xs) < 2:
        return RuleHypothesis("bit_composition_solver", "reject", None, False, "underdetermined_bit_rule", {"example_count": len(xs)})
    mask = (1 << width) - 1
    engine = _BitSearch(xs, ys, target, width, mask, budget)
    return engine.run()


class _BitSearch:
    def __init__(self, xs: tuple[int, ...], ys: tuple[int, ...], target: int, width: int, mask: int, budget: int) -> None:
        self.xs = xs
        self.ys = ys
        self.target = target
        self.width = width
        self.mask = mask
        self.budget = budget
        self.candidate_count = 0
        self.pruned_duplicate_count = 0
        self.pruned_failed_count = 0
        self.seen_vectors: set[tuple[int, ...]] = set()
        self.verified: list[BitExpr] = []
        self.depth_counts: Counter[str] = Counter()

    def run(self) -> RuleHypothesis:
        base = self._base_expressions()
        level0 = self._register_many(base)
        level1 = self._register_many(self._unary_and_mask_expressions(level0))
        if not self.verified:
            linear = self._affine_gf2_synthesis()
            if linear is not None:
                return linear
            bitwise = self._bit_position_synthesis()
            if bitwise is not None:
                return bitwise
        self._register_many(self._binary_expressions(level0 + level1, max_pool=220, depth=2))
        if not self.verified and self.candidate_count < self.budget:
            level2_seed = self._select_pool(max_pool=420)
            level2 = self._register_many(self._unary_compositions(level2_seed, depth=2))
            self._register_many(self._binary_expressions(level2_seed + level2, max_pool=300, depth=3))
        return self._result()

    def _result(self) -> RuleHypothesis:
        metadata = {
            "candidate_count": self.candidate_count,
            "tested": self.candidate_count,
            "pruned_duplicate_count": self.pruned_duplicate_count,
            "pruned_failed_count": self.pruned_failed_count,
            "depth_solved_distribution": dict(self.depth_counts),
            "output_vector_hashes": len(self.seen_vectors),
        }
        if self.candidate_count >= self.budget and not self.verified:
            mined = self._truth_table_mining_synthesis()
            if mined is not None:
                return mined
            return RuleHypothesis("bit_composition_solver", "none", None, False, "bit_candidate_budget_exhausted", metadata)
        if not self.verified:
            mined = self._truth_table_mining_synthesis()
            if mined is not None:
                return mined
            return RuleHypothesis("bit_composition_solver", "none", None, False, "no_verified_rule", metadata)
        outputs = {expr.target_value & self.mask for expr in self.verified}
        metadata["verified_count"] = len(self.verified)
        metadata["ambiguity_count"] = max(0, len(self.verified) - 1)
        metadata["verified_rules"] = [expr.expr_id for expr in self.verified[:25]]
        if len(outputs) != 1:
            metadata["target_outputs"] = sorted(format(value, f"0{self.width}b") for value in outputs)
            return RuleHypothesis("bit_composition_solver", "disagreement", None, False, "ambiguous_fits", metadata)
        value = outputs.pop()
        best = sorted(self.verified, key=lambda expr: (expr.depth, len(expr.operations), expr.expr_id))[0]
        metadata["target_prediction"] = format(value, f"0{self.width}b")
        metadata["expr_id"] = best.expr_id
        return RuleHypothesis("bit_composition_solver", best.expr_id, format(value, f"0{self.width}b"), True, "verified_unique", metadata)

    def _register_many(self, candidates) -> list[BitExpr]:
        kept: list[BitExpr] = []
        for candidate in candidates:
            if self.candidate_count >= self.budget:
                break
            self.candidate_count += 1
            vector = candidate.values
            if vector in self.seen_vectors:
                self.pruned_duplicate_count += 1
                continue
            self.seen_vectors.add(vector)
            kept.append(candidate)
            if vector == self.ys:
                self.verified.append(candidate)
                self.depth_counts[str(candidate.depth)] += 1
            else:
                self.pruned_failed_count += 1
        return kept

    def _base_expressions(self) -> list[BitExpr]:
        return [
            self._expr("x", lambda x: x, 0, ("identity",)),
            self._expr("const_0", lambda _x: 0, 0, ("const",)),
            self._expr("const_all", lambda _x: self.mask, 0, ("const",)),
        ]

    def _unary_and_mask_expressions(self, seeds: list[BitExpr]) -> list[BitExpr]:
        out: list[BitExpr] = []
        unary = self._unary_ops()
        for seed in seeds:
            for name, fn in unary:
                out.append(self._from_seed(f"{name}({seed.expr_id})", seed, fn, seed.depth + 1, name))
        # High-yield affine/mask primitives over the original input. These replace
        # thousands of blind pairwise candidates with canonical output vectors.
        for const in range(self.mask + 1):
            out.append(self._expr(f"xor_{const:0{self.width}b}", lambda x, const=const: x ^ const, 1, ("xor_mask",)))
            out.append(self._expr(f"and_{const:0{self.width}b}", lambda x, const=const: x & const, 1, ("and_mask",)))
            out.append(self._expr(f"or_{const:0{self.width}b}", lambda x, const=const: x | const, 1, ("or_mask",)))
            out.append(self._expr(f"add_{const:0{self.width}b}", lambda x, const=const: (x + const), 1, ("add_const",)))
            out.append(self._expr(f"sub_{const:0{self.width}b}", lambda x, const=const: (x - const), 1, ("sub_const",)))
        return out

    def _unary_compositions(self, seeds: list[BitExpr], *, depth: int) -> list[BitExpr]:
        out: list[BitExpr] = []
        for seed in seeds:
            for name, fn in self._unary_ops():
                out.append(self._from_seed(f"{name}({seed.expr_id})", seed, fn, depth, name))
        return out

    def _binary_expressions(self, pool: list[BitExpr], *, max_pool: int, depth: int) -> list[BitExpr]:
        ordered = sorted(pool, key=lambda expr: (expr.depth, expr.expr_id))[:max_pool]
        out: list[BitExpr] = []
        ops = (
            ("xor", lambda a, b: a ^ b),
            ("and", lambda a, b: a & b),
            ("or", lambda a, b: a | b),
        )
        for left, right in combinations_with_replacement(ordered, 2):
            for name, fn in ops:
                values = tuple(fn(a, b) & self.mask for a, b in zip(left.values, right.values))
                target = fn(left.target_value, right.target_value) & self.mask
                out.append(BitExpr(f"{name}({left.expr_id},{right.expr_id})", values, target, depth, left.operations + right.operations + (name,)))
                if len(out) + self.candidate_count >= self.budget:
                    return out
        return out

    def _select_pool(self, *, max_pool: int) -> list[BitExpr]:
        # Prefer shallow expressions and vectors that are close to the desired
        # output over arbitrary insertion order.
        candidates = [
            BitExpr("x", tuple(self.xs), self.target, 0, ("identity",)),
        ]
        for vector in self.seen_vectors:
            distance = sum((a ^ b).bit_count() for a, b in zip(vector, self.ys))
            candidates.append(BitExpr(f"vec_{stable_hash(vector)[:10]}", vector, 0, 2, ("cached", str(distance))))
        return sorted(candidates, key=lambda expr: (int(expr.operations[-1]) if expr.operations[-1].isdigit() else 999, expr.expr_id))[:max_pool]

    def _unary_ops(self) -> tuple[tuple[str, Callable[[int], int]], ...]:
        ops: list[tuple[str, Callable[[int], int]]] = [
            ("not", lambda x: ~x),
            ("rev", self._reverse_bits),
        ]
        for k in range(1, self.width):
            ops.append((f"shl{k}", lambda x, k=k: x << k))
            ops.append((f"shr{k}", lambda x, k=k: x >> k))
            ops.append((f"rol{k}", lambda x, k=k: self._rol(x, k)))
            ops.append((f"ror{k}", lambda x, k=k: self._ror(x, k)))
        return tuple(ops)

    def _bit_position_synthesis(self) -> RuleHypothesis | None:
        candidate_count = 0
        ambiguity_count = 0
        output_value = 0
        rule_ids: list[str] = []
        for out_index in range(self.width):
            target_shift = self.width - 1 - out_index
            desired = tuple((y >> target_shift) & 1 for y in self.ys)
            matches: list[tuple[str, int]] = []
            for rule_id, values, target_bit in self._boolean_bit_candidates():
                candidate_count += 1
                if values == desired:
                    matches.append((f"out{out_index}:{rule_id}", target_bit))
            if not matches:
                return None
            target_bits = {bit for _, bit in matches}
            ambiguity_count += max(0, len(matches) - 1)
            if len(target_bits) != 1:
                return RuleHypothesis(
                    "bit_composition_solver",
                    "bit_position_disagreement",
                    None,
                    False,
                    "ambiguous_fits",
                    {
                        "candidate_count": self.candidate_count + candidate_count,
                        "bit_position": out_index,
                        "ambiguity_count": ambiguity_count,
                        "target_outputs": sorted(target_bits),
                    },
                )
            bit = target_bits.pop()
            output_value = (output_value << 1) | bit
            rule_ids.extend(rule_id for rule_id, _ in matches[:3])
        return RuleHypothesis(
            "bit_composition_solver",
            "bit_position_boolean_synthesis",
            format(output_value & self.mask, f"0{self.width}b"),
            True,
            "verified_unique",
            {
                "candidate_count": self.candidate_count + candidate_count,
                "tested": self.candidate_count + candidate_count,
                "verified_count": self.width,
                "ambiguity_count": ambiguity_count,
                "depth_solved_distribution": {"bit_position": self.width},
                "expr_id": "bit_position_boolean_synthesis",
                "verified_rules": rule_ids[:25],
                "pruned_duplicate_count": self.pruned_duplicate_count,
                "pruned_failed_count": self.pruned_failed_count,
                "target_prediction": format(output_value & self.mask, f"0{self.width}b"),
            },
        )

    def _affine_gf2_synthesis(self) -> RuleHypothesis | None:
        if len(self.xs) < min(32, 1 << self.width):
            return None
        output_value = 0
        ambiguity_count = 0
        for out_index in range(self.width):
            shift = self.width - 1 - out_index
            rhs = tuple((y >> shift) & 1 for y in self.ys)
            target_bit = self._solve_affine_target_bit(rhs)
            if target_bit is None:
                return None
            if target_bit == "ambiguous":
                return RuleHypothesis(
                    "bit_composition_solver",
                    "affine_gf2_disagreement",
                    None,
                    False,
                    "ambiguous_fits",
                    {"bit_position": out_index, "candidate_count": self.candidate_count, "ambiguity_count": ambiguity_count + 1},
                )
            output_value = (output_value << 1) | int(target_bit)
        return RuleHypothesis(
            "bit_composition_solver",
            "affine_gf2_bit_synthesis",
            format(output_value & self.mask, f"0{self.width}b"),
            True,
            "verified_unique",
            {
                "candidate_count": self.candidate_count,
                "tested": self.candidate_count,
                "verified_count": self.width,
                "ambiguity_count": ambiguity_count,
                "depth_solved_distribution": {"affine_gf2": self.width},
                "expr_id": "affine_gf2_bit_synthesis",
                "pruned_duplicate_count": self.pruned_duplicate_count,
                "pruned_failed_count": self.pruned_failed_count,
                "target_prediction": format(output_value & self.mask, f"0{self.width}b"),
            },
        )

    def _solve_affine_target_bit(self, rhs: tuple[int, ...]) -> int | str | None:
        # Variables are constant term plus input bits in MSB order.
        rows = []
        for x, y in zip(self.xs, rhs):
            bits = [1] + [((x >> (self.width - 1 - bit)) & 1) for bit in range(self.width)]
            rows.append(bits + [y])
        rank, pivots = _gf2_rref(rows, self.width + 1)
        for row in rows:
            if not any(row[: self.width + 1]) and row[-1]:
                return None
        solution = [0] * (self.width + 1)
        for r, pivot in enumerate(pivots):
            solution[pivot] = rows[r][-1]
        target_vec = [1] + [((self.target >> (self.width - 1 - bit)) & 1) for bit in range(self.width)]
        value = _dot2(solution, target_vec)
        pivot_set = set(pivots)
        for free in range(self.width + 1):
            if free in pivot_set:
                continue
            null = [0] * (self.width + 1)
            null[free] = 1
            for r, pivot in enumerate(pivots):
                null[pivot] = rows[r][free]
            if _dot2(null, target_vec):
                return "ambiguous"
        return value

    def _truth_table_mining_synthesis(self) -> RuleHypothesis | None:
        if len(self.xs) < 8:
            return None
        candidate_count = 0
        ambiguity_count = 0
        output_value = 0
        rule_ids: list[str] = []
        disagreements: list[int] = []
        for out_index in range(self.width):
            shift = self.width - 1 - out_index
            desired = tuple((y >> shift) & 1 for y in self.ys)
            matches: list[tuple[str, int]] = []
            for rule_id, values, target_bit in self._truth_table_bit_candidates():
                candidate_count += 1
                if values == desired:
                    matches.append((f"out{out_index}:{rule_id}", target_bit))
            if not matches:
                return None
            target_bits = {bit for _, bit in matches}
            ambiguity_count += max(0, len(matches) - 1)
            if len(target_bits) != 1:
                disagreements.append(out_index)
                continue
            bit = target_bits.pop()
            output_value = (output_value << 1) | bit
            rule_ids.extend(rule_id for rule_id, _ in matches[:2])
        metadata = {
            "candidate_count": self.candidate_count + candidate_count,
            "tested": self.candidate_count + candidate_count,
            "verified_count": self.width,
            "ambiguity_count": ambiguity_count,
            "depth_solved_distribution": {"truth_table_mining": self.width},
            "expr_id": "truth_table_bit_mining",
            "verified_rules": rule_ids[:25],
            "vector_hash_count": len(self.seen_vectors),
            "pruned_duplicate_count": self.pruned_duplicate_count,
            "pruned_failed_count": self.pruned_failed_count,
            "truth_table_candidate_count": candidate_count,
        }
        if disagreements:
            metadata["disagreement_bit_positions"] = disagreements
            return RuleHypothesis("bit_composition_solver", "truth_table_disagreement", None, False, "ambiguous_fits", metadata)
        prediction = format(output_value & self.mask, f"0{self.width}b")
        metadata["target_prediction"] = prediction
        return RuleHypothesis("bit_composition_solver", "truth_table_bit_mining", prediction, True, "verified_unique", metadata)

    def _truth_table_bit_candidates(self):
        bit_vectors = []
        for bit in range(self.width):
            shift = self.width - 1 - bit
            values = tuple((x >> shift) & 1 for x in self.xs)
            target = (self.target >> shift) & 1
            bit_vectors.append((f"b{bit}", values, target))
        yield "const0", tuple(0 for _ in self.xs), 0
        yield "const1", tuple(1 for _ in self.xs), 1
        for size in range(4, self.width + 1):
            for combo in combinations(range(self.width), size):
                names = [bit_vectors[index][0] for index in combo]
                vectors = [bit_vectors[index][1] for index in combo]
                targets = [bit_vectors[index][2] for index in combo]
                joined = "_".join(names)
                yield f"xorN_{joined}", tuple(_xor_bits(values) for values in zip(*vectors)), _xor_bits(targets)
                yield f"orN_{joined}", tuple(1 if any(values) else 0 for values in zip(*vectors)), 1 if any(targets) else 0
                yield f"andN_{joined}", tuple(1 if all(values) else 0 for values in zip(*vectors)), 1 if all(targets) else 0
                threshold = (size // 2) + 1
                yield f"majN_{joined}", tuple(1 if sum(values) >= threshold else 0 for values in zip(*vectors)), 1 if sum(targets) >= threshold else 0
                for exact in range(1, size):
                    yield f"exact{exact}_{joined}", tuple(1 if sum(values) == exact else 0 for values in zip(*vectors)), 1 if sum(targets) == exact else 0
        popcounts = tuple(x.bit_count() for x in self.xs)
        target_pop = self.target.bit_count()
        for threshold in range(1, self.width + 1):
            yield f"pop_ge_{threshold}", tuple(1 if pc >= threshold else 0 for pc in popcounts), 1 if target_pop >= threshold else 0
            yield f"pop_le_{threshold}", tuple(1 if pc <= threshold else 0 for pc in popcounts), 1 if target_pop <= threshold else 0
            yield f"pop_eq_{threshold}", tuple(1 if pc == threshold else 0 for pc in popcounts), 1 if target_pop == threshold else 0
        yield "pop_parity", tuple(pc & 1 for pc in popcounts), target_pop & 1
        thresholds = sorted(set(self.xs))
        for threshold in thresholds:
            yield f"int_ge_{threshold}", tuple(1 if x >= threshold else 0 for x in self.xs), 1 if self.target >= threshold else 0
            yield f"int_le_{threshold}", tuple(1 if x <= threshold else 0 for x in self.xs), 1 if self.target <= threshold else 0

    def _boolean_bit_candidates(self):
        bit_vectors = []
        for bit in range(self.width):
            shift = self.width - 1 - bit
            values = tuple((x >> shift) & 1 for x in self.xs)
            target = (self.target >> shift) & 1
            bit_vectors.append((f"b{bit}", values, target))
        yield "const0", tuple(0 for _ in self.xs), 0
        yield "const1", tuple(1 for _ in self.xs), 1
        for name, values, target in bit_vectors:
            yield name, values, target
            yield f"not_{name}", tuple(1 - value for value in values), 1 - target
        for i, left in enumerate(bit_vectors):
            for right in bit_vectors[i + 1 :]:
                lname, lvals, ltarget = left
                rname, rvals, rtarget = right
                yield f"xor_{lname}_{rname}", tuple(a ^ b for a, b in zip(lvals, rvals)), ltarget ^ rtarget
                yield f"and_{lname}_{rname}", tuple(a & b for a, b in zip(lvals, rvals)), ltarget & rtarget
                yield f"or_{lname}_{rname}", tuple(a | b for a, b in zip(lvals, rvals)), ltarget | rtarget
        for i in range(self.width):
            for j in range(i + 1, self.width):
                for k in range(j + 1, self.width):
                    an, av, at = bit_vectors[i]
                    bn, bv, bt = bit_vectors[j]
                    cn, cv, ct = bit_vectors[k]
                    yield f"xor3_{an}_{bn}_{cn}", tuple(a ^ b ^ c for a, b, c in zip(av, bv, cv)), at ^ bt ^ ct
                    yield f"maj_{an}_{bn}_{cn}", tuple((a & b) | (a & c) | (b & c) for a, b, c in zip(av, bv, cv)), (at & bt) | (at & ct) | (bt & ct)
                    yield f"choice_{an}_{bn}_{cn}", tuple((a & b) | ((1 - a) & c) for a, b, c in zip(av, bv, cv)), (at & bt) | ((1 - at) & ct)

    def _expr(self, expr_id: str, fn: Callable[[int], int], depth: int, operations: tuple[str, ...]) -> BitExpr:
        return BitExpr(expr_id, tuple(fn(x) & self.mask for x in self.xs), fn(self.target) & self.mask, depth, operations)

    def _from_seed(self, expr_id: str, seed: BitExpr, fn: Callable[[int], int], depth: int, operation: str) -> BitExpr:
        return BitExpr(expr_id, tuple(fn(value) & self.mask for value in seed.values), fn(seed.target_value) & self.mask, depth, seed.operations + (operation,))

    def _rol(self, value: int, k: int) -> int:
        k %= self.width
        return ((value << k) | (value >> (self.width - k))) & self.mask

    def _ror(self, value: int, k: int) -> int:
        k %= self.width
        return ((value >> k) | (value << (self.width - k))) & self.mask

    def _reverse_bits(self, value: int) -> int:
        result = 0
        for i in range(self.width):
            result = (result << 1) | ((value >> i) & 1)
        return result


def _parse_single_stream(examples: tuple[tuple[str, str], ...], target_input: str) -> tuple[tuple[int, ...], tuple[int, ...], int, int] | RuleHypothesis:
    widths = {len(x) for x, _ in examples} | {len(y) for _, y in examples} | {len(target_input)}
    if len(widths) != 1 or not widths:
        return _reject("mixed_widths")
    width = widths.pop()
    if any(set(x) - {"0", "1"} or set(y) - {"0", "1"} for x, y in examples) or set(target_input) - {"0", "1"}:
        return _reject("non_bit_input")
    return tuple(int(x, 2) for x, _ in examples), tuple(int(y, 2) for _, y in examples), int(target_input, 2), width


def _reject(reason: str) -> RuleHypothesis:
    return RuleHypothesis("bit_composition_solver", "reject", None, False, reason, {})


def _gf2_rref(rows: list[list[int]], var_count: int) -> tuple[int, list[int]]:
    rank = 0
    pivots: list[int] = []
    for col in range(var_count):
        pivot = None
        for r in range(rank, len(rows)):
            if rows[r][col]:
                pivot = r
                break
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        for r in range(len(rows)):
            if r != rank and rows[r][col]:
                rows[r] = [a ^ b for a, b in zip(rows[r], rows[rank])]
        pivots.append(col)
        rank += 1
        if rank == len(rows):
            break
    return rank, pivots


def _dot2(left: list[int], right: list[int]) -> int:
    value = 0
    for a, b in zip(left, right):
        value ^= (a & b)
    return value


def _xor_bits(values) -> int:
    out = 0
    for value in values:
        out ^= int(value)
    return out
