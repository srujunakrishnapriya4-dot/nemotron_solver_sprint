from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import combinations, product
from typing import Callable
from .hard_family_router import RuleHypothesis


EXPR = re.compile(r"^\s*(-?\d+)\s*([+\-*/%@#$?&])\s*(-?\d+)\s*$")
SYMBOLIC_EXPR = re.compile(r"^(.{2})(.)(.{2})$")


@dataclass(frozen=True)
class ProgramCandidate:
    program_id: str
    fn: Callable[[str], str]
    depth: int
    cluster: str


def solve_equation_symbolic_problem(examples: tuple[tuple[str, str], ...], target_input: str) -> RuleHypothesis:
    string_result = _solve_string_transform(examples, target_input)
    if string_result.verified or string_result.reason in {"ambiguous_transform", "ambiguous_symbol_digit_mapping", "unseen_target_symbol", "underdetermined_equation_program"}:
        return string_result
    parsed = []
    for raw, out in examples:
        m = EXPR.match(raw)
        if not m or not re.fullmatch(r"-?\d+", out.strip()):
            return _reject("unsupported_equation")
        parsed.append((int(m.group(1)), m.group(2), int(m.group(3)), int(out)))
    tm = EXPR.match(target_input)
    if not tm:
        return _reject("unsupported_target")
    if len(parsed) < 2:
        return RuleHypothesis("equation_symbolic_solver", "reject", None, False, "underdetermined_equation_program", {"example_count": len(parsed)})
    explicit_ops = {op for _, op, _, _ in parsed}
    target_op = tm.group(2)
    if target_op not in explicit_ops:
        return _reject("unseen_target_operator")
    # In the competition's equation_symbolic family, even familiar-looking
    # operators are secret symbols. Do not impose normal + - * / semantics here;
    # the rule must be induced and verified from examples.
    allowed = _all_ops()
    verified = []
    if len(parsed) >= 3:
        for rule_id, fn in _affine_binary_ops(parsed):
            try:
                pred = fn(int(tm.group(1)), int(tm.group(3)))
            except Exception:
                continue
            verified.append((rule_id, pred))
    for rule_id, fn in allowed:
        values = []
        ok = True
        for a, _, b, y in parsed:
            try:
                value = fn(a, b)
            except ZeroDivisionError:
                ok = False
                break
            if value != y:
                ok = False
                break
            values.append(value)
        if ok:
            try:
                pred = fn(int(tm.group(1)), int(tm.group(3)))
            except ZeroDivisionError:
                continue
            verified.append((rule_id, pred))
    if not verified:
        return _reject("no_verified_operator")
    outputs = {value for _, value in verified}
    if len(outputs) != 1:
        return RuleHypothesis("equation_symbolic_solver", "disagreement", None, False, "ambiguous_operator", {"verified": verified})
    return RuleHypothesis("equation_symbolic_solver", verified[0][0], str(outputs.pop()), True, "verified_unique", {"verified_count": len(verified)})


def _affine_binary_ops(parsed: list[tuple[int, str, int, int]]) -> list[tuple[str, Callable[[int, int], int]]]:
    out: list[tuple[str, Callable[[int, int], int]]] = []
    for a_coef in range(-10, 11):
        for b_coef in range(-10, 11):
            constants = {y - a_coef * a - b_coef * b for a, _, b, y in parsed}
            if len(constants) != 1:
                continue
            const = next(iter(constants))
            if abs(const) > 500:
                continue
            out.append((f"affine_{a_coef}_{b_coef}_{const}", lambda a, b, a_coef=a_coef, b_coef=b_coef, const=const: a_coef * a + b_coef * b + const))
            if len(out) >= 64:
                return out
    return out


def _all_ops():
    return [
        ("add", lambda a, b: a + b),
        ("sub_ab", lambda a, b: a - b),
        ("sub_ba", lambda a, b: b - a),
        ("abs_diff", lambda a, b: abs(a - b)),
        ("mul", lambda a, b: a * b),
        ("mul_minus_1", lambda a, b: a * b - 1),
        ("mul_plus_1", lambda a, b: a * b + 1),
        ("mod", lambda a, b: a % b),
        ("mod_ba", lambda a, b: b % a),
        ("exact_div", _div),
        ("exact_div_ba", lambda a, b: _div(b, a)),
        ("concat_ab", lambda a, b: int(f"{a:02d}{b:02d}")),
        ("concat_ba", lambda a, b: int(f"{b:02d}{a:02d}")),
        ("reverse_concat_ab", lambda a, b: int(f"{a:02d}{b:02d}"[::-1])),
        ("reverse_concat_ba", lambda a, b: int(f"{b:02d}{a:02d}"[::-1])),
    ]


def _candidate_ops(op: str):
    if op == "+":
        return [("add", lambda a, b: a + b)]
    if op == "-":
        return [("sub_ab", lambda a, b: a - b), ("sub_ba", lambda a, b: b - a), ("abs_diff", lambda a, b: abs(a - b))]
    if op == "*":
        return [("mul", lambda a, b: a * b)]
    if op == "/":
        return [("exact_div", _div)]
    if op == "%":
        return [("mod", lambda a, b: a % b)]
    return _all_ops()


def _div(a: int, b: int) -> int:
    if b == 0 or a % b:
        raise ZeroDivisionError("non_exact_division")
    return a // b


def _reject(reason: str) -> RuleHypothesis:
    return RuleHypothesis("equation_symbolic_solver", "reject", None, False, reason, {})


def _solve_string_transform(examples: tuple[tuple[str, str], ...], target_input: str) -> RuleHypothesis:
    exact = _exact_lookup(examples, target_input)
    if exact is not None:
        return RuleHypothesis("equation_symbolic_solver", "exact_lookup", exact, True, "verified_exact_lookup", {"verified_count": len(examples)})
    if len(examples) < 2:
        return RuleHypothesis("equation_symbolic_solver", "reject", None, False, "underdetermined_equation_program", {"example_count": len(examples)})
    symbol_digit = _solve_symbol_digit_bindings(examples, target_input)
    if symbol_digit.verified or symbol_digit.reason in {"ambiguous_symbol_digit_mapping", "unseen_target_symbol"}:
        return symbol_digit
    candidates: list[ProgramCandidate] = []
    candidates.extend(_operator_dispatch_candidates(examples, target_input))
    allow_operator_projection = _operator_projection_safe(examples, target_input)
    if allow_operator_projection:
        candidates.extend(_fixed_subsequence_candidates(examples, allow_repeated=True))
        candidates.extend(_operand_rewrite_candidates(examples))
        candidates.extend(_constant_template_candidates(examples))
        candidates.extend(_remove_charset_candidates(examples))
    elif not _looks_like_symbolic_expression_set(examples, target_input):
        candidates.extend(_fixed_subsequence_candidates(examples, allow_repeated=False))
        candidates.extend(_constant_template_candidates(examples))
        candidates.extend(_remove_charset_candidates(examples))
    candidates.extend(_char_bijection_candidates(examples))
    candidates.extend(_token_bijection_candidates(examples))
    verified: list[tuple[str, str]] = []
    for candidate in candidates:
        try:
            if all(candidate.fn(x) == y for x, y in examples):
                verified.append((candidate.program_id, candidate.fn(target_input)))
        except Exception:
            continue
    if not verified:
        return RuleHypothesis("equation_symbolic_solver", "reject", None, False, "unsupported_equation_transform", {"candidate_count": len(candidates)})
    outputs = {pred for _, pred in verified}
    if len(outputs) != 1:
        return RuleHypothesis("equation_symbolic_solver", "disagreement", None, False, "ambiguous_transform", {"verified": verified[:20], "ambiguity_count": len(verified), "candidate_count": len(candidates)})
    return RuleHypothesis("equation_symbolic_solver", verified[0][0], outputs.pop(), True, "verified_unique", {"verified_count": len(verified), "ambiguity_count": max(0, len(verified) - 1), "candidate_count": len(candidates)})


def _solve_symbol_digit_bindings(examples: tuple[tuple[str, str], ...], target_input: str) -> RuleHypothesis:
    pattern = re.compile(r"^([A-Za-z])\s*([^A-Za-z0-9\s])\s*([A-Za-z])$")
    parsed = []
    symbols: set[str] = set()
    for raw, out in examples:
        m = pattern.match(raw)
        if not m or not re.fullmatch(r"-?\d+", out.strip()):
            return _reject("not_symbol_digit_binding")
        left, op, right = m.group(1), m.group(2), m.group(3)
        symbols.update((left, right))
        parsed.append((left, op, right, int(out)))
    tm = pattern.match(target_input)
    if not tm:
        return _reject("not_symbol_digit_binding")
    target_symbols = {tm.group(1), tm.group(3)}
    if not target_symbols.issubset(symbols):
        return RuleHypothesis("equation_symbolic_solver", "reject", None, False, "unseen_target_symbol", {"symbols": sorted(symbols), "target_symbols": sorted(target_symbols)})
    if len(symbols) > 7:
        return _reject("symbol_digit_search_too_large")
    ordered = sorted(symbols)
    ops = _all_symbol_digit_ops()
    verified: list[tuple[str, str]] = []
    candidate_count = 0
    for digits in product(range(10), repeat=len(ordered)):
        mapping = dict(zip(ordered, digits))
        for op_name, fn in ops:
            candidate_count += 1
            try:
                if all(fn(mapping[a], mapping[b]) == y for a, _, b, y in parsed):
                    pred = fn(mapping[tm.group(1)], mapping[tm.group(3)])
                    verified.append((f"symbol_digit_{op_name}", str(pred)))
            except ZeroDivisionError:
                continue
            if candidate_count > 100000:
                return _reject("symbol_digit_candidate_budget_exhausted")
    if not verified:
        return _reject("no_symbol_digit_mapping")
    outputs = {pred for _, pred in verified}
    if len(outputs) != 1:
        return RuleHypothesis("equation_symbolic_solver", "disagreement", None, False, "ambiguous_symbol_digit_mapping", {"ambiguity_count": len(verified), "candidate_count": candidate_count, "verified": verified[:20]})
    return RuleHypothesis("equation_symbolic_solver", verified[0][0], outputs.pop(), True, "verified_unique", {"verified_count": len(verified), "ambiguity_count": max(0, len(verified) - 1), "candidate_count": candidate_count})


def _all_symbol_digit_ops() -> list[tuple[str, Callable[[int, int], int]]]:
    return [
        ("add", lambda a, b: a + b),
        ("sub_ab", lambda a, b: a - b),
        ("sub_ba", lambda a, b: b - a),
        ("abs_diff", lambda a, b: abs(a - b)),
        ("mul", lambda a, b: a * b),
        ("mod", lambda a, b: a % b),
        ("mod_ba", lambda a, b: b % a),
        ("exact_div", _div),
        ("exact_div_ba", lambda a, b: _div(b, a)),
    ]


def _exact_lookup(examples: tuple[tuple[str, str], ...], target_input: str) -> str | None:
    seen: dict[str, str] = {}
    for x, y in examples:
        if x in seen and seen[x] != y:
            return None
        seen[x] = y
    return seen.get(target_input)


def _looks_like_symbolic_expression_set(examples: tuple[tuple[str, str], ...], target_input: str) -> bool:
    return len(target_input) >= 3 and all(len(x) == 5 for x, _ in examples)


def _operator_projection_safe(examples: tuple[tuple[str, str], ...], target_input: str) -> bool:
    if not _looks_like_symbolic_expression_set(examples, target_input):
        return True
    # Position 2 behaves as the operator in the real competition equation prompts.
    # Extrapolating a drop/reorder rule to an unseen operator produced verified-on-
    # examples but wrong target predictions, so require target operator evidence.
    return len(target_input) == 5 and target_input[2] in {x[2] for x, _ in examples}


def _operator_dispatch_candidates(examples, target_input):
    if len(target_input) != 5:
        return []
    if any(len(x) != 5 for x, _ in examples):
        return []
    target_op = target_input[2]
    groups = {}
    for x, y in examples:
        groups.setdefault(x[2], []).append((x, y))
    if target_op not in groups or len(groups[target_op]) < 2:
        return []
    per_op_options = {}
    for op, vals in groups.items():
        if len(vals) == 1:
            one_x, one_y = vals[0]
            opts = [ProgramCandidate(f"exact_seen_{op}", lambda text, one_x=one_x, one_y=one_y: one_y if text == one_x else (_raise()), 1, "lookup_table")]
        else:
            opts = _group_transform_options(vals)
            if not opts:
                return []
        per_op_options[op] = opts[:256]
    out = []
    # Keep the product bounded: choose the first verified transform per non-target op;
    # enumerate target-op alternatives because they affect the prediction.
    fixed = {op: opts[0] for op, opts in per_op_options.items() if op != target_op}
    for target_candidate in per_op_options[target_op]:
        funcs = dict(fixed)
        funcs[target_op] = target_candidate
        def fn(text, funcs=funcs):
            if len(text) != 5 or text[2] not in funcs:
                raise ValueError("unknown_operator")
            return funcs[text[2]].fn(text)
        out.append(ProgramCandidate("operator_dispatch_" + target_candidate.program_id, fn, target_candidate.depth + 1, "operator_conditioned"))
    return out


def _group_transform_options(vals):
    raw = []
    examples = tuple(vals)
    raw.extend(_fixed_subsequence_candidates(examples, allow_repeated=True))
    raw.extend(_operand_rewrite_candidates(examples))
    raw.extend(_constant_template_candidates(examples))
    raw.extend(_remove_charset_candidates(examples))
    # digit arithmetic options for two two-digit operands around a symbolic operator
    if len(examples) >= 2 and all(x[:2].isdigit() and x[3:].isdigit() and y.lstrip("-").isdigit() for x, y in examples):
        parsed = [(int(x[:2]), int(x[3:]), int(y)) for x, y in examples]
        arithmetic = [
            ("num_add", lambda a, b: a + b),
            ("num_sub", lambda a, b: a - b),
            ("num_sub_ba", lambda a, b: b - a),
            ("num_absdiff", lambda a, b: abs(a - b)),
            ("num_mul", lambda a, b: a * b),
            ("num_mul_minus_1", lambda a, b: a * b - 1),
            ("num_mul_plus_1", lambda a, b: a * b + 1),
            ("num_mod", lambda a, b: a % b),
            ("num_mod_ba", lambda a, b: b % a),
            ("num_exact_div", _div),
            ("num_exact_div_ba", lambda a, b: _div(b, a)),
            ("concat_ab", lambda a, b: int(f"{a:02d}{b:02d}")),
            ("concat_ba", lambda a, b: int(f"{b:02d}{a:02d}")),
            ("reverse_concat_ab", lambda a, b: int(f"{a:02d}{b:02d}"[::-1])),
            ("reverse_concat_ba", lambda a, b: int(f"{b:02d}{a:02d}"[::-1])),
        ]
        for name, op in arithmetic:
            try:
                ok = all(op(a, b) == y for a, b, y in parsed)
            except ZeroDivisionError:
                ok = False
            if ok:
                raw.append(ProgramCandidate(name, lambda text, op=op: str(op(int(text[:2]), int(text[3:]))), 1, "binary_operator_arithmetic"))
        if len(examples) >= 3:
            raw.extend(_numeric_affine_candidates(examples))
    out = []
    for candidate in raw:
        try:
            if all(candidate.fn(x) == y for x, y in examples):
                out.append(candidate)
        except Exception:
            continue
    # preserve deterministic order while removing duplicate rule ids
    seen = set()
    unique = []
    for candidate in out:
        if candidate.program_id not in seen:
            seen.add(candidate.program_id)
            unique.append(candidate)
    return unique


def _raise():
    raise ValueError("exact lookup cannot extrapolate")


def _fixed_subsequence_candidates(examples, *, allow_repeated: bool = False):
    out = []
    lengths = {len(x) for x, _ in examples}
    if len(lengths) == 1:
        width = next(iter(lengths))
        output_lengths = {len(y) for _, y in examples}
        for olen in sorted(output_lengths):
            for idxs in combinations(range(width), olen):
                def fn(text, idxs=idxs):
                    if len(text) != width:
                        raise ValueError("width")
                    return "".join(text[i] for i in idxs)
                out.append(ProgramCandidate(f"keep_positions_{','.join(map(str, idxs))}", fn, 1, "string_rewrite"))
                out.append(ProgramCandidate(f"keep_positions_reverse_{','.join(map(str, idxs))}", lambda text, fn=fn: fn(text)[::-1], 2, "string_rewrite"))
            if allow_repeated and olen <= 6 and width <= 6:
                for idxs in product(range(width), repeat=olen):
                    if len(set(idxs)) == olen:
                        continue
                    def fn(text, idxs=idxs, width=width):
                        if len(text) != width:
                            raise ValueError("width")
                        return "".join(text[i] for i in idxs)
                    out.append(ProgramCandidate(f"select_positions_{','.join(map(str, idxs))}", fn, 1, "string_rewrite"))
    return out


def _constant_template_candidates(examples):
    if len(examples) < 2:
        return []
    lengths = {len(x) for x, _ in examples}
    output_lengths = {len(y) for _, y in examples}
    if len(lengths) != 1 or len(output_lengths) != 1:
        return []
    width = next(iter(lengths))
    olen = next(iter(output_lengths))
    if width > 6 or olen > 6:
        return []
    slots = []
    for pos in range(olen):
        choices = []
        for idx in range(width):
            if all(x[idx] == y[pos] for x, y in examples):
                choices.append(("i", idx))
        constants = {y[pos] for _, y in examples}
        if len(constants) == 1:
            choices.append(("c", next(iter(constants))))
        if not choices:
            return []
        slots.append(choices[:8])
    out = []
    for spec in product(*slots):
        if not any(kind == "i" for kind, _ in spec):
            continue
        def fn(text, spec=spec, width=width):
            if len(text) != width:
                raise ValueError("width")
            return "".join(text[value] if kind == "i" else value for kind, value in spec)
        signature = ",".join(f"{kind}{value}" for kind, value in spec)
        out.append(ProgramCandidate(f"template_{signature}", fn, 2, "deletion_insertion"))
        if len(out) >= 256:
            break
    return out


def _numeric_affine_candidates(examples):
    if len(examples) < 3:
        return []
    parsed = []
    for x, y in examples:
        if not (len(x) == 5 and x[:2].isdigit() and x[3:].isdigit() and y.lstrip("-").isdigit()):
            return []
        parsed.append((int(x[:2]), int(x[3:]), int(y)))
    out = []
    for a_coef in range(-10, 11):
        for b_coef in range(-10, 11):
            constants = {y - a_coef * a - b_coef * b for a, b, y in parsed}
            if len(constants) != 1:
                continue
            const = next(iter(constants))
            if abs(const) > 500:
                continue
            def fn(text, a_coef=a_coef, b_coef=b_coef, const=const):
                if len(text) != 5 or not (text[:2].isdigit() and text[3:].isdigit()):
                    raise ValueError("numeric")
                a = int(text[:2])
                b = int(text[3:])
                return str(a_coef * a + b_coef * b + const)
            out.append(ProgramCandidate(f"affine_{a_coef}_{b_coef}_{const}", fn, 2, "binary_operator_arithmetic"))
            if len(out) >= 64:
                return out
    return out


def _operand_rewrite_candidates(examples):
    if not examples or any(len(x) != 5 for x, _ in examples):
        return []
    specs = [
        ("drop_operator", (0, 1, 3, 4)),
        ("swap_operands", (3, 4, 0, 1)),
        ("left_operand", (0, 1)),
        ("right_operand", (3, 4)),
        ("reverse_left_then_right", (1, 0, 3, 4)),
        ("left_then_reverse_right", (0, 1, 4, 3)),
        ("reverse_operands", (4, 3, 1, 0)),
        ("outer_inner", (0, 4, 1, 3)),
        ("inner_outer", (1, 3, 0, 4)),
        ("left_op", (0, 1, 2)),
        ("op_right", (2, 3, 4)),
        ("left_first_right_first", (0, 3)),
        ("left_second_right_second", (1, 4)),
    ]
    out = []
    for name, idxs in specs:
        def fn(text, idxs=idxs):
            if len(text) != 5:
                raise ValueError("width")
            return "".join(text[i] for i in idxs)
        out.append(ProgramCandidate(name, fn, 1, "string_rewrite"))
    return out


def _remove_charset_candidates(examples):
    chars = sorted(set("".join(x for x, _ in examples)))
    out = []
    for size in range(1, min(4, len(chars)) + 1):
        for removed in combinations(chars, size):
            removed_set = frozenset(removed)
            def fn(text, removed_set=removed_set):
                return "".join(ch for ch in text if ch not in removed_set)
            rid = "remove_chars_" + "".join(removed)
            out.append(ProgramCandidate(rid, fn, 1, "deletion_insertion"))
            out.append(ProgramCandidate(rid + "_reverse", lambda text, fn=fn: fn(text)[::-1], 2, "deletion_insertion"))
    return out


def _token_bijection_candidates(examples):
    if not examples:
        return []
    if any(" " not in x + y for x, y in examples):
        return []
    mapping = {}
    reverse = {}
    for x, y in examples:
        xs = x.split()
        ys = y.split()
        if len(xs) != len(ys):
            return []
        for a, b in zip(xs, ys):
            if mapping.get(a, b) != b or reverse.get(b, a) != a:
                return []
            mapping[a] = b
            reverse[b] = a
    def fn(text, mapping=mapping):
        toks = text.split()
        if any(tok not in mapping for tok in toks):
            raise ValueError("unseen_token")
        return " ".join(mapping[tok] for tok in toks)
    return [ProgramCandidate("token_bijection", fn, 1, "token_substitution")]


def _char_bijection_candidates(examples):
    mapping = {}
    reverse = {}
    for x, y in examples:
        if len(x) != len(y):
            return []
        for a, b in zip(x, y):
            if mapping.get(a, b) != b or reverse.get(b, a) != a:
                return []
            mapping[a] = b
            reverse[b] = a
    def fn(text, mapping=mapping):
        if any(ch not in mapping for ch in text):
            raise ValueError("unseen")
        return "".join(mapping[ch] for ch in text)
    return [ProgramCandidate("char_bijection", fn, 1, "char_substitution")]
