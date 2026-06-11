from __future__ import annotations

from dataclasses import dataclass, field
import random
import re
from typing import Any, Protocol

from kaggle_anti086.training.day1_teacher_verifier import (
    extract_boxed_answer,
    validate_target_text,
)


@dataclass(frozen=True)
class TeacherResult:
    answer: str | None
    trace: str | None
    verification_status: str
    ambiguity_count: int
    family: str
    rule_id: str | None
    confidence: float
    rejection_reason: str | None
    prompt: str | None = None
    target_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SyntheticExample:
    family: str
    prompt: str
    answer: str
    trace: str
    target_text: str
    rule_id: str
    difficulty: str
    prompt_style: str
    verification_status: str
    ambiguity_count: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class HardNegative:
    family: str
    prompt: str
    chosen: str
    rejected: str
    correct_answer: str
    rejected_answer: str | None
    reason_rejected: str
    verifier_status: str
    metadata: dict[str, Any]


class FamilyTeacher(Protocol):
    family: str

    def generate_synthetic(self, rng: random.Random, difficulty: str, prompt_style: str) -> SyntheticExample:
        ...

    def solve(self, prompt: str) -> TeacherResult:
        ...

    def verify_example(self, example: SyntheticExample) -> TeacherResult:
        ...

    def generate_hard_negative(self, example: SyntheticExample, rng: random.Random) -> HardNegative:
        ...


def _trace(lines: list[str], answer: str) -> str:
    return "\n".join([*lines, f"\\boxed{{{answer}}}"])


def _fail(family: str, prompt: str, reason: str, rule_id: str | None = None) -> TeacherResult:
    return TeacherResult(
        answer=None,
        trace=None,
        verification_status="FAIL",
        ambiguity_count=1,
        family=family,
        rule_id=rule_id,
        confidence=0.0,
        rejection_reason=reason,
        prompt=prompt,
    )


def _pass(family: str, prompt: str, answer: str, trace: str, rule_id: str, metadata: dict[str, Any]) -> TeacherResult:
    ok, reason = validate_target_text(trace, answer)
    return TeacherResult(
        answer=answer if ok else None,
        trace=trace if ok else None,
        target_text=trace if ok else None,
        verification_status="PASS" if ok else "FAIL",
        ambiguity_count=0 if ok else 1,
        family=family,
        rule_id=rule_id,
        confidence=1.0 if ok else 0.0,
        rejection_reason=None if ok else reason,
        prompt=prompt,
        metadata=metadata,
    )


def _example_from_result(result: TeacherResult, difficulty: str, prompt_style: str) -> SyntheticExample:
    if result.verification_status != "PASS" or result.answer is None or result.trace is None or result.prompt is None:
        raise ValueError(f"Cannot build synthetic example from failed result: {result.rejection_reason}")
    return SyntheticExample(
        family=result.family,
        prompt=result.prompt,
        answer=result.answer,
        trace=result.trace,
        target_text=result.trace,
        rule_id=result.rule_id or "unknown",
        difficulty=difficulty,
        prompt_style=prompt_style,
        verification_status=result.verification_status,
        ambiguity_count=result.ambiguity_count,
        metadata=dict(result.metadata),
    )


def _verify_generated(teacher: FamilyTeacher, example: SyntheticExample) -> TeacherResult:
    solved = teacher.solve(example.prompt)
    if solved.verification_status != "PASS" or solved.answer != example.answer:
        return _fail(example.family, example.prompt, "generated_example_does_not_resolve_to_expected", example.rule_id)
    ok, reason = validate_target_text(example.target_text, example.answer)
    if not ok:
        return _fail(example.family, example.prompt, reason or "target_format_error", example.rule_id)
    return solved


class SymbolMappingTeacher:
    family = "symbol_mapping"
    symbols = list("△□○◇☆@#%&ABCDXYZ")

    def generate_synthetic(self, rng: random.Random, difficulty: str, prompt_style: str) -> SyntheticExample:
        count = 4 if difficulty != "hard" else 6
        keys = rng.sample(self.symbols, count)
        values = rng.sample(list("0123456789"), count)
        mapping = dict(zip(keys, values))
        query = "".join(rng.choice(keys) for _ in range(3 if difficulty == "easy" else 4))
        prompt = _format_mapping_prompt(self.family, mapping, query, prompt_style)
        return _example_from_result(self.solve(prompt), difficulty, prompt_style)

    def solve(self, prompt: str) -> TeacherResult:
        parsed = _parse_mapping_prompt(prompt)
        if parsed is None:
            return _fail(self.family, prompt, "mapping_parse_failed", "symbol_mapping")
        mapping, query = parsed
        if len(set(mapping)) != len(mapping) or len(set(mapping.values())) != len(mapping.values()):
            return _fail(self.family, prompt, "duplicate_or_non_unique_mapping", "symbol_mapping")
        if any(ch not in mapping for ch in query):
            return _fail(self.family, prompt, "unseen_symbol_in_query", "symbol_mapping")
        answer = "".join(mapping[ch] for ch in query)
        trace = _trace([f"Apply the explicit mapping to {query}.", f"The query becomes {answer}."], answer)
        return _pass(self.family, prompt, answer, trace, "symbol_mapping", {"mapping": mapping, "query": query})

    def verify_example(self, example: SyntheticExample) -> TeacherResult:
        return _verify_generated(self, example)

    def generate_hard_negative(self, example: SyntheticExample, rng: random.Random) -> HardNegative:
        rejected_answer = example.answer[::-1] if len(example.answer) > 1 else example.answer + "0"
        rejected = _trace(["Use a swapped mapping incorrectly.", f"This gives {rejected_answer}."], rejected_answer)
        return _negative_from_trace(example, rejected, rejected_answer, "swapped_mapping")


class BitManipulationTeacher:
    family = "bit_manipulation"

    def generate_synthetic(self, rng: random.Random, difficulty: str, prompt_style: str) -> SyntheticExample:
        width = 8 if difficulty != "hard" else 10
        op = rng.choice(["XOR", "AND", "OR", "NOT", "REVERSE", "BIN2DEC", "DEC2BIN", "LSHIFT", "RSHIFT"])
        a = rng.randint(0, (1 << width) - 1)
        b = rng.randint(0, (1 << width) - 1)
        shift = rng.randint(1, 2)
        if op in {"XOR", "AND", "OR"}:
            prompt = f"[bit_manipulation] Use {width}-bit {op}. Input: {_bits(a, width)} {op} {_bits(b, width)}. Answer as decimal."
        elif op == "NOT":
            prompt = f"[bit_manipulation] Use {width}-bit NOT. Input: NOT {_bits(a, width)}. Answer as decimal."
        elif op == "REVERSE":
            prompt = f"[bit_manipulation] Use {width}-bit bit reverse. Input: {_bits(a, width)}. Answer as decimal."
        elif op == "BIN2DEC":
            prompt = f"[bit_manipulation] Convert binary to decimal. Input: {_bits(a, width)}."
        elif op == "DEC2BIN":
            prompt = f"[bit_manipulation] Convert decimal to {width}-bit binary. Input: {a}."
        elif op == "LSHIFT":
            prompt = f"[bit_manipulation] Use {width}-bit left shift by {shift}. Input: {_bits(a, width)}. Answer as decimal."
        else:
            prompt = f"[bit_manipulation] Use {width}-bit right shift by {shift}. Input: {_bits(a, width)}. Answer as decimal."
        return _example_from_result(self.solve(prompt), difficulty, prompt_style)

    def solve(self, prompt: str) -> TeacherResult:
        width_match = re.search(r"(\d+)-bit", prompt)
        width = int(width_match.group(1)) if width_match else None
        lower = prompt.lower()
        bits = re.findall(r"\b[01]+\b", prompt)
        if "binary to decimal" in lower:
            input_match = re.search(r"Input:\s*([0-9]+)", prompt, re.IGNORECASE)
            if input_match and not re.fullmatch(r"[01]+", input_match.group(1)):
                return _fail(self.family, prompt, "binary_invalid_digit", "bit")
        if "not" in lower and width is None:
            return _fail(self.family, prompt, "not_without_width", "not")
        if "reverse" in lower and width is None:
            return _fail(self.family, prompt, "bit_reverse_without_width", "reverse")
        try:
            if "xor" in lower and len(bits) >= 2 and width:
                value = int(bits[0], 2) ^ int(bits[1], 2)
                answer = str(value)
                detail = f"{bits[0]} XOR {bits[1]} = {_bits(value, width)}, which is {answer}."
                rule = "xor"
            elif " and " in f" {lower} " and len(bits) >= 2 and width:
                value = int(bits[0], 2) & int(bits[1], 2)
                answer = str(value)
                detail = f"{bits[0]} AND {bits[1]} = {_bits(value, width)}, which is {answer}."
                rule = "and"
            elif " or " in f" {lower} " and len(bits) >= 2 and width:
                value = int(bits[0], 2) | int(bits[1], 2)
                answer = str(value)
                detail = f"{bits[0]} OR {bits[1]} = {_bits(value, width)}, which is {answer}."
                rule = "or"
            elif "not" in lower and bits and width:
                value = ((1 << width) - 1) ^ int(bits[0], 2)
                answer = str(value)
                detail = f"NOT {bits[0]} = {_bits(value, width)}, which is {answer}."
                rule = "not"
            elif "reverse" in lower and bits and width:
                reversed_bits = bits[0][::-1]
                answer = str(int(reversed_bits, 2))
                detail = f"Reverse {bits[0]} to {reversed_bits}, which is {answer}."
                rule = "reverse"
            elif "binary to decimal" in lower and bits:
                answer = str(int(bits[0], 2))
                detail = f"Binary {bits[0]} is decimal {answer}."
                rule = "bin2dec"
            elif "decimal to" in lower and width:
                number = _last_int(prompt)
                if number is None or number < 0 or number >= (1 << width):
                    return _fail(self.family, prompt, "decimal_out_of_width_range", "dec2bin")
                answer = _bits(number, width)
                detail = f"Decimal {number} in {width}-bit binary is {answer}."
                rule = "dec2bin"
            elif "left shift" in lower and bits and width:
                shift = _shift_amount(prompt)
                value = (int(bits[0], 2) << shift) & ((1 << width) - 1)
                answer = str(value)
                detail = f"Left shift by {shift} gives {_bits(value, width)}, which is {answer}."
                rule = "lshift"
            elif "right shift" in lower and bits and width:
                shift = _shift_amount(prompt)
                value = int(bits[0], 2) >> shift
                answer = str(value)
                detail = f"Right shift by {shift} gives {_bits(value, width)}, which is {answer}."
                rule = "rshift"
            else:
                return _fail(self.family, prompt, "bit_operation_parse_failed", "bit")
        except ValueError:
            return _fail(self.family, prompt, "bit_operation_value_error", "bit")
        trace = _trace([f"Use {width}-bit {rule.upper()}." if width else f"Use {rule}.", detail], answer)
        return _pass(self.family, prompt, answer, trace, rule, {})

    def verify_example(self, example: SyntheticExample) -> TeacherResult:
        return _verify_generated(self, example)

    def generate_hard_negative(self, example: SyntheticExample, rng: random.Random) -> HardNegative:
        wrong = str((int(example.answer) + 1) % 256) if example.answer.isdigit() else example.answer[::-1]
        rejected = _trace(["Use the wrong bit operation.", f"This gives {wrong}."], wrong)
        return _negative_from_trace(example, rejected, wrong, "wrong_bit_operation")


class CharCipherTeacher:
    family = "char_cipher"

    def generate_synthetic(self, rng: random.Random, difficulty: str, prompt_style: str) -> SyntheticExample:
        word = _random_word(rng, 3, 8)
        rule = rng.choice(["caesar", "atbash", "reverse"])
        if rule == "caesar":
            shift = rng.randint(1, 12)
            prompt = f"[char_cipher] Caesar shift +{shift}. Encode: {word}"
        elif rule == "atbash":
            prompt = f"[char_cipher] Atbash encode: {word}"
        else:
            prompt = f"[char_cipher] Reverse string: {word}"
        return _example_from_result(self.solve(prompt), difficulty, prompt_style)

    def solve(self, prompt: str) -> TeacherResult:
        lower = prompt.lower()
        words = re.findall(r"[a-z]+", prompt)
        word = words[-1] if words else ""
        if not word.isalpha() or not word.islower():
            return _fail(self.family, prompt, "unknown_or_case_ambiguous_characters", "char")
        if "examples:" in lower and "query" in lower and "shift" not in lower:
            return _fail(self.family, prompt, "ambiguous_shift", "caesar")
        if "caesar" in lower or "shift" in lower:
            match = re.search(r"([+-]\d+)", prompt)
            if not match:
                return _fail(self.family, prompt, "missing_caesar_shift", "caesar")
            shift = int(match.group(1))
            answer = _caesar(word, shift)
            trace = _trace([f"The rule is Caesar shift {shift:+d}.", f"{word} becomes {answer}."], answer)
            return _pass(self.family, prompt, answer, trace, "caesar", {"shift": shift})
        if "atbash" in lower:
            answer = _atbash(word)
            trace = _trace(["Use Atbash a<->z, b<->y, and so on.", f"{word} becomes {answer}."], answer)
            return _pass(self.family, prompt, answer, trace, "atbash", {})
        if "reverse" in lower:
            answer = word[::-1]
            trace = _trace(["Reverse the string.", f"{word} becomes {answer}."], answer)
            return _pass(self.family, prompt, answer, trace, "reverse", {})
        return _fail(self.family, prompt, "char_cipher_parse_failed", "char")

    def verify_example(self, example: SyntheticExample) -> TeacherResult:
        return _verify_generated(self, example)

    def generate_hard_negative(self, example: SyntheticExample, rng: random.Random) -> HardNegative:
        wrong = example.answer[::-1] if len(example.answer) > 1 else example.answer + "x"
        rejected = _trace(["Apply the wrong cipher.", f"This gives {wrong}."], wrong)
        return _negative_from_trace(example, rejected, wrong, "wrong_cipher")


class UnitConversionTeacher:
    family = "unit_conversion"
    factors = {"mm": 1, "cm": 10, "m": 1000, "km": 1_000_000, "g": 1, "kg": 1000, "seconds": 1, "minutes": 60, "hours": 3600, "days": 86400, "units": 1, "dozens": 12}

    def generate_synthetic(self, rng: random.Random, difficulty: str, prompt_style: str) -> SyntheticExample:
        groups = [("mm", "cm", "m", "km"), ("g", "kg"), ("seconds", "minutes", "hours", "days"), ("units", "dozens")]
        group = rng.choice(groups)
        src, dst = rng.sample(group, 2)
        src_factor, dst_factor = self.factors[src], self.factors[dst]
        multiplier = dst_factor // src_factor if dst_factor >= src_factor and dst_factor % src_factor == 0 else src_factor // dst_factor
        value = rng.randint(1, 200) * multiplier
        if value * src_factor % dst_factor != 0:
            value *= dst_factor
        prompt = f"[unit_conversion] Convert {value} {src} to {dst}. Exact integer only."
        return _example_from_result(self.solve(prompt), difficulty, prompt_style)

    def solve(self, prompt: str) -> TeacherResult:
        value_token = re.search(r"Convert\s+([0-9]+(?:\.[0-9]+)?)\s+([a-z]+)\s+to\s+([a-z]+)", prompt, re.IGNORECASE)
        if value_token and "." in value_token.group(1):
            return _fail(self.family, prompt, "decimal_rounding_ambiguity", "unit")
        match = re.search(r"Convert\s+(\d+)\s+([a-z]+)\s+to\s+([a-z]+)", prompt, re.IGNORECASE)
        if not match:
            return _fail(self.family, prompt, "unit_conversion_parse_failed", "unit")
        value = int(match.group(1))
        src = match.group(2).lower()
        dst = match.group(3).lower()
        if src not in self.factors or dst not in self.factors:
            return _fail(self.family, prompt, "ambiguous_unit_name", "unit")
        base = value * self.factors[src]
        if base % self.factors[dst] != 0:
            return _fail(self.family, prompt, "non_integer_conversion", "unit")
        answer = str(base // self.factors[dst])
        trace = _trace([f"1 {dst} = {self.factors[dst]} base units and 1 {src} = {self.factors[src]} base units.", f"{value} {src} = {answer} {dst}."], answer)
        return _pass(self.family, prompt, answer, trace, "unit_exact", {"src": src, "dst": dst})

    def verify_example(self, example: SyntheticExample) -> TeacherResult:
        return _verify_generated(self, example)

    def generate_hard_negative(self, example: SyntheticExample, rng: random.Random) -> HardNegative:
        wrong = str(int(example.answer) * 10) if example.answer.isdigit() else example.answer + "0"
        rejected = _trace(["Use the wrong conversion factor.", f"This gives {wrong}."], wrong)
        return _negative_from_trace(example, rejected, wrong, "wrong_factor")


class NumericFormulaSafeTeacher:
    family = "numeric_formula_safe"

    def generate_synthetic(self, rng: random.Random, difficulty: str, prompt_style: str) -> SyntheticExample:
        rule = rng.choice(["linear", "quadratic"])
        if rule == "linear":
            a = rng.randint(1, 20)
            b = rng.randint(-50, 50)
            xs = [0, 1, 2]
            q = rng.randint(3, 50)
            examples = "; ".join(f"{x}->{a*x+b}" for x in xs)
            prompt = f"[numeric_formula_safe] Formula class: linear. Examples: {examples}. Query {q}?"
        else:
            a = rng.randint(1, 10)
            b = rng.randint(-20, 20)
            c = rng.randint(-20, 20)
            xs = [0, 1, 2, 3]
            q = rng.randint(4, 50)
            examples = "; ".join(f"{x}->{a*x*x+b*x+c}" for x in xs)
            prompt = f"[numeric_formula_safe] Formula class: quadratic. Examples: {examples}. Query {q}?"
        return _example_from_result(self.solve(prompt), difficulty, prompt_style)

    def solve(self, prompt: str) -> TeacherResult:
        pairs = [(int(a), int(b)) for a, b in re.findall(r"(-?\d+)\s*->\s*(-?\d+)", prompt)]
        query_match = re.search(r"Query\s+(-?\d+)", prompt, re.IGNORECASE)
        lower = prompt.lower()
        if query_match is None:
            return _fail(self.family, prompt, "missing_query", "formula")
        q = int(query_match.group(1))
        if len(pairs) < 3:
            return _fail(self.family, prompt, "too_few_points_ambiguous", "formula")
        if "formula class" not in lower and len(pairs) < 4:
            return _fail(self.family, prompt, "too_few_points_ambiguous", "formula")
        if "1 -> 1" in prompt and "2 -> 4" in prompt and len(pairs) < 3:
            return _fail(self.family, prompt, "known_ambiguous_pattern", "formula")
        candidates: dict[str, int] = {}
        linear = _fit_linear(pairs)
        if linear is not None:
            a, b = linear
            candidates[f"linear:{a}:{b}"] = a * q + b
        quadratic = _fit_quadratic(pairs)
        if quadratic is not None and len(pairs) >= 4:
            a, b, c = quadratic
            candidates[f"quadratic:{a}:{b}:{c}"] = a * q * q + b * q + c
        outputs = set(candidates.values())
        if not candidates:
            return _fail(self.family, prompt, "no_exact_integer_formula", "formula")
        if len(outputs) > 1:
            return _fail(self.family, prompt, "multi_fit_different_query_output", "formula")
        rule, answer_value = next(iter(candidates.items()))
        answer = str(answer_value)
        trace = _trace([f"The exact safe formula candidate is {rule}.", f"For x = {q}, the output is {answer}."], answer)
        return _pass(self.family, prompt, answer, trace, rule.split(":")[0], {"candidate_count": len(candidates)})

    def verify_example(self, example: SyntheticExample) -> TeacherResult:
        return _verify_generated(self, example)

    def generate_hard_negative(self, example: SyntheticExample, rng: random.Random) -> HardNegative:
        wrong = str(int(example.answer) + rng.choice([-2, -1, 1, 2]))
        rejected = _trace(["Use the wrong coefficient.", f"This gives {wrong}."], wrong)
        return _negative_from_trace(example, rejected, wrong, "wrong_formula_coefficient")


class WordCipherTeacher:
    family = "word_cipher"
    words = ["red", "blue", "green", "stone", "river", "mint", "gold", "silver"]

    def generate_synthetic(self, rng: random.Random, difficulty: str, prompt_style: str) -> SyntheticExample:
        words = rng.sample(self.words, 3 if difficulty != "hard" else 4)
        rule = rng.choice(["reverse_words", "first_letters", "last_letters", "sort_letters"])
        prompt = f"[word_cipher] Rule: {rule}. Input: {' '.join(words)}"
        return _example_from_result(self.solve(prompt), difficulty, prompt_style)

    def solve(self, prompt: str) -> TeacherResult:
        lower = prompt.lower()
        if "synonym" in lower or "semantic" in lower:
            return _fail(self.family, prompt, "semantic_transform_rejected", "word")
        match = re.search(r"Rule:\s*([a-z_]+)\.\s*Input:\s*([a-z ]+)", prompt, re.IGNORECASE)
        if not match:
            return _fail(self.family, prompt, "word_cipher_parse_failed", "word")
        rule = match.group(1).lower()
        words = match.group(2).strip().split()
        if rule == "reverse_words":
            answer = " ".join(reversed(words))
            detail = "The rule reverses word order."
        elif rule == "first_letters":
            answer = "".join(word[0] for word in words)
            detail = "Take the first letter of each word."
        elif rule == "last_letters":
            answer = "".join(word[-1] for word in words)
            detail = "Take the last letter of each word."
        elif rule == "sort_letters":
            answer = " ".join("".join(sorted(word)) for word in words)
            detail = "Sort letters inside each word."
        else:
            return _fail(self.family, prompt, "unknown_word_rule", rule)
        trace = _trace([detail, f"{' '.join(words)} becomes {answer}."], answer)
        return _pass(self.family, prompt, answer, trace, rule, {})

    def verify_example(self, example: SyntheticExample) -> TeacherResult:
        return _verify_generated(self, example)

    def generate_hard_negative(self, example: SyntheticExample, rng: random.Random) -> HardNegative:
        wrong = example.answer[::-1]
        if wrong == example.answer:
            wrong = example.answer + "x"
        rejected = _trace(["Apply a different word rule.", f"This gives {wrong}."], wrong)
        return _negative_from_trace(example, rejected, wrong, "wrong_word_rule")


class CustomNumeralTeacher:
    family = "custom_numeral"
    symbols = list("ABCDEFGH@#%&△□○◇")

    def generate_synthetic(self, rng: random.Random, difficulty: str, prompt_style: str) -> SyntheticExample:
        base = rng.randint(2, 8)
        alphabet = rng.sample(self.symbols, base)
        mode = rng.choice(["decode", "encode"])
        value = rng.randint(1, min(120, base**3 - 1))
        numeral = _encode_custom(value, alphabet)
        mapping = ",".join(f"{sym}={idx}" for idx, sym in enumerate(alphabet))
        if mode == "decode":
            prompt = f"[custom_numeral] Base {base}. Digits: {mapping}. Decode {numeral} to decimal."
        else:
            prompt = f"[custom_numeral] Base {base}. Digits: {mapping}. Encode decimal {value}."
        return _example_from_result(self.solve(prompt), difficulty, prompt_style)

    def solve(self, prompt: str) -> TeacherResult:
        base_match = re.search(r"Base\s+(\d+)", prompt, re.IGNORECASE)
        digits_match = re.search(r"Digits:\s*([^.]+)\.", prompt)
        if base_match is None or digits_match is None:
            return _fail(self.family, prompt, "ambiguous_base_or_digit_order", "custom")
        base = int(base_match.group(1))
        entries = [part.strip().split("=") for part in digits_match.group(1).split(",")]
        if len(entries) != base:
            return _fail(self.family, prompt, "mapping_size_does_not_match_base", "custom")
        mapping = {sym: int(idx) for sym, idx in entries}
        if len(mapping) != base or set(mapping.values()) != set(range(base)):
            return _fail(self.family, prompt, "mapping_not_unique", "custom")
        if "Decode" in prompt:
            token = re.search(r"Decode\s+(\S+)\s+to decimal", prompt)
            if token is None:
                return _fail(self.family, prompt, "decode_query_missing", "custom_decode")
            numeral = token.group(1)
            if any(ch not in mapping for ch in numeral):
                return _fail(self.family, prompt, "query_has_unseen_symbol", "custom_decode")
            value = 0
            for ch in numeral:
                value = value * base + mapping[ch]
            answer = str(value)
            trace = _trace([f"Use base {base} with the given digit mapping.", f"{numeral} decodes to {answer}."], answer)
            return _pass(self.family, prompt, answer, trace, "custom_decode", {"base": base})
        number = _last_int(prompt)
        if number is None:
            return _fail(self.family, prompt, "encode_query_missing", "custom_encode")
        alphabet = [sym for sym, _idx in sorted(mapping.items(), key=lambda item: item[1])]
        answer = _encode_custom(number, alphabet)
        trace = _trace([f"Use base {base} with the given digit mapping.", f"Decimal {number} encodes as {answer}."], answer)
        return _pass(self.family, prompt, answer, trace, "custom_encode", {"base": base})

    def verify_example(self, example: SyntheticExample) -> TeacherResult:
        return _verify_generated(self, example)

    def generate_hard_negative(self, example: SyntheticExample, rng: random.Random) -> HardNegative:
        wrong = example.answer[::-1] if len(example.answer) > 1 else example.answer + "A"
        rejected = _trace(["Use the wrong base or digit order.", f"This gives {wrong}."], wrong)
        return _negative_from_trace(example, rejected, wrong, "wrong_custom_base")


class PermutationSortingTeacher:
    family = "permutation_sorting"

    def generate_synthetic(self, rng: random.Random, difficulty: str, prompt_style: str) -> SyntheticExample:
        values = rng.sample(range(1, 30), 4 if difficulty != "hard" else 5)
        rule = rng.choice(["ascending", "descending"])
        prompt = f"[permutation_sorting] Sort numbers {rule}. Input: {','.join(str(v) for v in values)}"
        return _example_from_result(self.solve(prompt), difficulty, prompt_style)

    def solve(self, prompt: str) -> TeacherResult:
        lower = prompt.lower()
        nums = [int(value) for value in re.findall(r"-?\d+", prompt.split("Input:")[-1])]
        if len(nums) != len(set(nums)):
            return _fail(self.family, prompt, "ties_without_explicit_policy", "sort")
        if "ascending" in lower:
            ordered = sorted(nums)
            rule = "ascending"
        elif "descending" in lower:
            ordered = sorted(nums, reverse=True)
            rule = "descending"
        else:
            return _fail(self.family, prompt, "ambiguous_ordering", "sort")
        answer = ",".join(str(value) for value in ordered)
        trace = _trace([f"The rule sorts the numbers {rule}.", f"The normalized output is {answer}."], answer)
        return _pass(self.family, prompt, answer, trace, f"sort_{rule}", {})

    def verify_example(self, example: SyntheticExample) -> TeacherResult:
        return _verify_generated(self, example)

    def generate_hard_negative(self, example: SyntheticExample, rng: random.Random) -> HardNegative:
        wrong = ",".join(reversed(example.answer.split(",")))
        rejected = _trace(["Sort in the wrong direction.", f"This gives {wrong}."], wrong)
        return _negative_from_trace(example, rejected, wrong, "wrong_sort_direction")


class EquationOperatorTeacher:
    family = "equation_operator"
    rule_ids = ("add", "sub_ab", "sub_ba", "mul", "exact_div_ab", "exact_div_ba", "mod_ab", "mod_ba", "min", "max", "abs_diff", "affine_small")

    def generate_synthetic(self, rng: random.Random, difficulty: str, prompt_style: str) -> SyntheticExample:
        for _attempt in range(100):
            rule = rng.choice(self.rule_ids)
            params = _equation_params(rule, rng)
            pairs = _equation_pairs(rule, params, rng, 5 if difficulty != "easy" else 4)
            query = _equation_pair(rule, params, rng)
            examples = "; ".join(f"{a} @ {b} = {_apply_equation(rule, a, b, params)}" for a, b in pairs)
            prompt = f"[equation_operator] Rule: {rule}. Examples: {examples}. Query: {query[0]} @ {query[1]}?"
            result = self.solve(prompt)
            if result.verification_status == "PASS" and result.ambiguity_count == 0:
                return _example_from_result(result, difficulty, prompt_style)
        raise ValueError("Unable to generate safe equation_operator example")

    def solve(self, prompt: str) -> TeacherResult:
        examples = [(int(a), int(b), int(c)) for a, b, c in re.findall(r"(-?\d+)\s*@\s*(-?\d+)\s*=\s*(-?\d+)", prompt)]
        query_match = re.search(r"Query:\s*(-?\d+)\s*@\s*(-?\d+)\?", prompt)
        if not examples or query_match is None:
            return _fail(self.family, prompt, "malformed_operator_examples_or_query", "equation_operator")
        qa, qb = int(query_match.group(1)), int(query_match.group(2))
        explicit = re.search(r"Rule:\s*([a-z_]+)", prompt)
        candidate_outputs: dict[str, int] = {}
        rules = [explicit.group(1)] if explicit else list(self.rule_ids)
        for rule in rules:
            if rule not in self.rule_ids:
                continue
            param_candidates = _candidate_equation_params(rule, examples)
            for params in param_candidates:
                try:
                    if all(_apply_equation(rule, a, b, params) == y for a, b, y in examples):
                        candidate_outputs[_equation_key(rule, params)] = _apply_equation(rule, qa, qb, params)
                except ZeroDivisionError:
                    continue
        if not candidate_outputs:
            return _fail(self.family, prompt, "no_verified_operator_candidate", "equation_operator")
        if len(candidate_outputs) != 1:
            return _fail(self.family, prompt, "ambiguous_operator_candidates", "equation_operator")
        rule_key, answer_value = next(iter(candidate_outputs.items()))
        answer = str(answer_value)
        trace = _trace([f"The operator rule is {rule_key}.", f"{qa} @ {qb} = {answer}."], answer)
        return _pass(self.family, prompt, answer, trace, rule_key, {"candidate_count": 1})

    def verify_example(self, example: SyntheticExample) -> TeacherResult:
        return _verify_generated(self, example)

    def generate_hard_negative(self, example: SyntheticExample, rng: random.Random) -> HardNegative:
        wrong = str(int(example.answer) + rng.choice([-2, -1, 1, 2]))
        rejected = _trace(["Use the wrong operator.", f"This gives {wrong}."], wrong)
        return _negative_from_trace(example, rejected, wrong, "wrong_operator")


class SequencePatternTeacher:
    family = "sequence_pattern"
    rule_ids = ("arithmetic", "geometric_integer", "second_order_arithmetic", "fibonacci_like", "alternating_arithmetic_two_stream")

    def generate_synthetic(self, rng: random.Random, difficulty: str, prompt_style: str) -> SyntheticExample:
        for _attempt in range(100):
            rule = rng.choice(self.rule_ids)
            seq = _generate_sequence(rule, rng)
            prompt = f"[sequence_pattern] Rule: {rule}. Sequence: {', '.join(str(v) for v in seq)}, ?"
            result = self.solve(prompt)
            if result.verification_status == "PASS" and result.ambiguity_count == 0:
                return _example_from_result(result, difficulty, prompt_style)
        raise ValueError("Unable to generate safe sequence_pattern example")

    def solve(self, prompt: str) -> TeacherResult:
        if "?" not in prompt:
            return _fail(self.family, prompt, "missing_next_term_query", "sequence")
        seq_part = prompt.split("?")[0]
        seq = [int(value) for value in re.findall(r"-?\d+", seq_part.split("Sequence:")[-1])]
        explicit = re.search(r"Rule:\s*([a-z_]+)", prompt)
        if len(seq) < 5:
            return _fail(self.family, prompt, "too_short_sequence", "sequence")
        rules = [explicit.group(1)] if explicit else list(self.rule_ids)
        candidates: dict[str, int] = {}
        for rule in rules:
            prediction = _sequence_prediction(rule, seq)
            if prediction is not None:
                candidates[rule] = prediction
        if not candidates:
            return _fail(self.family, prompt, "no_verified_sequence_candidate", "sequence")
        if len(candidates) != 1:
            return _fail(self.family, prompt, "ambiguous_sequence_candidates", "sequence")
        rule, answer_value = next(iter(candidates.items()))
        answer = str(answer_value)
        trace = _trace([f"The sequence rule is {rule}.", f"The next term is {answer}."], answer)
        return _pass(self.family, prompt, answer, trace, rule, {"candidate_count": 1})

    def verify_example(self, example: SyntheticExample) -> TeacherResult:
        return _verify_generated(self, example)

    def generate_hard_negative(self, example: SyntheticExample, rng: random.Random) -> HardNegative:
        wrong = str(int(example.answer) + rng.choice([-2, -1, 1, 2]))
        rejected = _trace(["Use the wrong continuation.", f"This gives {wrong}."], wrong)
        return _negative_from_trace(example, rejected, wrong, "wrong_sequence_next_term")


class GravityNumericTeacher:
    family = "gravity_numeric"
    rule_ids = ("force_mass_gravity", "distance_speed_time", "speed_distance_time_exact", "time_distance_speed_exact")

    def generate_synthetic(self, rng: random.Random, difficulty: str, prompt_style: str) -> SyntheticExample:
        rule = rng.choice(self.rule_ids)
        if rule == "force_mass_gravity":
            g = rng.choice([2, 5, 10, 12])
            m = rng.randint(1, 300)
            prompt = f"[gravity_numeric] Use F = m*g. Let g = {g}. If m = {m}, what is F?"
        elif rule == "distance_speed_time":
            speed = rng.randint(2, 200)
            time = rng.randint(2, 80)
            prompt = f"[gravity_numeric] Use distance = speed*time. If speed = {speed} and time = {time}, what is distance?"
        elif rule == "speed_distance_time_exact":
            time = rng.randint(2, 80)
            speed = rng.randint(2, 200)
            distance = speed * time
            prompt = f"[gravity_numeric] Use speed = distance/time. If distance = {distance} and time = {time}, what is speed?"
        else:
            speed = rng.randint(2, 200)
            time = rng.randint(2, 80)
            distance = speed * time
            prompt = f"[gravity_numeric] Use time = distance/speed. If distance = {distance} and speed = {speed}, what is time?"
        return _example_from_result(self.solve(prompt), difficulty, prompt_style)

    def solve(self, prompt: str) -> TeacherResult:
        if "9.8" in prompt or re.search(r"\d+\.\d+", prompt):
            return _fail(self.family, prompt, "rounding_or_decimal_gravity_rejected", "gravity")
        lower = prompt.lower()
        if "f = m*g" in lower or "weight = mass * gravity" in lower:
            g = _named_int(prompt, "g")
            m = _named_int(prompt, "m")
            if g is None or m is None:
                return _fail(self.family, prompt, "missing_explicit_gravity_or_mass", "force_mass_gravity")
            answer = str(m * g)
            trace = _trace([f"Use F = m*g with g = {g}.", f"F = {m}*{g} = {answer}."], answer)
            return _pass(self.family, prompt, answer, trace, "force_mass_gravity", {"g": g, "m": m})
        if "distance = speed*time" in lower:
            speed = _named_int(prompt, "speed")
            time = _named_int(prompt, "time")
            if speed is None or time is None:
                return _fail(self.family, prompt, "missing_speed_or_time", "distance_speed_time")
            answer = str(speed * time)
            trace = _trace(["Use distance = speed*time.", f"distance = {speed}*{time} = {answer}."], answer)
            return _pass(self.family, prompt, answer, trace, "distance_speed_time", {})
        if "speed = distance/time" in lower:
            distance = _named_int(prompt, "distance")
            time = _named_int(prompt, "time")
            if distance is None or time is None or time == 0 or distance % time != 0:
                return _fail(self.family, prompt, "non_exact_speed_or_missing_values", "speed_distance_time_exact")
            answer = str(distance // time)
            trace = _trace(["Use speed = distance/time.", f"speed = {distance}/{time} = {answer}."], answer)
            return _pass(self.family, prompt, answer, trace, "speed_distance_time_exact", {})
        if "time = distance/speed" in lower:
            distance = _named_int(prompt, "distance")
            speed = _named_int(prompt, "speed")
            if distance is None or speed is None or speed == 0 or distance % speed != 0:
                return _fail(self.family, prompt, "non_exact_time_or_missing_values", "time_distance_speed_exact")
            answer = str(distance // speed)
            trace = _trace(["Use time = distance/speed.", f"time = {distance}/{speed} = {answer}."], answer)
            return _pass(self.family, prompt, answer, trace, "time_distance_speed_exact", {})
        return _fail(self.family, prompt, "unsupported_or_ambiguous_physics_prompt", "gravity")

    def verify_example(self, example: SyntheticExample) -> TeacherResult:
        return _verify_generated(self, example)

    def generate_hard_negative(self, example: SyntheticExample, rng: random.Random) -> HardNegative:
        wrong = str(int(example.answer) * 10 if example.answer != "0" else 10)
        rejected = _trace(["Use the wrong constant or inverse formula.", f"This gives {wrong}."], wrong)
        return _negative_from_trace(example, rejected, wrong, "wrong_physics_formula")


def get_teacher(family: str) -> FamilyTeacher:
    aliases = {"numeric_formula": "numeric_formula_safe"}
    key = aliases.get(family, family)
    teachers = available_teachers()
    if key not in teachers:
        raise KeyError(f"No Day1 Phase2 teacher implemented for family: {family}")
    return teachers[key]


def available_teachers() -> dict[str, FamilyTeacher]:
    return {
        "symbol_mapping": SymbolMappingTeacher(),
        "bit_manipulation": BitManipulationTeacher(),
        "char_cipher": CharCipherTeacher(),
        "unit_conversion": UnitConversionTeacher(),
        "numeric_formula_safe": NumericFormulaSafeTeacher(),
        "word_cipher": WordCipherTeacher(),
        "custom_numeral": CustomNumeralTeacher(),
        "permutation_sorting": PermutationSortingTeacher(),
        "equation_operator": EquationOperatorTeacher(),
        "sequence_pattern": SequencePatternTeacher(),
        "gravity_numeric": GravityNumericTeacher(),
    }


def generate_family_examples(
    family: str,
    n: int,
    seed: int,
    difficulties: list[str] | tuple[str, ...],
    prompt_styles: list[str] | tuple[str, ...],
) -> list[SyntheticExample]:
    rng = random.Random(seed)
    teacher = get_teacher(family)
    examples = []
    for index in range(n):
        difficulty = difficulties[index % len(difficulties)]
        prompt_style = prompt_styles[index % len(prompt_styles)]
        examples.append(teacher.generate_synthetic(rng, difficulty, prompt_style))
    return examples


def _format_mapping_prompt(family: str, mapping: dict[str, str], query: str, prompt_style: str) -> str:
    pairs = "; ".join(f"{key}->{value}" for key, value in mapping.items())
    if prompt_style == "table":
        return f"[{family}] Table mapping: {pairs}. Query: {query}"
    if prompt_style == "story":
        return f"[{family}] A codebook says {pairs}. Decode the query {query}."
    return f"[{family}] Examples: {pairs}. Query: {query}"


def _parse_mapping_prompt(prompt: str) -> tuple[dict[str, str], str] | None:
    query_match = re.search(r"(?:Query:|query)\s*([^\s.]+)", prompt)
    if query_match is None:
        query_match = re.search(r"query\s+([^\s.]+)", prompt, re.IGNORECASE)
    if query_match is None:
        return None
    query = query_match.group(1)
    prefix = prompt[: query_match.start()]
    pairs = re.findall(r"([^\s;,.]+)->([^\s;,.]+)", prefix)
    if not pairs:
        return None
    mapping: dict[str, str] = {}
    for key, value in pairs:
        if key in mapping and mapping[key] != value:
            return None
        mapping[key] = value
    return mapping, query


def _negative_from_trace(example: SyntheticExample, rejected: str, rejected_answer: str | None, reason: str) -> HardNegative:
    ok, _why = validate_target_text(rejected, example.answer)
    return HardNegative(
        family=example.family,
        prompt=example.prompt,
        chosen=example.target_text,
        rejected=rejected,
        correct_answer=example.answer,
        rejected_answer=rejected_answer,
        reason_rejected=reason,
        verifier_status="FAIL" if not ok else "FAIL_EXPECTED_WRONG_ANSWER",
        metadata={"rule_id": example.rule_id},
    )


def _bits(value: int, width: int) -> str:
    return format(value, f"0{width}b")


def _last_int(text: str) -> int | None:
    matches = re.findall(r"-?\d+", text)
    return int(matches[-1]) if matches else None


def _shift_amount(text: str) -> int:
    match = re.search(r"shift by (\d+)", text, re.IGNORECASE)
    return int(match.group(1)) if match else 1


def _random_word(rng: random.Random, min_len: int, max_len: int) -> str:
    letters = "abcdefghijklmnopqrstuvwxyz"
    return "".join(rng.choice(letters) for _ in range(rng.randint(min_len, max_len)))


def _caesar(word: str, shift: int) -> str:
    return "".join(chr((ord(ch) - 97 + shift) % 26 + 97) for ch in word)


def _atbash(word: str) -> str:
    return "".join(chr(122 - (ord(ch) - 97)) for ch in word)


def _fit_linear(pairs: list[tuple[int, int]]) -> tuple[int, int] | None:
    x1, y1 = pairs[0]
    x2, y2 = pairs[1]
    if x2 == x1 or (y2 - y1) % (x2 - x1) != 0:
        return None
    a = (y2 - y1) // (x2 - x1)
    b = y1 - a * x1
    return (a, b) if all(a * x + b == y for x, y in pairs) else None


def _fit_quadratic(pairs: list[tuple[int, int]]) -> tuple[int, int, int] | None:
    if len(pairs) < 3:
        return None
    (x1, y1), (x2, y2), (x3, y3) = pairs[:3]
    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    if denom == 0:
        return None
    a_num = x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)
    b_num = x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)
    c_num = x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3
    if a_num % denom or b_num % denom or c_num % denom:
        return None
    a, b, c = a_num // denom, b_num // denom, c_num // denom
    return (a, b, c) if all(a * x * x + b * x + c == y for x, y in pairs) else None


def _encode_custom(value: int, alphabet: list[str]) -> str:
    base = len(alphabet)
    if value == 0:
        return alphabet[0]
    digits = []
    current = value
    while current:
        current, rem = divmod(current, base)
        digits.append(alphabet[rem])
    return "".join(reversed(digits))


def _equation_params(rule: str, rng: random.Random) -> tuple[int, ...]:
    if rule == "affine_small":
        while True:
            params = (rng.randint(-3, 3), rng.randint(-3, 3), rng.randint(-3, 3))
            if params[1] != 0 or params[2] != 0:
                return params
    return ()


def _equation_pair(rule: str, params: tuple[int, ...], rng: random.Random) -> tuple[int, int]:
    del params
    if rule == "exact_div_ab":
        b = rng.randint(1, 9)
        q = rng.randint(1, 12)
        return b * q, b
    if rule == "exact_div_ba":
        a = rng.randint(1, 9)
        q = rng.randint(1, 12)
        return a, a * q
    if rule == "mod_ab":
        return rng.randint(1, 30), rng.randint(1, 9)
    if rule == "mod_ba":
        return rng.randint(1, 9), rng.randint(1, 30)
    return rng.randint(-9, 14), rng.randint(-9, 14)


def _equation_pairs(rule: str, params: tuple[int, ...], rng: random.Random, n: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    while len(pairs) < n:
        pair = _equation_pair(rule, params, rng)
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append(pair)
    return pairs


def _apply_equation(rule: str, a: int, b: int, params: tuple[int, ...]) -> int:
    if rule == "add":
        return a + b
    if rule == "sub_ab":
        return a - b
    if rule == "sub_ba":
        return b - a
    if rule == "mul":
        return a * b
    if rule == "exact_div_ab":
        if b == 0 or a % b != 0:
            raise ZeroDivisionError("non_exact_div_ab")
        return a // b
    if rule == "exact_div_ba":
        if a == 0 or b % a != 0:
            raise ZeroDivisionError("non_exact_div_ba")
        return b // a
    if rule == "mod_ab":
        if b == 0:
            raise ZeroDivisionError("mod_zero_ab")
        return a % b
    if rule == "mod_ba":
        if a == 0:
            raise ZeroDivisionError("mod_zero_ba")
        return b % a
    if rule == "min":
        return min(a, b)
    if rule == "max":
        return max(a, b)
    if rule == "abs_diff":
        return abs(a - b)
    if rule == "affine_small":
        c0, c1, c2 = params
        return c0 + c1 * a + c2 * b
    raise ValueError(f"Unsupported equation rule: {rule}")


def _candidate_equation_params(rule: str, examples: list[tuple[int, int, int]]) -> list[tuple[int, ...]]:
    if rule != "affine_small":
        return [()]
    candidates = []
    for c0 in range(-3, 4):
        for c1 in range(-3, 4):
            for c2 in range(-3, 4):
                if c1 == 0 and c2 == 0:
                    continue
                params = (c0, c1, c2)
                if all(_apply_equation(rule, a, b, params) == y for a, b, y in examples):
                    candidates.append(params)
    return candidates


def _equation_key(rule: str, params: tuple[int, ...]) -> str:
    if rule == "affine_small":
        return f"affine_small:{params[0]}:{params[1]}:{params[2]}"
    return rule


def _generate_sequence(rule: str, rng: random.Random) -> list[int]:
    if rule == "arithmetic":
        start = rng.randint(-20, 20)
        step = rng.choice([value for value in range(-7, 8) if value != 0])
        return [start + index * step for index in range(5)]
    if rule == "geometric_integer":
        start = rng.choice([value for value in range(-5, 6) if value not in {0}])
        ratio = rng.choice([-3, -2, 2, 3])
        return [start * (ratio**index) for index in range(5)]
    if rule == "second_order_arithmetic":
        first = rng.randint(-10, 10)
        diff = rng.randint(-5, 5)
        second_diff = rng.choice([value for value in range(-4, 5) if value != 0])
        seq = [first]
        current = first
        delta = diff
        for _ in range(5):
            current += delta
            seq.append(current)
            delta += second_diff
        return seq
    if rule == "fibonacci_like":
        a = rng.randint(1, 6)
        b = rng.randint(1, 8)
        seq = [a, b]
        while len(seq) < 7:
            seq.append(seq[-1] + seq[-2])
        return seq
    if rule == "alternating_arithmetic_two_stream":
        even_start = rng.randint(-10, 10)
        odd_start = rng.randint(-10, 10)
        even_step = rng.choice([value for value in range(-5, 6) if value != 0])
        odd_step = rng.choice([value for value in range(-5, 6) if value != 0 and value != even_step])
        seq = []
        for index in range(8):
            if index % 2 == 0:
                seq.append(even_start + (index // 2) * even_step)
            else:
                seq.append(odd_start + (index // 2) * odd_step)
        return seq
    raise ValueError(f"Unsupported sequence rule: {rule}")


def _sequence_prediction(rule: str, seq: list[int]) -> int | None:
    if rule == "arithmetic":
        if len(seq) < 5:
            return None
        step = seq[1] - seq[0]
        return seq[-1] + step if all(seq[i] - seq[i - 1] == step for i in range(1, len(seq))) else None
    if rule == "geometric_integer":
        if len(seq) < 5 or seq[0] == 0:
            return None
        if seq[1] % seq[0] != 0:
            return None
        ratio = seq[1] // seq[0]
        if ratio in {0, 1, -1}:
            return None
        return seq[-1] * ratio if all(seq[i - 1] != 0 and seq[i] == seq[i - 1] * ratio for i in range(1, len(seq))) else None
    if rule == "second_order_arithmetic":
        if len(seq) < 6:
            return None
        diffs = [seq[i] - seq[i - 1] for i in range(1, len(seq))]
        second = diffs[1] - diffs[0]
        if second == 0:
            return None
        return seq[-1] + diffs[-1] + second if all(diffs[i] - diffs[i - 1] == second for i in range(1, len(diffs))) else None
    if rule == "fibonacci_like":
        if len(seq) < 7:
            return None
        return seq[-1] + seq[-2] if all(seq[i] == seq[i - 1] + seq[i - 2] for i in range(2, len(seq))) else None
    if rule == "alternating_arithmetic_two_stream":
        if len(seq) < 8:
            return None
        evens = seq[0::2]
        odds = seq[1::2]
        even_step = evens[1] - evens[0]
        odd_step = odds[1] - odds[0]
        if not all(evens[i] - evens[i - 1] == even_step for i in range(1, len(evens))):
            return None
        if not all(odds[i] - odds[i - 1] == odd_step for i in range(1, len(odds))):
            return None
        return evens[-1] + even_step if len(seq) % 2 == 0 else odds[-1] + odd_step
    return None


def _named_int(prompt: str, name: str) -> int | None:
    match = re.search(rf"\b{name}\s*=\s*(-?\d+)\b", prompt, re.IGNORECASE)
    return int(match.group(1)) if match else None
