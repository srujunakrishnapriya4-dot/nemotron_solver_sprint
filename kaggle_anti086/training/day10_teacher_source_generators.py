from __future__ import annotations

from decimal import Decimal
import hashlib
import math
import random
from typing import Any

from kaggle_anti086.solvers.bit_transform_solver import _format_bits, _rol
from kaggle_anti086.solvers.numeric_parsing import format_decimal
from kaggle_anti086.solvers.roman_solver import int_to_roman


GENERATION_POLICY_VERSION = "day10_pass10c_v1"

DIRECT_TARGET_COUNTS = {
    "bit_manipulation": 900,
    "symbol_mapping": 850,
    "char_cipher": 850,
    "unit_conversion": 750,
    "numeric_formula": 750,
    "gravity_numeric": 550,
    "word_cipher": 650,
    "roman_numeral": 544,
    "format_only": 300,
}

ABSTAIN_TARGET_COUNTS = {
    "custom_numeral": 128,
    "sequence_pattern": 128,
    "permutation_sorting": 128,
    "equation_operator": 128,
}


def build_day10_repair_source_rows(
    *,
    target_direct_rows: int = 6144,
    target_abstain_rows: int = 512,
    seed: int = 1110,
    attempt_multiplier: int = 2,
) -> dict[str, Any]:
    direct_counts = _scaled_counts(DIRECT_TARGET_COUNTS, target_direct_rows)
    abstain_counts = _scaled_counts(ABSTAIN_TARGET_COUNTS, target_abstain_rows)
    direct_rows: list[dict[str, Any]] = []
    abstain_rows: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for family, count in direct_counts.items():
        direct_rows.extend(_generate_family(family, count * max(attempt_multiplier, 6), rng))
    for family, count in abstain_counts.items():
        abstain_rows.extend(_generate_abstain_family(family, count, rng))
    return {
        "generation_policy_version": GENERATION_POLICY_VERSION,
        "seed": seed,
        "source_mode": "day10_repair",
        "target_direct_rows": target_direct_rows,
        "target_abstain_rows": target_abstain_rows,
        "direct_target_counts": direct_counts,
        "abstain_target_counts": abstain_counts,
        "direct_rows": _interleave_by_family(direct_rows),
        "abstain_rows": _interleave_by_family(abstain_rows),
    }


def parameter_tuple_hash(value: Any) -> str:
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()


def _generate_family(family: str, count: int, rng: random.Random) -> list[dict[str, Any]]:
    builders = {
        "numeric_formula": _numeric_row,
        "gravity_numeric": _gravity_row,
        "unit_conversion": _unit_row,
        "roman_numeral": _roman_row,
        "bit_manipulation": _bit_row,
        "symbol_mapping": _symbol_row,
        "char_cipher": _char_row,
        "word_cipher": _word_row,
        "format_only": _format_only_row,
    }
    return [builders[family](i, rng) for i in range(count)]


def _generate_abstain_family(family: str, count: int, rng: random.Random) -> list[dict[str, Any]]:
    rows = []
    for i in range(count):
        prompts = {
            "custom_numeral": f"custom glyph set {i}: one -> @; two -> &&. target: three",
            "sequence_pattern": f"sequence rule {i}: 2, 4, 8, 16, ?",
            "permutation_sorting": f"hidden sorting rule {i}: cab -> abc; bca -> abc; query: dac",
            "equation_operator": f"operator rule {i}: A @ B -> C. infer A # B?",
        }
        rows.append(
            _row(
                family=family,
                subfamily="expected_abstain",
                i=i,
                prompt=_wrap_prompt(prompts[family], i, family),
                answer="ABSTAIN",
                behavior="abstain",
                parameter_tuple=(family, "abstain", i),
                answer_type="generic",
            )
        )
    return rows


def _numeric_row(i: int, rng: random.Random) -> dict[str, Any]:
    q = 5 + i % 97
    mode = i % 4
    if mode == 0:
        a = 2 + i % 11
        b = 1 + (i * 3) % 17
        prompt = f"1 -> {a + b}; 2 -> {2 * a + b}; 3 -> {3 * a + b}. Now solve: {q}"
        answer = str(a * q + b)
        sub = "linear_offset"
        params = ("linear", a, b, q)
    elif mode == 1:
        a = 1 + i % 7
        prompt = f"1 -> {a}; 2 -> {4 * a}; 3 -> {9 * a}. Query {q}?"
        answer = str(a * q * q)
        sub = "quadratic_scale"
        params = ("quadratic", a, q)
    elif mode == 2:
        a = 2 + i % 9
        b = -1 - i % 13
        prompt = f"1 -> {a + b}; 2 -> {2 * a + b}; 4 -> {4 * a + b}. Now solve: {q}"
        answer = str(a * q + b)
        sub = "linear_negative_offset"
        params = ("linear_negative", a, b, q)
    else:
        a = Decimal(2 + i % 8) / Decimal("2")
        b = Decimal(i % 5)
        prompt = f"2 -> {format_decimal(a * 2 + b, 1)}; 4 -> {format_decimal(a * 4 + b, 1)}; 6 -> {format_decimal(a * 6 + b, 1)}. Query {q}?"
        answer = format_decimal(a * Decimal(q) + b, 1)
        sub = "decimal_linear"
        params = ("decimal_linear", str(a), str(b), q)
    return _row("numeric_formula", sub, i, _wrap_prompt(prompt, i, "numeric_formula"), answer, "answer", params, "numeric")


def _gravity_row(i: int, rng: random.Random) -> dict[str, Any]:
    t = 4 + i % 80
    half_g = Decimal("4.80") + Decimal(i % 31) / Decimal("100")
    prompt = f"time 1 -> {format_decimal(half_g, 2)}; time 2 -> {format_decimal(half_g * 4, 2)}; time 3 -> {format_decimal(half_g * 9, 2)}. Query time {t}?"
    answer = format_decimal(half_g * Decimal(t * t), 2)
    return _row("gravity_numeric", "gravity_distance", i, _wrap_prompt(prompt, i, "gravity_numeric"), answer, "answer", ("gravity", str(half_g), t), "numeric")


def _unit_row(i: int, rng: random.Random) -> dict[str, Any]:
    q = Decimal(7 + i % 93)
    mode = i % 3
    if mode == 0:
        scale = Decimal("1.20") + Decimal(i % 41) / Decimal("100")
        prompt = f"2 -> {format_decimal(Decimal(2) * scale, 2)}; 4 -> {format_decimal(Decimal(4) * scale, 2)}; 6 -> {format_decimal(Decimal(6) * scale, 2)}. Convert {q}"
        answer = format_decimal(q * scale, 2)
        sub = "multiplicative_scale_round_2"
        params = ("scale", str(scale), str(q))
    elif mode == 1:
        offset = Decimal(i % 19 - 5)
        prompt = f"10 -> {format_decimal(Decimal(10) + offset, 0)}; 20 -> {format_decimal(Decimal(20) + offset, 0)}; 30 -> {format_decimal(Decimal(30) + offset, 0)}. Input: {q}"
        answer = format_decimal(q + offset, 0)
        sub = "linear_offset_round_0"
        params = ("offset", str(offset), str(q))
    else:
        scale = Decimal("0.75") + Decimal(i % 17) / Decimal("100")
        prompt = f"8 -> {format_decimal(Decimal(8) * scale, 3)}; 12 -> {format_decimal(Decimal(12) * scale, 3)}; 16 -> {format_decimal(Decimal(16) * scale, 3)}. Convert {q}"
        answer = format_decimal(q * scale, 3)
        sub = "multiplicative_scale_round_3"
        params = ("scale3", str(scale), str(q))
    return _row("unit_conversion", sub, i, _wrap_prompt(prompt, i, "unit_conversion"), answer, "answer", params, "numeric")


def _roman_row(i: int, rng: random.Random) -> dict[str, Any]:
    n = 1 + ((i * 37 + 43) % 3999)
    prompt = f"11 -> XI; 15 -> XV. solve for {n}"
    return _row("roman_numeral", "standard_roman", i, _wrap_prompt(prompt, i, "roman_numeral"), int_to_roman(n), "answer", ("roman", n), "roman")


def _bit_row(i: int, rng: random.Random) -> dict[str, Any]:
    width = 16 if i % 3 == 0 else 8
    modulus = 1 << width
    mask_limit = modulus - 1
    q = _format_bits((i * 1543 + 19) % modulus, width)
    inputs = [_format_bits((i * 929 + j * 1549 + 7) % modulus, width) for j in range(4)]
    mode = i % 4
    if mode == 0:
        mask = ((0xA5A5 if width == 16 else 0xA5) ^ (i * 37)) & mask_limit
        examples = [(bits, _format_bits(int(bits, 2) ^ mask, width)) for bits in inputs]
        answer = _format_bits(int(q, 2) ^ mask, width)
        sub = "xor_mask"
        params = ("xor", width, mask, q)
    elif mode == 1:
        examples = [(bits, bits[::-1]) for bits in inputs]
        answer = q[::-1]
        sub = "reverse_bits"
        params = ("reverse", width, q)
    elif mode == 2:
        k = 1 + i % (width - 1)
        examples = [(bits, _rol(bits, k)) for bits in inputs]
        answer = _rol(q, k)
        sub = "rotate_left"
        params = ("rol", width, k, q)
    else:
        k = 1 + i % min(7, width - 1)
        mask = ((0x3333 if width == 16 else 0x33) + i * 17) & mask_limit
        examples = [(bits, _format_bits(int(_rol(bits, k), 2) ^ mask, width)) for bits in inputs]
        answer = _format_bits(int(_rol(q, k), 2) ^ mask, width)
        sub = "rotate_then_xor"
        params = ("rol_xor", width, k, mask, q)
    prompt = "; ".join(f"{a} -> {b}" for a, b in examples) + f". Input: {q}"
    return _row("bit_manipulation", sub, i, _wrap_prompt(prompt, i, "bit_manipulation"), answer, "answer", params, "binary")


def _symbol_row(i: int, rng: random.Random) -> dict[str, Any]:
    alphabets = ("!@#$%^&*", "[]{}()_~", "01234567")
    output_chars = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnpqrstuvwxyz23456789"
    src = alphabets[i % len(alphabets)]
    step = 1 + ((i // len(alphabets)) % 17)
    while math.gcd(step, len(output_chars)) != 1:
        step += 1
    offset = (i * 11 + i // 3) % len(output_chars)
    dst = "".join(output_chars[(offset + step * pos) % len(output_chars)] for pos in range(8))
    mapping = dict(zip(src, dst))
    query_len = 3 + (i % 2)
    query = "".join(src[(i + 1 + step * 2) % 8] for step in range(query_len))
    reverse = i % 5 == 0

    def tr(text: str) -> str:
        base = text[::-1] if reverse else text
        return "".join(mapping[ch] for ch in base)

    examples = [(src[0:4], tr(src[0:4])), (src[4:8], tr(src[4:8])), (src[1:5], tr(src[1:5]))]
    prompt = "; ".join(f"{a} -> {b}" for a, b in examples) + f"; query: {query} output ?"
    return _row(
        "symbol_mapping",
        "reversal_substitution" if reverse else "char_substitution",
        i,
        _wrap_prompt(prompt, i, "symbol_mapping"),
        tr(query),
        "answer",
        ("symbol", src, dst, query, reverse),
        "symbol",
    )


def _char_row(i: int, rng: random.Random) -> dict[str, Any]:
    words = tuple(_letter_word(j * 19 + i) for j in range(10))
    mode = i % 4
    q = words[i % len(words)]
    if mode in {0, 3}:
        shift = 1 + i % 17
        prompt = f"abc -> {_shift('abc', shift)}; xyz -> {_shift('xyz', shift)}; query: {q}"
        answer = _shift(q, shift)
        sub = "caesar_shift"
        params = ("caesar", shift, q)
    elif mode == 1:
        prompt = f"abc -> cba; book -> koob; query: {q}"
        answer = q[::-1]
        sub = "reverse_string"
        params = ("reverse", q)
    else:
        src = "abcdefghijklmnopqrstuvwxyz"
        shifted = _shift(src, 5 + i % 11)
        query = q[:3] + words[(i + 3) % len(words)][:3]
        mapping = dict(zip(src, shifted))
        prompt = f"abc -> {''.join(mapping[c] for c in 'abc')}; cat -> {''.join(mapping[c] for c in 'cat')}; query: {query}"
        answer = "".join(mapping[c] for c in query)
        sub = "monoalphabetic_substitution"
        params = ("mono", 5 + i % 11, query)
    return _row("char_cipher", sub, i, _wrap_prompt(prompt, i, "char_cipher"), answer, "answer", params, "text_phrase")


def _word_row(i: int, rng: random.Random) -> dict[str, Any]:
    enc = [_letter_word(i * 13 + j * 17) for j in range(6)]
    plain_bank = tuple(_letter_word(5000 + i * 101 + j * 23) for j in range(12))
    plain = [plain_bank[(i + j * 3) % len(plain_bank)] for j in range(6)]
    query_ids = (i % 6, (i + 2) % 6)
    encrypted_phrase = " ".join(enc)
    plain_phrase = " ".join(plain)
    query_phrase = f"{enc[query_ids[0]]} {enc[query_ids[1]]}"
    if i % 3 == 0:
        prompt = f'"{encrypted_phrase}" -> "{plain_phrase}"; encrypted: {query_phrase} plaintext: ?'
    elif i % 3 == 1:
        prompt = f"`{encrypted_phrase}` = `{plain_phrase}`; decode: {query_phrase}?"
    else:
        prompt = f"{encrypted_phrase} -> {plain_phrase}; decode: {query_phrase}?"
    answer = f"{plain[query_ids[0]]} {plain[query_ids[1]]}"
    return _row("word_cipher", "word_substitution", i, _wrap_prompt(prompt, i, "word_cipher"), answer, "text", ("word", tuple(enc), tuple(plain), query_ids), "text_phrase")


def _format_only_row(i: int, rng: random.Random) -> dict[str, Any]:
    cases = [
        ("boxed_answer", r"\boxed{%s}", "numeric", lambda j: format_decimal(Decimal(100 + j) / Decimal("3"), 2)),
        ("answer_prefix_strip", "Answer: %s.", "symbol", lambda j: ("@&", "#!", "[]", "{}")[j % 4]),
        ("think_suffix_strip", "</think>\n%s", "roman", lambda j: int_to_roman(1 + (j * 19) % 3999)),
        ("first_line_binary", "%s\nExplanation: copied scratch work.", "binary", lambda j: _format_bits((j * 11 + 5) % 256, 8)),
        ("text_phrase_trim", "The answer is: %s.", "text_phrase", lambda j: ("cat book", "blue stone", "silver map", "logic gate")[j % 4]),
        ("markdown_inline_trim", "`%s`", "numeric", lambda j: str(200 + j)),
        ("arrow_output", "38 -> %s", "roman", lambda j: int_to_roman(1 + (j * 7) % 3999)),
        ("symbol_trim", "formatted answer: '%s'", "symbol", lambda j: ("<>", "~=", "|:", "?!")[j % 4]),
        ("numeric_round_trim", "candidate = %s", "numeric", lambda j: format_decimal(Decimal(j * 17 + 9) / Decimal("8"), 3)),
        ("no_explanation_final_only", "output: %s", "text_phrase", lambda j: ("plain token", "final value", "short reply", "clean text")[j % 4]),
    ]
    sub, template, answer_type, fn = cases[i % len(cases)]
    expected = fn(i)
    raw = template % expected
    prompt = f"Format-only cleanup task. Raw output: {raw}\nExtract final answer only."
    return _row("format_only", sub, i, _wrap_prompt(prompt, i, "format_only"), expected, "answer", ("format", sub, raw), answer_type)


def _row(
    family: str,
    subfamily: str,
    i: int,
    prompt: str,
    answer: str,
    behavior: str,
    parameter_tuple: Any,
    answer_type: str,
) -> dict[str, Any]:
    parameter_hash = parameter_tuple_hash(parameter_tuple)
    template_id = f"day10_template_{family}_{i % 8}"
    rule_signature = f"day10_pass10c_{family}_{subfamily}_{parameter_hash[:16]}"
    return {
        "id": f"day10_source_{family}_{subfamily}_{i:05d}_{parameter_hash[:10]}",
        "family": family,
        "subfamily": subfamily,
        "rule_id": f"day10_rule_{family}_{subfamily}_{parameter_hash[:16]}",
        "prompt": prompt,
        "answer": answer,
        "source": "day10_pass10c_source_generator",
        "solver_name": "solver_ensemble",
        "verification_status": "verified",
        "difficulty": 3,
        "split": "train",
        "leakage_group": f"day10_lg_{family}_{parameter_hash[:16]}",
        "metadata": {
            "expected_solver_behavior": behavior,
            "rule_signature": rule_signature,
            "parameter_tuple": repr(parameter_tuple),
            "parameter_tuple_hash": parameter_hash,
            "prompt_template_id": template_id,
            "generator_id": f"day10_pass10c_{family}_gen_v1",
            "source_namespace": "day10_pass10c_repair",
            "answer_type": answer_type,
        },
    }


def _wrap_prompt(prompt: str, i: int, family: str) -> str:
    tag = _alpha_tag(i)
    if family in {"symbol_mapping", "word_cipher", "char_cipher"}:
        variants = (
            f"{prompt}\nMarker {tag}. Return only the final answer.",
            f"{prompt}\nOutput only the transformed value for marker {tag}.",
            f"{prompt}\nNo explanation. Marker {tag}.",
            f"{prompt}\nFinal answer only. Marker {tag}.",
            f"{prompt}\nBare answer required. Marker {tag}.",
            f"{prompt}\nMarker {tag}. No extra text.",
            f"{prompt}\nReturn exactly the solved token/string. Marker {tag}.",
            f"{prompt}\nMarker {tag}.",
        )
        return variants[i % len(variants)]
    variants = (
        f"Marker {tag}. Infer the transformation from examples. {prompt}",
        f"{prompt}\nOutput only the final transformed value for marker {tag}.",
        f"Examples define a hidden rule for {family}. Marker {tag}. {prompt}",
        f"{prompt}\nFor marker {tag}, no explanation; output the final answer token/string.",
        f"Use the examples and return only the answer. Marker {tag}. {prompt}",
        f"Final only task marker {tag}. {prompt}",
        f"{prompt}\nReturn a bare answer. Marker {tag}.",
        f"Marker {tag}. Transformation examples follow. {prompt}",
    )
    return variants[i % len(variants)]


def _scaled_counts(base: dict[str, int], rows: int) -> dict[str, int]:
    total = sum(base.values())
    scaled = {family: int(count * rows / total) for family, count in base.items()}
    while sum(scaled.values()) < rows:
        family = max(base, key=lambda item: (base[item] * rows / total) - scaled[item])
        scaled[family] += 1
    while sum(scaled.values()) > rows:
        family = max((item for item in scaled if scaled[item] > 0), key=lambda item: scaled[item])
        scaled[family] -= 1
    return scaled


def _interleave_by_family(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(str(row["family"]), []).append(row)
    ordered: list[dict[str, Any]] = []
    while any(buckets.values()):
        for family in sorted(buckets):
            if buckets[family]:
                ordered.append(buckets[family].pop(0))
    return ordered


def _shift(text: str, shift: int) -> str:
    output = []
    for char in text:
        if "a" <= char <= "z":
            output.append(chr(ord("a") + ((ord(char) - ord("a") + shift) % 26)))
        else:
            output.append(char)
    return "".join(output)


def _alpha_tag(idx: int) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    n = max(0, int(idx))
    chars = []
    while True:
        chars.append(alphabet[n % len(alphabet)])
        n = n // len(alphabet) - 1
        if n < 0:
            break
    return "tag" + "".join(reversed(chars))


def _letter_word(value: int) -> str:
    syllables = (
        "ba",
        "ce",
        "di",
        "fo",
        "gu",
        "ha",
        "jo",
        "ki",
        "lu",
        "me",
        "na",
        "po",
        "ri",
        "sa",
        "tu",
        "ve",
        "wo",
        "xi",
        "ya",
        "zo",
    )
    a = syllables[value % len(syllables)]
    b = syllables[(value // len(syllables) + 7) % len(syllables)]
    c = syllables[(value // (len(syllables) * 3) + 11) % len(syllables)]
    return a + b + c
