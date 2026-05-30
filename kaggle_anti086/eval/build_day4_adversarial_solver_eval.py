from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.schema import validate_rows
from kaggle_anti086.kaggle_path_safety import safe_write_text
from kaggle_anti086.solvers.bit_transform_solver import _format_bits, _rol
from kaggle_anti086.solvers.numeric_parsing import format_decimal
from kaggle_anti086.solvers.roman_solver import int_to_roman


NOISE_PROFILES = [
    "clean",
    "quoted_examples",
    "extra_explanation_text",
    "distractor_pair",
    "query_with_arrow",
    "query_with_colon",
    "whitespace_noise",
    "answer_only_style",
    "conflicting_examples",
    "unknown_symbols_or_words",
]


def build_rows(*, variant: str = "day4") -> list[dict]:
    rows: list[dict] = []
    rows += _bit_rows(variant)
    rows += _symbol_rows(variant)
    rows += _char_rows(variant)
    rows += _word_rows(variant)
    rows += _unit_rows(variant)
    rows += _numeric_gravity_rows(variant)
    rows += _roman_rows(variant)
    _assert_unique_fields(rows)
    return rows


def write_eval(path: str | Path, *, variant: str = "day4") -> list[dict]:
    rows = build_rows(variant=variant)
    report = validate_rows(rows, context=f"{variant}_adversarial")
    if report["failure_count"]:
        raise SystemExit(json.dumps(report, indent=2, sort_keys=True))
    safe_write_text(path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), field_name=f"{variant}_adversarial_solver_eval")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Day 4 adversarial deterministic solver eval.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--variant", default="day4", choices=["day4", "day4_1"])
    args = parser.parse_args(argv)
    rows = write_eval(args.out, variant=args.variant)
    print(json.dumps({"status": "WROTE", "row_count": len(rows), "variant": args.variant, "out": args.out}, sort_keys=True))
    return 0


def _base(row_id: str, family: str, subfamily: str, prompt: str, answer: str, behavior: str, rule_signature: str, *, variant: str, difficulty: int = 2, noise_profile: str = "clean", metadata: dict | None = None) -> dict:
    safe_signature = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in rule_signature)[:120]
    return {
        "id": row_id,
        "family": family,
        "subfamily": subfamily,
        "rule_id": f"{variant}_{row_id}_{safe_signature}",
        "prompt": prompt,
        "answer": answer,
        "source": f"{variant}_adversarial_solver_eval",
        "solver_name": "solver_ensemble",
        "verification_status": "verified",
        "difficulty": difficulty,
        "split": "dev",
        "leakage_group": f"{variant}_lg_{row_id}_{safe_signature}",
        "metadata": {
            "expected_solver_behavior": behavior,
            "rule_signature": rule_signature,
            "generator_id": f"{variant}_{family}_gen_v2",
            "difficulty_score": difficulty,
            "noise_profile": noise_profile,
        }
        | (metadata or {}),
    }


def _bit_rows(variant: str) -> list[dict]:
    rows: list[dict] = []
    seeds = list(range(42))
    for i in seeds:
        q = _format_bits((i * 41 + 23) % 256, 8)
        profile = NOISE_PROFILES[i % 8]
        if i % 6 == 0:
            mask = (0xA5 ^ (i * 3)) & 0xFF
            ex = [_bit_pair(x, int(x, 2) ^ mask) for x in ["00000000", "11111111", "01010101", "00110011"]]
            ans = _format_bits(int(q, 2) ^ mask, 8)
            sub = "xor_mask"
            sig = f"bit_xor_mask_seed_{i:03d}_mask_{mask:02x}"
        elif i % 6 == 1:
            ex = [(x, x[::-1]) for x in _bit_inputs(i, 4)]
            ans = q[::-1]
            sub = "reverse"
            sig = f"bit_reverse_seed_{i:03d}"
        elif i % 6 == 2:
            k = 1 + (i % 7)
            ex = [(x, _rol(x, k)) for x in _bit_inputs(i, 4)]
            ans = _rol(q, k)
            sub = "rotate_left"
            sig = f"bit_rotate_left_seed_{i:03d}_k_{k}"
        elif i % 6 == 3:
            k = 1 + (i % 5)
            mask = (0x33 + i * 5) & 0xFF
            ex = [(x, _format_bits(int(_rol(x, k), 2) ^ mask, 8)) for x in _bit_inputs(i, 4)]
            ans = _format_bits(int(_rol(q, k), 2) ^ mask, 8)
            sub = "rotate_then_xor"
            sig = f"bit_rotate_xor_seed_{i:03d}_k_{k}_mask_{mask:02x}"
        elif i % 6 == 4:
            perm = (2, 0, 1, 3, 6, 4, 7, 5)
            ex = [(x, _permute(x, perm)) for x in _permutation_basis_inputs()]
            ans = _permute(q, perm)
            sub = "bit_position_permutation"
            sig = f"bit_perm_seed_{i:03d}_perm_{'-'.join(map(str, perm))}"
        else:
            perm = (1, 0, 3, 2, 5, 4, 7, 6)
            mask = (0x0F ^ i) & 0xFF
            ex = [(x, _format_bits(int(_permute(x, perm), 2) ^ mask, 8)) for x in _permutation_basis_inputs()]
            ans = _format_bits(int(_permute(q, perm), 2) ^ mask, 8)
            sub = "bit_position_permutation_then_xor"
            sig = f"bit_perm_xor_seed_{i:03d}_mask_{mask:02x}"
        prompt = _bit_prompt(ex, q, profile, i)
        rows.append(_base(f"{variant}_bit_answer_{i:03d}", "bit_manipulation", sub, prompt, ans, "answer", sig, variant=variant, noise_profile=profile, difficulty=3))
    for i in range(14):
        profile = ["mixed_bit_widths", "insufficient_examples", "ambiguous_transform_candidates", "missing_query"][i % 4]
        if i % 4 == 0:
            prompt = f"Noise {i}: 0101 -> 1010; 11110000 -> 00001111. Input: 0101"
        elif i % 4 == 1:
            prompt = f"One example only {i}: 01010101 -> 10101010. Input: 11110000"
        elif i % 4 == 2:
            prompt = f"Ambiguous {i}: 0000 -> 0000; 1111 -> 1111; 0011 -> 1100. Input: 0101"
        else:
            prompt = f"Missing query {i}: 00001111 -> 11110000; 10100000 -> 00000101."
        rows.append(_base(f"{variant}_bit_abstain_{i:03d}", "bit_manipulation", "expected_abstain", prompt, "ABSTAIN", "abstain", f"bit_abstain_seed_{i:03d}_{profile}", variant=variant, difficulty=4, noise_profile=profile))
    return rows


def _symbol_rows(variant: str) -> list[dict]:
    rows: list[dict] = []
    alphabets = ["!@#$%^&*", "[]{}<>?/", "+-=~|:;.", "`'\",._"]
    targets = ["ABCDEFGH", "qrstuvwx", "01234567", "@&![]{}#"]
    for i in range(42):
        src = alphabets[i % len(alphabets)]
        dst = targets[i % len(targets)]
        mapping = {a: b for a, b in zip(src, dst)}
        query = src[1] + src[3] + src[5]
        profile = NOISE_PROFILES[(i + 2) % 8]
        reverse = i % 5 == 0
        left1 = src[0:3]
        left2 = src[3:6]
        right1 = "".join(mapping[ch] for ch in (left1[::-1] if reverse else left1))
        right2 = "".join(mapping[ch] for ch in (left2[::-1] if reverse else left2))
        answer = "".join(mapping[ch] for ch in (query[::-1] if reverse else query))
        prompt = _symbol_prompt([(left1, right1), (left2, right2), (src[1:4], "".join(mapping[ch] for ch in ((src[1:4])[::-1] if reverse else src[1:4])))], query, profile, i)
        sub = "reversal_substitution" if reverse else "char_substitution"
        sig = f"symbol_sub_perm_seed_{i:03d}_alphabet_{i % len(alphabets)}_reverse_{int(reverse)}"
        rows.append(_base(f"{variant}_symbol_answer_{i:03d}", "symbol_mapping", sub, prompt, answer, "answer", sig, variant=variant, difficulty=3, noise_profile=profile))
    for i in range(14):
        profile = ["conflicting_examples", "unknown_symbols_or_words", "unsupported_deletion", "compression_unsupported"][i % 4]
        if i % 4 == 0:
            prompt = f"!@ -> ab; !# -> xy; query: !# output ? // conflict case {i}"
        elif i % 4 == 1:
            prompt = f"!@ -> ab; #$ -> cd; query: !Z% output ? // unknown case {i}"
        elif i % 4 == 2:
            prompt = f"!*@ -> xy; #$% -> uv. query: !*@ output ? // deletion case {i}"
        else:
            prompt = f"!!@@ -> !@; ##$$ -> #$. query: !!@@ output ? // compression case {i}"
        rows.append(_base(f"{variant}_symbol_abstain_{i:03d}", "symbol_mapping", "expected_abstain", prompt, "ABSTAIN", "abstain", f"symbol_abstain_seed_{i:03d}_{profile}", variant=variant, difficulty=4, noise_profile=profile))
    return rows


def _char_rows(variant: str) -> list[dict]:
    rows: list[dict] = []
    caesar_queries = ["mno", "logic", "trace", "vector", "alpha", "reason"]
    for i in range(33):
        profile = NOISE_PROFILES[(i + 4) % 8]
        if i % 3 == 0:
            shift = 1 + (i % 12)
            q = caesar_queries[i % len(caesar_queries)]
            prompt = f"cipher examples seed {i}: abc -> {_shift('abc', shift)}; xyz -> {_shift('xyz', shift)}; query: {q}"
            answer = _shift(q, shift)
            sub = "caesar_shift"
            sig = f"char_caesar_seed_{i:03d}_shift_{shift}"
        elif i % 3 == 1:
            q = ["starlight", "vector", "matrix", "solver"][i % 4]
            prompt = f"reverse style {i}: abc -> cba; book -> koob; solve {q} -> ?"
            answer = q[::-1]
            sub = "reverse_string"
            sig = f"char_reverse_seed_{i:03d}"
        else:
            prompt = f"substitution block {i}: trb -> cat; hffk -> book; query: trbhffk"
            answer = "catbook"
            sub = "monoalphabetic_substitution"
            sig = f"char_mono_seed_{i:03d}_cat_book"
        rows.append(_base(f"{variant}_char_answer_{i:03d}", "char_cipher", sub, prompt, answer, "answer", sig, variant=variant, difficulty=3, noise_profile=profile))
    for i in range(11):
        profile = ["conflicting_examples", "unknown_symbols_or_words", "missing_query"][i % 3]
        if i % 3 == 0:
            prompt = f"abc -> def; abc -> xyz; query: abc // char conflict {i}"
        elif i % 3 == 1:
            prompt = f"trb -> cat; hffk -> book; query: zzz // unknown char {i}"
        else:
            prompt = f"abc -> def; xyz -> abc // no query {i}"
        rows.append(_base(f"{variant}_char_abstain_{i:03d}", "char_cipher", "expected_abstain", prompt, "ABSTAIN", "abstain", f"char_abstain_seed_{i:03d}_{profile}", variant=variant, difficulty=4, noise_profile=profile))
    return rows


def _word_rows(variant: str) -> list[dict]:
    rows: list[dict] = []
    dictionaries = [
        (["wkgqa", "lsrqaq", "wneeke", "nwrjnvaq"], ["mouse", "chases", "mirror", "imagines"]),
        (["brmrla", "nv", "zod", "plene"], ["palace", "in", "quiet", "garden"]),
        (["fai", "lume", "sora", "dax"], ["red", "blue", "green", "stone"]),
    ]
    for i in range(27):
        enc, plain = dictionaries[i % len(dictionaries)]
        if i % 4 == 0:
            prompt = f'word seed {i}: "{enc[0]} {enc[1]} {enc[2]}" -> "{plain[0]} {plain[1]} {plain[2]}"; encrypted: {enc[0]} {enc[2]} plaintext: ?'
            answer = f"{plain[0]} {plain[2]}"
            profile = "quoted_examples"
        elif i % 4 == 1:
            prompt = f"word seed {i}: `{enc[0]} {enc[3]}` = `{plain[0]} {plain[3]}`; `{enc[1]} {enc[2]}` = `{plain[1]} {plain[2]}`; decode: {enc[3]} {enc[0]}?"
            answer = f"{plain[3]} {plain[0]}"
            profile = "query_with_colon"
        elif i % 4 == 2:
            prompt = f"Ignore story seed {i}. encrypted: {enc[0]} {enc[1]} plaintext: {plain[0]} {plain[1]}; encrypted: {enc[2]} {enc[0]} plaintext: {plain[2]} {plain[0]}; encrypted: {enc[0]} {enc[0]} plaintext: ?"
            answer = f"{plain[0]} {plain[0]}"
            profile = "extra_explanation_text"
        else:
            prompt = f"word seed {i}: {enc[0]} {enc[1]} -> {plain[0]} {plain[1]}; {enc[2]} {enc[3]} -> {plain[2]} {plain[3]}; {enc[1]} {enc[3]} -> ?"
            answer = f"{plain[1]} {plain[3]}"
            profile = "answer_only_style"
        rows.append(_base(f"{variant}_word_answer_{i:03d}", "word_cipher", "word_substitution", prompt, answer, "answer", f"word_dict_seed_{i:03d}_dict_{i % len(dictionaries)}", variant=variant, difficulty=3, noise_profile=profile))
    for i in range(9):
        enc, plain = dictionaries[i % len(dictionaries)]
        profile = ["conflicting_examples", "unknown_symbols_or_words", "mismatched_phrase_length"][i % 3]
        if i % 3 == 0:
            prompt = f'word abstain seed {i}: "{enc[0]} {enc[1]}" -> "{plain[0]} {plain[1]}"; "{enc[0]} {enc[2]}" -> "wrong {plain[2]}"; encrypted: {enc[0]} {enc[2]} plaintext: ?'
        elif i % 3 == 1:
            prompt = f'word abstain seed {i}: "{enc[0]} {enc[1]}" -> "{plain[0]} {plain[1]}"; encrypted: {enc[0]} unknown{i} plaintext: ?'
        else:
            prompt = f'word abstain seed {i}: "{enc[0]} {enc[1]} {enc[2]}" -> "{plain[0]} {plain[1]}"; decode: {enc[0]} {enc[1]}?'
        rows.append(_base(f"{variant}_word_abstain_{i:03d}", "word_cipher", "expected_abstain", prompt, "ABSTAIN", "abstain", f"word_abstain_seed_{i:03d}_{profile}", variant=variant, difficulty=4, noise_profile=profile))
    return rows


def _unit_rows(variant: str) -> list[dict]:
    rows: list[dict] = []
    for i in range(27):
        q = Decimal(i + 10) / (Decimal("2") if i % 4 == 0 else Decimal("1"))
        if i % 3 == 0:
            scale = Decimal("1.50") + Decimal(i % 5) / Decimal("100")
            prompt = f"unit scale seed {i}: 2 -> {format_decimal(Decimal(2) * scale, 2)}; 4 -> {format_decimal(Decimal(4) * scale, 2)}; 6 -> {format_decimal(Decimal(6) * scale, 2)}. Convert {q}"
            answer = format_decimal(q * scale, 2)
            sub = "multiplicative_scale_round_2"
            sig = f"unit_scale_seed_{i:03d}_{scale}"
        elif i % 3 == 1:
            offset = Decimal(i % 7 + 3)
            prompt = f"offset examples {i}: 10 -> {format_decimal(Decimal(10) + offset, 0)}; 20 -> {format_decimal(Decimal(20) + offset, 0)}; 30 -> {format_decimal(Decimal(30) + offset, 0)}. Input: {q}"
            answer = format_decimal(q + offset, 0)
            sub = "linear_offset_round_0"
            sig = f"unit_offset_seed_{i:03d}_{offset}"
        else:
            scale = Decimal("0.75")
            offset = Decimal("2.25")
            prompt = f"target unit rule {i}: 4 -> {format_decimal(Decimal(4) * scale + offset, 2)}; 8 -> {format_decimal(Decimal(8) * scale + offset, 2)}; 12 -> {format_decimal(Decimal(12) * scale + offset, 2)}. What is {q} in the target unit?"
            answer = format_decimal(q * scale + offset, 2)
            sub = "linear_offset_round_2"
            sig = f"unit_scale_offset_seed_{i:03d}"
        rows.append(_base(f"{variant}_unit_answer_{i:03d}", "unit_conversion", sub, prompt, answer, "answer", sig, variant=variant, difficulty=3, noise_profile=NOISE_PROFILES[i % 7]))
    for i in range(9):
        prompt = f"insufficient unit seed {i}: 1 -> 2. For 5, output ?"
        rows.append(_base(f"{variant}_unit_abstain_{i:03d}", "unit_conversion", "expected_abstain", prompt, "ABSTAIN", "abstain", f"unit_abstain_inconsistent_seed_{i:03d}", variant=variant, difficulty=4, noise_profile="conflicting_examples"))
    return rows


def _numeric_gravity_rows(variant: str) -> list[dict]:
    rows: list[dict] = []
    for i in range(13):
        q = i + 4
        if i % 2 == 0:
            a = i % 4 + 2
            prompt = f"numeric quadratic seed {i}: 1 -> {a}; 2 -> {4*a}; 3 -> {9*a}. Query {q}?"
            answer = str(a * q * q)
            sub = "quadratic_scale"
            sig = f"numeric_quad_seed_{i:03d}_a_{a}"
        else:
            a = i % 5 + 2
            b = i % 3 + 1
            prompt = f"numeric linear seed {i}: 1 -> {a+b}; 2 -> {2*a+b}; 3 -> {3*a+b}. Now solve: {q}"
            answer = str(a * q + b)
            sub = "linear_offset"
            sig = f"numeric_linear_seed_{i:03d}_a_{a}_b_{b}"
        rows.append(_base(f"{variant}_numeric_answer_{i:03d}", "numeric_formula", sub, prompt, answer, "answer", sig, variant=variant, difficulty=3, noise_profile=NOISE_PROFILES[i % 6]))
    for i in range(14):
        t = i + 4
        g_half = Decimal("4.90") + Decimal(i % 3) / Decimal("100")
        prompt = f"falling distance seed {i}: time 1 -> {format_decimal(g_half, 2)}; time 2 -> {format_decimal(g_half * 4, 2)}; time 3 -> {format_decimal(g_half * 9, 2)}. Query time {t}?"
        rows.append(_base(f"{variant}_gravity_answer_{i:03d}", "gravity_numeric", "gravity_distance", prompt, format_decimal(g_half * Decimal(t * t), 2), "answer", f"gravity_seed_{i:03d}_half_g_{g_half}", variant=variant, difficulty=3, noise_profile=NOISE_PROFILES[(i + 3) % 6]))
    for i in range(9):
        rows.append(_base(f"{variant}_gravity_abstain_{i:03d}", "gravity_numeric", "expected_abstain", f"gravity mismatch seed {i}: 1 -> 2; 2 -> 4; 3 -> 6. Query 4?", "ABSTAIN", "abstain", f"gravity_abstain_seed_{i:03d}", variant=variant, difficulty=4, noise_profile="conflicting_examples"))
    return rows


def _roman_rows(variant: str) -> list[dict]:
    rows: list[dict] = []
    nums = [4, 9, 38, 44, 58, 94, 145, 399, 944, 1994, 2024, 3999, 27, 73, 166, 388, 512, 777, 2444, 333]
    for i, n in enumerate(nums):
        prompt = _roman_prompt(n, i)
        rows.append(_base(f"{variant}_roman_answer_{i:03d}", "roman_numeral", "standard_roman", prompt, int_to_roman(n), "answer", f"roman_standard_seed_{i:03d}_n_{n}", variant=variant, difficulty=2, noise_profile=NOISE_PROFILES[i % 7]))
    custom_prompts = [
        ("custom glyph invalid seed 0: A -> @; B -> &&. target number: C", "ABSTAIN"),
        ("custom numeral contradiction seed 1: 1 -> @; 2 -> @. solve for 3", "ABSTAIN"),
    ]
    for i, (prompt, answer) in enumerate(custom_prompts):
        rows.append(_base(f"{variant}_custom_abstain_{i:03d}", "custom_numeral", "expected_abstain", prompt, answer, "abstain", f"custom_abstain_seed_{i:03d}", variant=variant, difficulty=4, noise_profile="invalid_custom_numeral"))
    for i in range(6):
        rows.append(_base(f"{variant}_roman_abstain_{i:03d}", "roman_numeral", "expected_abstain", f"invalid roman seed {i}: 11 -> XII; 15 -> XV. solve for 38", "ABSTAIN", "abstain", f"roman_invalid_seed_{i:03d}", variant=variant, difficulty=4, noise_profile="conflicting_examples"))
    return rows


def _bit_pair(bits: str, out_value: int) -> tuple[str, str]:
    return bits, _format_bits(out_value, len(bits))


def _bit_inputs(seed: int, count: int) -> list[str]:
    values = []
    for j in range(count):
        values.append(_format_bits((seed * 29 + j * 67 + 11) % 256, 8))
    return values


def _permutation_basis_inputs() -> list[str]:
    return ["00000000"] + [_format_bits(1 << bit, 8) for bit in range(8)]


def _bit_prompt(examples: list[tuple[str, str]], query: str, profile: str, seed: int) -> str:
    body = "; ".join(f"{a} -> {b}" for a, b in examples)
    if profile == "query_with_arrow":
        return f"Binary transform seed {seed}. {body}; {query} -> ?"
    if profile == "query_with_colon":
        return f"Examples seed {seed}: {body}. Input: {query}"
    if profile == "whitespace_noise":
        return f"  {body.replace(';', ' ;  ')}  \n target: {query}  "
    if profile == "extra_explanation_text":
        return f"Use the same bit rule, ignore this sentence. {body}. solve {query}"
    return f"Seed {seed} bit examples: {body}. Input: {query}"


def _symbol_prompt(examples: list[tuple[str, str]], query: str, profile: str, seed: int) -> str:
    body = "; ".join(f"`{a}` -> `{b}`" if profile == "quoted_examples" else f"{a} -> {b}" for a, b in examples)
    if profile == "query_with_arrow":
        return f"Symbol seed {seed}. {body}; {query} -> ?"
    if profile == "extra_explanation_text":
        return f"Map punctuation only seed {seed}; ignore alphabetic prose; {body}; input: {query} output ?"
    return f"Symbol seed {seed}. {body}; query: {query} output ?"


def _roman_prompt(n: int, seed: int) -> str:
    forms = [
        f"11 -> XI; 15 -> XV. solve for {n}",
        f"Given the examples above, 11 -> XI and 15 -> XV, write {n} in the same system.",
        f"Wonderland numeral examples: 38 -> XXXVIII; 94 -> XCIV. target number: {n}",
        f"Examples: 4 -> IV; 9 -> IX. output for {n}",
    ]
    return forms[seed % len(forms)]


def _permute(bits: str, permutation: tuple[int, ...]) -> str:
    return "".join(bits[index] for index in permutation)


def _shift(text: str, shift: int) -> str:
    out = []
    for char in text:
        base = ord("a")
        out.append(chr(base + ((ord(char) - base + shift) % 26)))
    return "".join(out)


def _assert_unique_fields(rows: list[dict]) -> None:
    for field in ("id", "rule_id", "leakage_group"):
        values = [row[field] for row in rows]
        if len(values) != len(set(values)):
            raise AssertionError(f"duplicate {field} in generated eval")


if __name__ == "__main__":
    raise SystemExit(main())
