from __future__ import annotations

from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import random

from kaggle_anti086.data.schema import validate_rows
from kaggle_anti086.kaggle_path_safety import safe_write_text
from kaggle_anti086.solvers.bit_transform_solver import _format_bits, _rol
from kaggle_anti086.solvers.numeric_parsing import format_decimal
from kaggle_anti086.solvers.roman_solver import int_to_roman


NOISE_PROFILES = (
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
    "ambiguous_rule",
    "long_prompt",
    "format_trap",
)

PRIVATE_LIKE_COUNTS = {
    "bit_manipulation": 64,
    "symbol_mapping": 64,
    "char_cipher": 56,
    "word_cipher": 56,
    "unit_conversion": 56,
    "numeric_formula": 48,
    "gravity_numeric": 40,
    "roman_numeral": 32,
    "custom_numeral": 24,
    "sequence_pattern": 24,
    "permutation_sorting": 16,
    "format_only": 16,
    "equation_operator": 16,
}

RULE_HOLDOUT_COUNTS = {
    "bit_manipulation": 72,
    "symbol_mapping": 72,
    "char_cipher": 56,
    "word_cipher": 56,
    "unit_conversion": 56,
    "numeric_formula": 48,
    "gravity_numeric": 40,
    "roman_numeral": 32,
    "custom_numeral": 24,
    "sequence_pattern": 24,
    "format_only": 16,
    "equation_operator": 16,
}

FAMILY_HARD_COUNTS = {
    "symbol_mapping": 96,
    "char_cipher": 80,
    "bit_manipulation": 80,
    "custom_numeral": 48,
    "equation_operator": 48,
    "numeric_formula": 40,
    "unit_conversion": 32,
    "word_cipher": 32,
    "gravity_numeric": 24,
    "roman_numeral": 16,
    "format_only": 16,
}

ANTI_LEAK_COUNTS = {
    "bit_manipulation": 32,
    "symbol_mapping": 32,
    "char_cipher": 28,
    "word_cipher": 28,
    "unit_conversion": 28,
    "numeric_formula": 24,
    "gravity_numeric": 20,
    "roman_numeral": 16,
    "custom_numeral": 16,
    "sequence_pattern": 12,
    "format_only": 10,
    "equation_operator": 10,
}

SPLIT_BY_EVAL = {
    "private_like": "private_like_eval",
    "rule_holdout": "rule_holdout_eval",
    "family_hard": "family_hard_eval",
    "anti_leak": "anti_leak_eval",
}

UNSUPPORTED_DIRECT_ANSWER_FAMILIES = {
    "custom_numeral",
    "equation_operator",
    "permutation_sorting",
    "sequence_pattern",
}


def build_eval_rows(eval_name: str, rows: int, seed: int) -> list[dict]:
    counts = _scaled_counts(_counts_for_eval(eval_name), rows)
    rng = random.Random(seed)
    output: list[dict] = []
    for family, count in counts.items():
        for local_idx in range(count):
            global_idx = len(output)
            output.append(_row_for_family(eval_name, family, local_idx, global_idx, rng))
    _assert_unique(output, "id")
    _assert_unique(output, "rule_id")
    _assert_unique(output, "leakage_group")
    report = validate_rows(output, context=f"day5_{eval_name}")
    if report["failure_count"]:
        raise SystemExit(json.dumps(report, indent=2, sort_keys=True))
    return output


def build_answerable_rows(source_eval: str, rows: int, seed: int) -> list[dict]:
    """Build an answer-accuracy-only eval without mutating the mixed behavior eval."""
    output: list[dict] = []
    seen_ids: set[str] = set()
    seen_rules: set[str] = set()
    seen_leakage: set[str] = set()
    round_idx = 0
    while len(output) < rows and round_idx < 20:
        candidates = build_eval_rows(source_eval, max(rows * 2, 128), seed + round_idx * 997)
        by_family: dict[str, list[dict]] = {}
        for row in candidates:
            if _is_verified_answerable(row):
                by_family.setdefault(str(row["family"]), []).append(row)
        ordered_candidates: list[dict] = []
        while any(by_family.values()):
            for family in sorted(by_family):
                if by_family[family]:
                    ordered_candidates.append(by_family[family].pop(0))
        for row in ordered_candidates:
            answer_idx = len(output)
            answer_row = _clone_answerable_row(row, source_eval, answer_idx, round_idx)
            if (
                answer_row["id"] in seen_ids
                or answer_row["rule_id"] in seen_rules
                or answer_row["leakage_group"] in seen_leakage
            ):
                continue
            output.append(answer_row)
            seen_ids.add(answer_row["id"])
            seen_rules.add(answer_row["rule_id"])
            seen_leakage.add(answer_row["leakage_group"])
            if len(output) >= rows:
                break
        round_idx += 1
    if len(output) != rows:
        raise SystemExit(f"could only build {len(output)} answerable {source_eval} rows; requested {rows}")
    report = validate_rows(output, context=f"day5_{source_eval}_answerable")
    if report["failure_count"]:
        raise SystemExit(json.dumps(report, indent=2, sort_keys=True))
    return output


def write_rows(path: str | Path, rows: list[dict], *, field_name: str) -> dict:
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    target = safe_write_text(path, payload, field_name=field_name)
    data = target.read_bytes()
    summary = summarize_rows(rows) | {
        "path": str(target),
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    print(json.dumps(summary, sort_keys=True))
    return summary


def summarize_rows(rows: list[dict]) -> dict:
    behaviors = Counter(str(row.get("metadata", {}).get("expected_solver_behavior", "answer")) for row in rows)
    return {
        "row_count": len(rows),
        "family_distribution": dict(sorted(Counter(row["family"] for row in rows).items())),
        "unique_rule_id_count": len({row["rule_id"] for row in rows}),
        "unique_leakage_group_count": len({row["leakage_group"] for row in rows}),
        "answerable_count": behaviors.get("answer", 0),
        "expected_abstain_count": behaviors.get("abstain", 0),
    }


def _is_verified_answerable(row: dict) -> bool:
    metadata = row.get("metadata", {})
    answer = str(row.get("answer", ""))
    return (
        metadata.get("expected_solver_behavior") == "answer"
        and row.get("verification_status") == "verified"
        and bool(answer.strip())
        and answer.strip().upper() != "ABSTAIN"
        and row.get("family") not in UNSUPPORTED_DIRECT_ANSWER_FAMILIES
    )


def _clone_answerable_row(row: dict, source_eval: str, answer_idx: int, round_idx: int) -> dict:
    copied = json.loads(json.dumps(row))
    base_signature = str(copied.get("metadata", {}).get("rule_signature", copied["rule_id"]))
    suffix = f"{source_eval}_answerable_{answer_idx:04d}_{round_idx:02d}"
    copied["id"] = f"day5_{suffix}"
    copied["rule_id"] = f"day5_{suffix}_{_slug(base_signature)}"
    copied["leakage_group"] = f"day5_lg_{suffix}_{_slug(base_signature)}"
    copied["prompt"] = f"[answerable {source_eval} #{answer_idx}] {copied['prompt']}"
    copied["source"] = f"day5_{source_eval}_answerable_builder"
    metadata = dict(copied.get("metadata", {}))
    metadata.update(
        {
            "eval_purpose": "answer_accuracy_eval",
            "source_eval": source_eval,
            "expected_solver_behavior": "answer",
            "source_row_id": row["id"],
            "source_rule_id": row["rule_id"],
            "source_leakage_group": row["leakage_group"],
            "rule_signature": f"{source_eval}_answerable_{base_signature}_{answer_idx:04d}",
        }
    )
    copied["metadata"] = metadata
    return copied


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _counts_for_eval(eval_name: str) -> dict[str, int]:
    if eval_name == "private_like":
        return PRIVATE_LIKE_COUNTS
    if eval_name == "rule_holdout":
        return RULE_HOLDOUT_COUNTS
    if eval_name == "family_hard":
        return FAMILY_HARD_COUNTS
    if eval_name == "anti_leak":
        return ANTI_LEAK_COUNTS
    raise ValueError(f"unknown eval: {eval_name}")


def _scaled_counts(base: dict[str, int], rows: int) -> dict[str, int]:
    total = sum(base.values())
    if rows == total:
        return dict(base)
    scaled = {family: int(count * rows / total) for family, count in base.items()}
    while sum(scaled.values()) < rows:
        family = max(base, key=lambda item: (base[item] * rows / total) - scaled[item])
        scaled[family] += 1
    while sum(scaled.values()) > rows:
        family = max((item for item in scaled if scaled[item] > 0), key=lambda item: scaled[item])
        scaled[family] -= 1
    return scaled


def _row_for_family(eval_name: str, family: str, local_idx: int, global_idx: int, rng: random.Random) -> dict:
    hard_bias = eval_name == "family_hard"
    holdout = eval_name == "rule_holdout"
    anti_leak = eval_name == "anti_leak"
    family_builders = {
        "bit_manipulation": _bit_case,
        "symbol_mapping": _symbol_case,
        "char_cipher": _char_case,
        "word_cipher": _word_case,
        "unit_conversion": _unit_case,
        "numeric_formula": _numeric_case,
        "gravity_numeric": _gravity_case,
        "roman_numeral": _roman_case,
        "custom_numeral": _unsupported_case,
        "sequence_pattern": _unsupported_case,
        "permutation_sorting": _unsupported_case,
        "format_only": _format_only_case,
        "equation_operator": _unsupported_case,
    }
    case = family_builders[family](eval_name, family, local_idx, rng, hard_bias)
    row_id = f"day5_{eval_name}_{family}_{local_idx:04d}"
    rule_prefix = "holdout_" if holdout else f"day5_{eval_name}_"
    raw_signature = case["rule_signature"]
    signature = f"holdout_semantic_{raw_signature}" if holdout else f"{eval_name}_{raw_signature}"
    rule_id = f"{rule_prefix}{family}_{case['subfamily']}_{local_idx:04d}_{_slug(signature)}"
    metadata = {
        "expected_solver_behavior": case["behavior"],
        "rule_signature": signature,
        "generator_id": f"{eval_name}_{family}_gen_v1",
        "difficulty_score": case["difficulty"],
        "noise_profile": case["noise_profile"],
        "public_like": eval_name == "private_like",
        "private_like": eval_name == "private_like",
        "rule_holdout_reserved": holdout,
    } | case.get("metadata", {})
    if holdout:
        metadata.update({"holdout_only": True, "train_allowed": False, "rule_holdout_reserved": True})
    if anti_leak:
        metadata.update({"anti_leak": True, "leakage_probe_type": _anti_leak_type(local_idx)})
    return {
        "id": row_id,
        "family": family,
        "subfamily": case["subfamily"],
        "rule_id": rule_id,
        "prompt": f"[{eval_name} #{global_idx}] {case['prompt']}",
        "answer": case["answer"],
        "source": f"day5_{eval_name}_builder",
        "solver_name": "solver_ensemble",
        "verification_status": "verified",
        "difficulty": case["difficulty"],
        "split": SPLIT_BY_EVAL[eval_name],
        "leakage_group": f"day5_{eval_name}_lg_{family}_{local_idx:04d}_{_slug(signature)}",
        "metadata": metadata,
    }


def _bit_case(eval_name: str, family: str, i: int, rng: random.Random, hard_bias: bool) -> dict:
    if _should_abstain(i, hard_bias, period=5 if hard_bias else 9):
        if i % 3 == 0:
            prompt = f"0101 -> 1010; 11110000 -> 00001111. Input: 0101"
            sig = f"bit_mixed_width_{i}"
        elif i % 3 == 1:
            prompt = f"01010101 -> 10101010. Input: 11110000"
            sig = f"bit_insufficient_{i}"
        else:
            prompt = f"0000 -> 0000; 1111 -> 1111; 0011 -> 1100. Input: 0101"
            sig = f"bit_ambiguous_{i}"
        return _case("expected_abstain", prompt, "ABSTAIN", "abstain", sig, 4, "ambiguous_rule")
    q = _format_bits((i * 37 + 19) % 256, 8)
    inputs = [_format_bits((i * 23 + j * 61 + 7) % 256, 8) for j in range(4)]
    mode = i % (5 if hard_bias else 4)
    if mode == 0:
        mask = (0xA5 ^ (i * 7)) & 0xFF
        examples = [(bits, _format_bits(int(bits, 2) ^ mask, 8)) for bits in inputs]
        answer = _format_bits(int(q, 2) ^ mask, 8)
        sub = "xor_mask"
        sig = f"bit_xor_mask_{mask:02x}_{i}"
    elif mode == 1:
        examples = [(bits, bits[::-1]) for bits in inputs]
        answer = q[::-1]
        sub = "reverse_bits"
        sig = f"bit_reverse_{i}"
    elif mode == 2:
        k = 1 + i % 7
        examples = [(bits, _rol(bits, k)) for bits in inputs]
        answer = _rol(q, k)
        sub = "rotate_left"
        sig = f"bit_rotate_left_{k}_{i}"
    else:
        k = 1 + i % 5
        mask = (0x33 + i * 3) & 0xFF
        examples = [(bits, _format_bits(int(_rol(bits, k), 2) ^ mask, 8)) for bits in inputs]
        answer = _format_bits(int(_rol(q, k), 2) ^ mask, 8)
        sub = "rotate_then_xor"
        sig = f"bit_rotate_xor_{k}_{mask:02x}_{i}"
    prompt = "; ".join(f"{a} -> {b}" for a, b in examples) + f". Input: {q}"
    return _case(sub, prompt, answer, "answer", sig, 3 if hard_bias else 2, _noise(i))


def _symbol_case(eval_name: str, family: str, i: int, rng: random.Random, hard_bias: bool) -> dict:
    if _should_abstain(i, hard_bias, period=4 if hard_bias else 10):
        prompt = f"!@ -> ab; !# -> xy; query: !# output ?" if i % 2 == 0 else f"!@ -> ab; #$ -> cd; query: !Z% output ?"
        return _case("expected_abstain", prompt, "ABSTAIN", "abstain", f"symbol_conflict_unknown_{i}", 4, "unknown_symbols_or_words")
    src = "!@#$%^&*" if i % 2 == 0 else "[]{}<>?/"
    symbol_targets = ("@&![]{}#", "01234567", "+-=~|:;.")
    dst = symbol_targets[i % len(symbol_targets)]
    mapping = dict(zip(src, dst))
    query = src[1] + src[3] + src[5]
    reverse = hard_bias and i % 5 == 0
    def tr(text: str) -> str:
        base = text[::-1] if reverse else text
        return "".join(mapping[ch] for ch in base)
    examples = [(src[0:3], tr(src[0:3])), (src[3:6], tr(src[3:6])), (src[1:4], tr(src[1:4]))]
    prompt = "; ".join(f"{a} -> {b}" for a, b in examples) + f"; query: {query} output ?"
    return _case("reversal_substitution" if reverse else "char_substitution", prompt, tr(query), "answer", f"symbol_map_{i}_reverse_{int(reverse)}", 4 if hard_bias else 2, _noise(i))


def _char_case(eval_name: str, family: str, i: int, rng: random.Random, hard_bias: bool) -> dict:
    if _should_abstain(i, hard_bias, period=5 if hard_bias else 11):
        prompt = f"abc -> def; abc -> xyz; query: abc" if i % 2 == 0 else f"trb -> cat; hffk -> book; query: zzz"
        return _case("expected_abstain", prompt, "ABSTAIN", "abstain", f"char_abstain_{i}", 4, "conflicting_examples")
    if i % 3 == 0:
        shift = 1 + i % 12
        q = ["mno", "logic", "trace", "vector"][i % 4]
        return _case("caesar_shift", f"abc -> {_shift('abc', shift)}; xyz -> {_shift('xyz', shift)}; query: {q}", _shift(q, shift), "answer", f"char_caesar_{shift}_{i}", 3, _noise(i))
    if i % 3 == 1:
        q = ["solver", "matrix", "vector"][i % 3]
        return _case("reverse_string", f"abc -> cba; book -> koob; query: {q}", q[::-1], "answer", f"char_reverse_{i}", 2, _noise(i))
    return _case("monoalphabetic_substitution", "trb -> cat; hffk -> book; query: trbhffk", "catbook", "answer", f"char_mono_{i}", 3, _noise(i))


def _word_case(eval_name: str, family: str, i: int, rng: random.Random, hard_bias: bool) -> dict:
    enc = ["wkgqa", "lsrqaq", "wneeke", "nwrjnvaq"]
    plain = ["mouse", "chases", "mirror", "imagines"]
    if _should_abstain(i, hard_bias, period=7 if hard_bias else 14):
        prompt = f'"{enc[0]} {enc[1]}" -> "{plain[0]} {plain[1]}"; encrypted: {enc[0]} unknown{i} plaintext: ?'
        return _case("expected_abstain", prompt, "ABSTAIN", "abstain", f"word_unknown_{i}", 4, "unknown_symbols_or_words")
    if i % 2 == 0:
        prompt = f'"{enc[0]} {enc[1]} {enc[2]}" -> "{plain[0]} {plain[1]} {plain[2]}"; encrypted: {enc[0]} {enc[2]} plaintext: ?'
        answer = f"{plain[0]} {plain[2]}"
    else:
        prompt = f"`{enc[0]} {enc[3]}` = `{plain[0]} {plain[3]}`; `{enc[1]} {enc[2]}` = `{plain[1]} {plain[2]}`; decode: {enc[3]} {enc[0]}?"
        answer = f"{plain[3]} {plain[0]}"
    return _case("word_substitution", prompt, answer, "answer", f"word_dict_{i}", 2, _noise(i))


def _unit_case(eval_name: str, family: str, i: int, rng: random.Random, hard_bias: bool) -> dict:
    if _should_abstain(i, hard_bias, period=13 if hard_bias else 20):
        return _case("expected_abstain", f"1 -> 2. For 5, output ?", "ABSTAIN", "abstain", f"unit_insufficient_{i}", 4, "ambiguous_rule")
    q = Decimal(i + 10)
    if i % 3 == 0:
        scale = Decimal("1.50") + Decimal(i % 5) / Decimal("100")
        prompt = f"2 -> {format_decimal(Decimal(2) * scale, 2)}; 4 -> {format_decimal(Decimal(4) * scale, 2)}; 6 -> {format_decimal(Decimal(6) * scale, 2)}. Convert {q}"
        answer = format_decimal(q * scale, 2)
        sig = f"unit_scale_{scale}_{i}"
        sub = "multiplicative_scale_round_2"
    else:
        offset = Decimal(i % 7 + 3)
        prompt = f"10 -> {format_decimal(Decimal(10) + offset, 0)}; 20 -> {format_decimal(Decimal(20) + offset, 0)}; 30 -> {format_decimal(Decimal(30) + offset, 0)}. Input: {q}"
        answer = format_decimal(q + offset, 0)
        sig = f"unit_offset_{offset}_{i}"
        sub = "linear_offset_round_0"
    return _case(sub, prompt, answer, "answer", sig, 3, _noise(i))


def _numeric_case(eval_name: str, family: str, i: int, rng: random.Random, hard_bias: bool) -> dict:
    q = i + 4
    if i % 2 == 0:
        a = i % 4 + 2
        return _case("quadratic_scale", f"1 -> {a}; 2 -> {4*a}; 3 -> {9*a}. Query {q}?", str(a * q * q), "answer", f"numeric_quad_{a}_{i}", 3, _noise(i))
    a = i % 5 + 2
    b = i % 3 + 1
    return _case("linear_offset", f"1 -> {a+b}; 2 -> {2*a+b}; 3 -> {3*a+b}. Now solve: {q}", str(a * q + b), "answer", f"numeric_linear_{a}_{b}_{i}", 3, _noise(i))


def _gravity_case(eval_name: str, family: str, i: int, rng: random.Random, hard_bias: bool) -> dict:
    if _should_abstain(i, hard_bias, period=8 if hard_bias else 16):
        return _case("expected_abstain", "gravity mismatch: 1 -> 2; 2 -> 4; 3 -> 6. Query 4?", "ABSTAIN", "abstain", f"gravity_mismatch_{i}", 4, "conflicting_examples")
    t = i + 4
    half_g = Decimal("4.90") + Decimal(i % 3) / Decimal("100")
    prompt = f"time 1 -> {format_decimal(half_g, 2)}; time 2 -> {format_decimal(half_g * 4, 2)}; time 3 -> {format_decimal(half_g * 9, 2)}. Query time {t}?"
    return _case("gravity_distance", prompt, format_decimal(half_g * Decimal(t * t), 2), "answer", f"gravity_half_g_{half_g}_{i}", 3, _noise(i))


def _roman_case(eval_name: str, family: str, i: int, rng: random.Random, hard_bias: bool) -> dict:
    n = [4, 9, 38, 44, 58, 94, 145, 399, 944, 1994, 2024, 3999, 27, 73, 166, 388][i % 16]
    prompt = f"11 -> XI; 15 -> XV. solve for {n}"
    return _case("standard_roman", prompt, int_to_roman(n), "answer", f"roman_{n}_{i}", 2, _noise(i))


def _format_only_case(eval_name: str, family: str, i: int, rng: random.Random, hard_bias: bool) -> dict:
    cases = [
        ("boxed_answer", r"\boxed{154.620}", "154.62", "numeric"),
        ("think_tag_suffix", "</think>\nXXXVIII", "XXXVIII", "roman"),
        ("arrow_output", "38 -> XXXVIII", "XXXVIII", "roman"),
        ("answer_prefix_symbol", "Answer: @&.", "@&", "symbol"),
        ("text_answer_prefix", "The answer is: cat book.", "cat book", "text_phrase"),
        ("binary_first_line", "00101010\nExplanation: copied from scratch work.", "00101010", "binary"),
    ]
    sub, raw, answer, answer_type = cases[i % len(cases)]
    prompt = f"Format-only cleanup task. Raw output: {raw}\nExtract final answer only."
    return _case(sub, prompt, answer, "answer", f"format_only_{sub}_{i}", 1, "format_trap", {"answer_type": answer_type})


def _unsupported_case(eval_name: str, family: str, i: int, rng: random.Random, hard_bias: bool) -> dict:
    prompts = {
        "custom_numeral": f"custom glyphs {i}: 1 -> @; 2 -> &&. target number: 3",
        "sequence_pattern": f"sequence {i}: 2, 4, 8, 16, ?",
        "permutation_sorting": f"sort by hidden rule {i}: cab -> abc; bca -> abc; query: dac",
        "equation_operator": f"operator equation {i}: A @ B -> C. infer target A # B?",
    }
    return _case("expected_abstain", prompts.get(family, f"unsupported {family} {i}"), "ABSTAIN", "abstain", f"{family}_unsupported_{i}", 4 if hard_bias else 3, "format_trap" if family == "format_only" else "ambiguous_rule")


def _case(subfamily: str, prompt: str, answer: str, behavior: str, signature: str, difficulty: int, noise: str, metadata: dict | None = None) -> dict:
    return {
        "subfamily": subfamily,
        "prompt": prompt,
        "answer": answer,
        "behavior": behavior,
        "rule_signature": signature,
        "difficulty": difficulty,
        "noise_profile": noise,
        "metadata": metadata or {},
    }


def _should_abstain(i: int, hard_bias: bool, *, period: int) -> bool:
    return i % period == period - 1


def _noise(i: int) -> str:
    return NOISE_PROFILES[i % len(NOISE_PROFILES)]


def _shift(text: str, shift: int) -> str:
    return "".join(chr(ord("a") + ((ord(char) - ord("a") + shift) % 26)) for char in text)


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value)[:80]


def _anti_leak_type(i: int) -> str:
    return (
        "surface_template_reuse",
        "same_style_different_rule",
        "distractor_rule",
        "conflicting_examples",
        "near_duplicate_structure",
        "answer_format_trap",
    )[i % 6]


def _assert_unique(rows: list[dict], key: str) -> None:
    values = [row[key] for row in rows]
    if len(values) != len(set(values)):
        raise AssertionError(f"duplicate {key}")
