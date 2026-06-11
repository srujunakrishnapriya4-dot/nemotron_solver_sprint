from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import random
import re
from typing import Any, Callable

from kaggle_anti086.training.day1_teacher_trainability_gate import (
    detect_abstain_placeholder,
    verify_trainable_batch,
    verify_trainable_row,
)
from kaggle_anti086.training.day1_teacher_verifier import count_boxed_answers, validate_target_text


SOURCE = "day1x_composed_synthetic"
BASE_PROMPT_STYLES = ("direct", "minimal", "examples_query", "table", "story")
DIFFICULTIES = ("easy", "medium", "hard")
REQUIRED_COMPOSED_FAMILIES = (
    "composed_custom_numeral_arithmetic",
    "composed_symbol_equation",
    "composed_cipher_mapping",
    "composed_unit_formula",
    "composed_bit_conversion",
    "composed_sequence_operator",
    "composed_permutation_mapping",
    "composed_word_cipher",
    "composed_gravity_unit",
)
PLACEHOLDER_TOKENS = ("ABSTAIN", "TODO", "UNKNOWN", "N/A", "NONE-AS-ANSWER", "EXPECTED-ABSTAIN", "PLACEHOLDER")


def normalize_answer(value: Any) -> str:
    text = str(value).strip()
    if re.fullmatch(r"[+-]?\d+", text):
        return str(int(text))
    return text


def make_boxed_target(trace: str, answer: str) -> str:
    answer = normalize_answer(answer)
    trace = str(trace).strip()
    if r"\boxed{" in trace:
        raise ValueError("trace_must_not_contain_intermediate_box")
    return f"{trace}\n\\boxed{{{answer}}}"


def validate_composed_row(row: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not isinstance(row, dict):
        return False, ["row_not_dict"]
    family = str(row.get("family", ""))
    if not family.startswith("composed_"):
        errors.append("family_not_composed")
    if family not in COMPOSED_TEACHER_REGISTRY:
        errors.append("unknown_composed_family")
    if not isinstance(row.get("subfamilies"), list) or len(row.get("subfamilies", [])) < 2:
        errors.append("subfamilies_too_short")
    if not isinstance(row.get("substeps"), list) or len(row.get("substeps", [])) < 2:
        errors.append("substeps_too_short")
    if row.get("verification_status") != "PASS":
        errors.append("final_verification_not_pass")
    if row.get("ambiguity_count") != 0:
        errors.append("final_ambiguity_nonzero")
    if detect_abstain_placeholder(row):
        errors.append("abstain_or_placeholder_present")
    joined = "\n".join(str(row.get(key, "")) for key in ("prompt", "trace", "answer", "target_text"))
    if any(token in joined.upper() for token in PLACEHOLDER_TOKENS):
        errors.append("placeholder_token_present")
    answer = normalize_answer(row.get("answer", ""))
    target_text = str(row.get("target_text", ""))
    if count_boxed_answers(target_text) != 1:
        errors.append("target_not_exactly_one_box")
    if not target_text.endswith(f"\\boxed{{{answer}}}"):
        errors.append("target_does_not_end_with_answer_box")
    ok, reason = validate_target_text(target_text, answer)
    if not ok:
        errors.append(reason or "target_text_invalid")
    substeps_ok, substep_errors = verify_substeps(row)
    if not substeps_ok:
        errors.extend(substep_errors)
    trainability = verify_trainable_row(row)
    if not trainability.trainable:
        errors.append(f"trainability_gate_failed:{trainability.rejection_reason}")
    if row.get("trainable") is not True:
        errors.append("trainable_flag_not_true")
    return not errors, errors


def assert_composed_row_trainable(row: dict[str, Any]) -> None:
    ok, errors = validate_composed_row(row)
    if not ok:
        raise ValueError(f"Composed row is not trainable: {errors}")


def verify_substeps(row: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    substeps = row.get("substeps")
    if not isinstance(substeps, list):
        return False, ["substeps_not_list"]
    for expected_index, step in enumerate(substeps):
        if not isinstance(step, dict):
            errors.append(f"substep_{expected_index}_not_dict")
            continue
        if step.get("index") != expected_index:
            errors.append(f"substep_{expected_index}_bad_index")
        if step.get("verification_status") != "PASS":
            errors.append(f"substep_{expected_index}_verification_not_pass")
        if step.get("ambiguity_count") != 0:
            errors.append(f"substep_{expected_index}_ambiguity_nonzero")
        try:
            recomputed = _verify_step_output(step)
        except Exception as exc:
            errors.append(f"substep_{expected_index}_recompute_error:{type(exc).__name__}")
            continue
        if normalize_answer(recomputed) != normalize_answer(step.get("output", "")):
            errors.append(f"substep_{expected_index}_output_mismatch")
    if substeps:
        if normalize_answer(substeps[-1].get("output", "")) != normalize_answer(row.get("answer", "")):
            errors.append("final_answer_not_last_substep_output")
    return not errors, errors


@dataclass(frozen=True)
class ComposedTeacher:
    family: str
    subfamilies: tuple[str, ...]
    generator_version: str
    builder: Callable[[random.Random, int, str, str], dict[str, Any]]

    def generate(self, seed: int, count: int, prompt_style: str | None = None, difficulty: str | None = None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for index in range(count):
            style = prompt_style or BASE_PROMPT_STYLES[index % len(BASE_PROMPT_STYLES)]
            diff = difficulty or DIFFICULTIES[index % len(DIFFICULTIES)]
            rng = random.Random(seed + _family_offset(self.family) + index * 9973)
            row = self.builder(rng, index, style, diff)
            _retag_row(row, f"{self.family}-{seed}-{index}")
            ok, errors = validate_composed_row(row)
            if not ok:
                raise ValueError(f"{self.family} generated invalid row {index}: {errors}")
            rows.append(row)
        return rows

    def verify(self, row: dict[str, Any]) -> tuple[bool, list[str]]:
        if row.get("family") != self.family:
            return False, ["wrong_family"]
        return validate_composed_row(row)

    def generate_hard_negative(self, row: dict[str, Any], seed: int) -> dict[str, Any] | None:
        rng = random.Random(seed + _family_offset(self.family))
        wrong = _wrong_answer(str(row["answer"]), rng)
        negative = deepcopy(row)
        negative["id"] = _stable_id("hardneg", str(row.get("id")), wrong, str(seed))
        negative["answer"] = wrong
        negative["trace"] = make_boxed_target("Incorrectly perturb the final composed result.", wrong)
        negative["target_text"] = negative["trace"]
        negative["verification_status"] = "FAIL"
        negative["trainable"] = False
        return negative

    def supported_prompt_styles(self) -> list[str]:
        return list(BASE_PROMPT_STYLES)

    def supported_difficulties(self) -> list[str]:
        return list(DIFFICULTIES)


def list_composed_families() -> list[str]:
    return sorted(COMPOSED_TEACHER_REGISTRY)


def generate_composed_rows(per_family: int, seed: int, families: list[str] | None = None) -> list[dict[str, Any]]:
    selected = families or list(REQUIRED_COMPOSED_FAMILIES)
    rows: list[dict[str, Any]] = []
    for family in selected:
        rows.extend(COMPOSED_TEACHER_REGISTRY[family].generate(seed=seed, count=per_family))
    return rows


def verify_composed_dataset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    validation = [validate_composed_row(row) for row in rows]
    checks, trainability = verify_trainable_batch(rows)
    per_family: dict[str, dict[str, int]] = {}
    for family in sorted({str(row.get("family")) for row in rows}):
        family_rows = [row for row in rows if row.get("family") == family]
        family_valid = [validate_composed_row(row)[0] for row in family_rows]
        per_family[family] = {
            "generated": len(family_rows),
            "verified_pass": sum(1 for ok in family_valid if ok),
            "verified_fail": sum(1 for ok in family_valid if not ok),
        }
    failures = [error for ok, errors in validation if not ok for error in errors]
    return {
        "schema_version": 1,
        "total_rows": len(rows),
        "verified_pass": sum(1 for ok, _errors in validation if ok),
        "verified_fail": sum(1 for ok, _errors in validation if not ok),
        "format_error_count": sum(1 for error in failures if "target" in error or "box" in error),
        "verification_fail_count": sum(1 for check in checks if not check.trainable),
        "ambiguity_accepted": sum(1 for row in rows if row.get("ambiguity_count") != 0),
        "abstain_accepted": sum(1 for row in rows if detect_abstain_placeholder(row)),
        "duplicate_prompt_count": trainability.duplicate_prompt_count,
        "per_family": per_family,
        "errors": dict(sorted(Counter(failures).items())),
        "trainable": trainability.trainable,
    }


def _make_row(
    *,
    family: str,
    subfamilies: list[str],
    difficulty: str,
    prompt_style: str,
    prompt: str,
    trace_lines: list[str],
    answer: str,
    substeps: list[dict[str, Any]],
    generator: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    answer = normalize_answer(answer)
    trace = make_boxed_target("\n".join(trace_lines), answer)
    row = {
        "schema_version": 1,
        "id": _stable_id(family, prompt, answer, generator),
        "record_id": _stable_id("record", family, prompt, answer, generator),
        "family": family,
        "rule_id": generator,
        "subfamilies": subfamilies,
        "difficulty": difficulty,
        "prompt_style": prompt_style,
        "prompt": prompt,
        "trace": trace,
        "answer": answer,
        "target_text": trace,
        "verification_status": "PASS",
        "ambiguity_count": 0,
        "source": SOURCE,
        "generator": generator,
        "seed": 0,
        "split": "day1x_sample",
        "teacher_version": generator,
        "trainable": True,
        "metadata": {"output_type": "integer", **(metadata or {})},
        "substeps": substeps,
    }
    row["messages"] = [{"role": "user", "content": prompt}, {"role": "assistant", "content": trace}]
    return row


def _retag_row(row: dict[str, Any], sample_key: str) -> None:
    prompt = f"{row['prompt']} Sample key: {sample_key}."
    row["prompt"] = prompt
    row["id"] = _stable_id(str(row["family"]), prompt, str(row["answer"]), str(row["generator"]))
    row["record_id"] = _stable_id("record", str(row["family"]), prompt, str(row["answer"]), str(row["generator"]))
    row["messages"] = [{"role": "user", "content": prompt}, {"role": "assistant", "content": row["target_text"]}]


def _step(index: int, family: str, input_value: str, output: Any, op: str, params: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": index,
        "family": family,
        "input": input_value,
        "output": normalize_answer(output),
        "verification_status": "PASS",
        "ambiguity_count": 0,
        "op": op,
        "params": params,
    }


def _build_custom_numeral_arithmetic(rng: random.Random, index: int, style: str, difficulty: str) -> dict[str, Any]:
    alphabet = list("ABCDEFGH")[: rng.choice([4, 5, 6])]
    base = len(alphabet)
    value = rng.randint(10 + index, 90 + index)
    numeral = _encode_base(value, alphabet)
    op = rng.choice(["add", "mul", "sub", "affine", "square"])
    c = rng.randint(2, 9)
    if op == "add":
        final = value + c
        phrase = f"then add {c}"
    elif op == "mul":
        final = value * c
        phrase = f"then multiply by {c}"
    elif op == "sub":
        final = value - c
        phrase = f"then subtract {c}"
    elif op == "square":
        value = rng.randint(2, 14)
        numeral = _encode_base(value, alphabet)
        final = value * value
        phrase = "then square it"
    else:
        a, b = rng.randint(2, 5), rng.randint(-5, 9)
        final = a * value + b
        phrase = f"then compute {a}*x+{b}"
        c = b
    mapping = ", ".join(f"{symbol}={digit}" for digit, symbol in enumerate(alphabet))
    prompt = _styled_prompt(style, "custom numeral arithmetic", f"Base {base} digits: {mapping}. Decode {numeral}, {phrase}.")
    arithmetic_params = {"a": a, "b": b} if op == "affine" else {"c": c}
    substeps = [
        _step(0, "custom_numeral", numeral, value, "custom_decode", {"alphabet": alphabet, "base": base}),
        _step(1, "arithmetic", str(value), final, op, arithmetic_params),
    ]
    return _make_row(
        family="composed_custom_numeral_arithmetic",
        subfamilies=["custom_numeral", "arithmetic"],
        difficulty=difficulty,
        prompt_style=style,
        prompt=prompt,
        trace_lines=[f"Decode {numeral} in base {base} to {value}.", f"Apply the explicit arithmetic to get {final}."],
        answer=str(final),
        substeps=substeps,
        generator="composed_custom_numeral_arithmetic_v1",
    )


def _build_symbol_equation(rng: random.Random, index: int, style: str, difficulty: str) -> dict[str, Any]:
    symbols = ["A", "B", "C", "D", "X", "Y", "Z"]
    chosen = rng.sample(symbols, 3)
    values = [rng.randint(1, 12) + index % 5 for _ in chosen]
    mapping = dict(zip(chosen, values))
    op = rng.choice(["add_sub", "mul_sub", "max_add", "abs_add"])
    a, b, c = chosen
    if op == "add_sub":
        mapped = mapping[a] + mapping[b]
        final = mapped - mapping[c]
        expr = f"{a}+{b}-{c}"
    elif op == "mul_sub":
        mapped = mapping[a] * mapping[b]
        final = mapped - mapping[c]
        expr = f"{a}*{b}-{c}"
    elif op == "max_add":
        mapped = max(mapping[a], mapping[b])
        final = mapped + mapping[c]
        expr = f"max({a},{b})+{c}"
    else:
        mapped = abs(mapping[a] - mapping[b])
        final = mapped + mapping[c]
        expr = f"abs({a}-{b})+{c}"
    prompt = _styled_prompt(style, "symbol equation", f"Let {', '.join(f'{k}={v}' for k, v in mapping.items())}. Evaluate {expr}.")
    substeps = [
        _step(0, "symbol_mapping", expr, mapped, "symbol_expression", {"mapping": mapping, "op": op, "symbols": [a, b]}),
        _step(1, "equation_operator", str(mapped), final, "add" if op in {"max_add", "abs_add"} else "sub", {"c": mapping[c]}),
    ]
    return _make_row(
        family="composed_symbol_equation",
        subfamilies=["symbol_mapping", "equation_operator"],
        difficulty=difficulty,
        prompt_style=style,
        prompt=prompt,
        trace_lines=[f"Map symbols in {expr} using the explicit table.", f"The composed value is {final}."],
        answer=str(final),
        substeps=substeps,
        generator="composed_symbol_equation_v1",
    )


def _build_cipher_mapping(rng: random.Random, index: int, style: str, difficulty: str) -> dict[str, Any]:
    words = ("MATH", "CODE", "LOGIC", "PRIME", "VECTOR", "SOLVE")
    text = words[(index + rng.randint(0, 5)) % len(words)]
    cipher = rng.choice(["caesar", "atbash", "reverse"])
    if cipher == "caesar":
        shift = rng.randint(1, 5)
        encoded = _caesar(text, shift)
        decoded = _caesar(encoded, -shift)
        cipher_phrase = f"Caesar shift -{shift}"
    elif cipher == "atbash":
        shift = 0
        encoded = _atbash(text)
        decoded = _atbash(encoded)
        cipher_phrase = "Atbash decode"
    else:
        shift = 0
        encoded = text[::-1]
        decoded = encoded[::-1]
        cipher_phrase = "reverse the string"
    total = sum(ord(ch) - 64 for ch in decoded)
    prompt = _styled_prompt(style, "cipher mapping", f"{cipher_phrase} on {encoded}, then sum A=1 through Z=26 for the decoded letters.")
    substeps = [
        _step(0, "char_cipher", encoded, decoded, cipher, {"shift": -shift if cipher == "caesar" else shift}),
        _step(1, "symbol_mapping", decoded, total, "a1z26_sum", {}),
    ]
    return _make_row(
        family="composed_cipher_mapping",
        subfamilies=["char_cipher", "symbol_mapping"],
        difficulty=difficulty,
        prompt_style=style,
        prompt=prompt,
        trace_lines=[f"The cipher decodes {encoded} to {decoded}.", f"A1Z26 sum is {total}."],
        answer=str(total),
        substeps=substeps,
        generator="composed_cipher_mapping_v1",
    )


def _build_unit_formula(rng: random.Random, index: int, style: str, difficulty: str) -> dict[str, Any]:
    cases = [("m", "cm", 100), ("kg", "g", 1000), ("minutes", "seconds", 60), ("days", "hours", 24), ("dozens", "units", 12)]
    src, dst, factor = cases[(index + rng.randint(0, len(cases) - 1)) % len(cases)]
    value = rng.randint(1, 20)
    converted = value * factor
    a, b = rng.randint(2, 5), rng.randint(-20, 30)
    final = a * converted + b
    prompt = _styled_prompt(style, "unit formula", f"Convert {value} {src} to {dst}; call that x. Compute {a}*x+{b}.")
    substeps = [
        _step(0, "unit_conversion", f"{value} {src} to {dst}", converted, "unit_convert", {"src": src, "dst": dst, "factor": factor}),
        _step(1, "numeric_formula_safe", str(converted), final, "affine", {"a": a, "b": b}),
    ]
    return _make_row(
        family="composed_unit_formula",
        subfamilies=["unit_conversion", "numeric_formula_safe"],
        difficulty=difficulty,
        prompt_style=style,
        prompt=prompt,
        trace_lines=[f"The exact conversion gives x = {converted}.", f"{a}*x+{b} = {final}."],
        answer=str(final),
        substeps=substeps,
        generator="composed_unit_formula_v1",
    )


def _build_bit_conversion(rng: random.Random, index: int, style: str, difficulty: str) -> dict[str, Any]:
    width = rng.choice([4, 5, 6, 8])
    a = rng.randint(0, (1 << width) - 1)
    b = rng.randint(0, (1 << width) - 1)
    op = rng.choice(["xor", "and", "or", "not", "reverse", "lshift", "rshift"])
    shift = rng.randint(1, 2)
    if op == "xor":
        value = a ^ b
        phrase = f"Compute {a} XOR {b}"
    elif op == "and":
        value = a & b
        phrase = f"Compute {a} AND {b}"
    elif op == "or":
        value = a | b
        phrase = f"Compute {a} OR {b}"
    elif op == "not":
        value = ((1 << width) - 1) ^ a
        phrase = f"Apply NOT to {a} using width {width}"
    elif op == "reverse":
        value = int(format(a, f"0{width}b")[::-1], 2)
        phrase = f"Reverse the {width}-bit binary form of {a}"
    elif op == "lshift":
        value = (a << shift) & ((1 << width) - 1)
        phrase = f"Left shift {a} by {shift} within width {width}"
    else:
        value = a >> shift
        phrase = f"Right shift {a} by {shift} within width {width}"
    final = int(str(value), 10)
    prompt = _styled_prompt(style, "bit conversion", f"{phrase}, then give the decimal result.")
    substeps = [
        _step(0, "bit_manipulation", str(a), value, op, {"a": a, "b": b, "width": width, "shift": shift}),
        _step(1, "unit_conversion", str(value), final, "decimal_identity", {}),
    ]
    return _make_row(
        family="composed_bit_conversion",
        subfamilies=["bit_manipulation", "unit_conversion"],
        difficulty=difficulty,
        prompt_style=style,
        prompt=prompt,
        trace_lines=[f"The bit operation gives {value}.", f"Decimal representation is {final}."],
        answer=str(final),
        substeps=substeps,
        generator="composed_bit_conversion_v1",
    )


def _build_sequence_operator(rng: random.Random, index: int, style: str, difficulty: str) -> dict[str, Any]:
    rule = rng.choice(["arithmetic", "geometric", "second_order"])
    if rule == "arithmetic":
        start, step = rng.randint(-10, 10), rng.choice([2, 3, 4, 5, -2, -3])
        seq = [start + step * i for i in range(5)]
        nxt = seq[-1] + step
    elif rule == "geometric":
        start, ratio = rng.randint(1, 5), rng.choice([2, 3])
        seq = [start * (ratio**i) for i in range(5)]
        nxt = seq[-1] * ratio
    else:
        start, diff, second = rng.randint(-5, 5), rng.randint(1, 4), rng.randint(1, 3)
        seq = [start]
        cur = start
        d = diff
        for _ in range(5):
            cur += d
            seq.append(cur)
            d += second
        nxt = seq[-1] + d
    c = rng.randint(2, 9)
    op = rng.choice(["mul", "add", "sub", "absdiff"])
    final = nxt * c if op == "mul" else nxt + c if op == "add" else nxt - c if op == "sub" else abs(nxt - c)
    prompt = _styled_prompt(style, "sequence operator", f"Sequence: {', '.join(str(v) for v in seq)}. Find the next value, then apply {op} with {c}.")
    substeps = [
        _step(0, "sequence_pattern", ",".join(map(str, seq)), nxt, "sequence_next", {"rule": rule, "sequence": seq}),
        _step(1, "equation_operator", str(nxt), final, op, {"c": c}),
    ]
    return _make_row(
        family="composed_sequence_operator",
        subfamilies=["sequence_pattern", "equation_operator"],
        difficulty=difficulty,
        prompt_style=style,
        prompt=prompt,
        trace_lines=[f"The unique {rule} sequence next value is {nxt}.", f"Apply {op} with {c} to get {final}."],
        answer=str(final),
        substeps=substeps,
        generator="composed_sequence_operator_v1",
    )


def _build_permutation_mapping(rng: random.Random, index: int, style: str, difficulty: str) -> dict[str, Any]:
    values = rng.sample(range(1, 30), 5)
    op = rng.choice(["sort_asc", "sort_desc", "reverse", "rotate_left"])
    if op == "sort_asc":
        arranged = sorted(values)
    elif op == "sort_desc":
        arranged = sorted(values, reverse=True)
    elif op == "reverse":
        arranged = list(reversed(values))
    else:
        k = rng.randint(1, 4)
        arranged = values[k:] + values[:k]
    take = rng.randint(1, 3)
    final = sum(arranged[:take])
    prompt = _styled_prompt(style, "permutation mapping", f"Apply {op} to {values}, then sum the first {take} values.")
    substeps = [
        _step(0, "permutation_sorting", ",".join(map(str, values)), ",".join(map(str, arranged)), op, {"values": values, "k": locals().get("k", 0)}),
        _step(1, "symbol_mapping", ",".join(map(str, arranged[:take])), final, "sum_first", {"take": take}),
    ]
    return _make_row(
        family="composed_permutation_mapping",
        subfamilies=["permutation_sorting", "symbol_mapping"],
        difficulty=difficulty,
        prompt_style=style,
        prompt=prompt,
        trace_lines=[f"The list becomes {arranged}.", f"The requested sum is {final}."],
        answer=str(final),
        substeps=substeps,
        generator="composed_permutation_mapping_v1",
    )


def _build_word_cipher(rng: random.Random, index: int, style: str, difficulty: str) -> dict[str, Any]:
    bank = ["red", "blue", "green", "amber", "violet", "white", "black", "silver"]
    words = rng.sample(bank, 4)
    op = rng.choice(["first_letters", "last_letters", "reverse_words", "sort_words"])
    if op == "first_letters":
        token = "".join(word[0] for word in words).upper()
    elif op == "last_letters":
        token = "".join(word[-1] for word in words).upper()
    elif op == "reverse_words":
        token = "".join(word[0] for word in reversed(words)).upper()
    else:
        token = "".join(word[0] for word in sorted(words)).upper()
    final = sum(ord(ch) - 64 for ch in token)
    prompt = _styled_prompt(style, "word cipher", f"Words: {' '.join(words)}. Apply {op}, then sum A=1 through Z=26 for the resulting letters.")
    substeps = [
        _step(0, "word_cipher", " ".join(words), token, op, {"words": words}),
        _step(1, "char_cipher", token, final, "a1z26_sum", {}),
    ]
    return _make_row(
        family="composed_word_cipher",
        subfamilies=["word_cipher", "char_cipher"],
        difficulty=difficulty,
        prompt_style=style,
        prompt=prompt,
        trace_lines=[f"The word transform gives {token}.", f"The A1Z26 sum is {final}."],
        answer=str(final),
        substeps=substeps,
        generator="composed_word_cipher_v1",
    )


def _build_gravity_unit(rng: random.Random, index: int, style: str, difficulty: str) -> dict[str, Any]:
    case = rng.choice(["weight", "distance_minutes", "potential"])
    if case == "weight":
        m, g, extra = rng.randint(1, 20), rng.choice([5, 10, 12]), rng.randint(1, 9)
        base = m * g
        final = base + extra
        prompt = _styled_prompt(style, "gravity unit", f"Mass is {m} kg and g={g}. Compute weight in N, then add {extra}.")
        first = _step(0, "gravity_numeric", f"m={m},g={g}", base, "weight", {"m": m, "g": g})
        second = _step(1, "arithmetic", str(base), final, "add", {"c": extra})
    elif case == "distance_minutes":
        minutes, speed = rng.randint(1, 8), rng.randint(2, 12)
        seconds = minutes * 60
        final = seconds * speed
        prompt = _styled_prompt(style, "gravity unit", f"Convert {minutes} minutes to seconds. Speed is {speed} m/s. Find distance.")
        first = _step(0, "unit_conversion", f"{minutes} minutes to seconds", seconds, "unit_convert", {"src": "minutes", "dst": "seconds", "factor": 60})
        second = _step(1, "gravity_numeric", f"speed={speed},time={seconds}", final, "distance", {"speed": speed, "time": seconds})
    else:
        m, g, h = rng.randint(1, 8), rng.choice([5, 10]), rng.randint(2, 9)
        base = m * g
        final = base * h
        prompt = _styled_prompt(style, "gravity unit", f"m={m}, g={g}, h={h}. Compute m*g first, then potential energy m*g*h.")
        first = _step(0, "gravity_numeric", f"m={m},g={g}", base, "weight", {"m": m, "g": g})
        second = _step(1, "gravity_numeric", f"base={base},h={h}", final, "multiply_height", {"base": base, "h": h})
    return _make_row(
        family="composed_gravity_unit",
        subfamilies=["gravity_numeric", "unit_conversion"],
        difficulty=difficulty,
        prompt_style=style,
        prompt=prompt,
        trace_lines=[f"The first verified physics/unit step gives {first['output']}.", f"The final composed value is {second['output']}."],
        answer=str(second["output"]),
        substeps=[first, second],
        generator="composed_gravity_unit_v1",
    )


def _verify_step_output(step: dict[str, Any]) -> str:
    op = str(step.get("op", ""))
    params = dict(step.get("params") or {})
    if op == "custom_decode":
        return str(_decode_base(str(step["input"]), list(params["alphabet"])))
    if op in {"add", "sub", "mul", "absdiff"}:
        x = int(str(step["input"]).split(",")[0])
        c = int(params["c"])
        return str(x + c if op == "add" else x - c if op == "sub" else x * c if op == "mul" else abs(x - c))
    if op == "square":
        x = int(step["input"])
        return str(x * x)
    if op == "affine":
        return str(int(params["a"]) * int(step["input"]) + int(params["b"]))
    if op == "symbol_expression":
        mapping = {str(k): int(v) for k, v in dict(params["mapping"]).items()}
        symbols = list(params["symbols"])
        if any(symbol not in mapping for symbol in symbols):
            raise ValueError("unmapped_symbol")
        kind = params["op"]
        a, b = mapping[symbols[0]], mapping[symbols[1]]
        if kind == "add_sub":
            return str(a + b)
        if kind == "mul_sub":
            return str(a * b)
        if kind == "max_add":
            return str(max(a, b))
        if kind == "abs_add":
            return str(abs(a - b))
        raise ValueError("unsupported_symbol_expression")
    step_family = str(step.get("family", ""))
    if op in {"caesar", "atbash"} or (op == "reverse" and step_family == "char_cipher"):
        text = str(step["input"])
        if op == "caesar":
            return _caesar(text, int(params["shift"]))
        if op == "atbash":
            return _atbash(text)
        return text[::-1]
    if op == "a1z26_sum":
        return str(sum(ord(ch) - 64 for ch in str(step["input"]).upper() if "A" <= ch <= "Z"))
    if op == "unit_convert":
        if re.search(r"\d+\.\d+", str(step["input"])):
            raise ValueError("decimal_rounding_ambiguity")
        amount = int(re.search(r"-?\d+", str(step["input"])).group(0))
        return str(amount * int(params["factor"]))
    if op in {"xor", "and", "or", "not", "reverse", "lshift", "rshift"} and step_family == "bit_manipulation":
        width = int(params.get("width") or 0)
        if op in {"not", "reverse"} and width <= 0:
            raise ValueError("width_required")
        a, b, shift = int(params["a"]), int(params.get("b", 0)), int(params.get("shift", 1))
        if op == "xor":
            return str(a ^ b)
        if op == "and":
            return str(a & b)
        if op == "or":
            return str(a | b)
        if op == "not":
            return str(((1 << width) - 1) ^ a)
        if op == "reverse":
            return str(int(format(a, f"0{width}b")[::-1], 2))
        if op == "lshift":
            return str((a << shift) & ((1 << width) - 1))
        return str(a >> shift)
    if op == "decimal_identity":
        return normalize_answer(step["input"])
    if op == "sequence_next":
        return str(_sequence_next(params["rule"], [int(v) for v in params["sequence"]]))
    if op in {"sort_asc", "sort_desc", "reverse", "rotate_left"} and step_family == "permutation_sorting":
        values = [int(v) for v in params["values"]]
        if len(values) != len(set(values)):
            raise ValueError("duplicate_values")
        if op == "sort_asc":
            arranged = sorted(values)
        elif op == "sort_desc":
            arranged = sorted(values, reverse=True)
        elif op == "reverse":
            arranged = list(reversed(values))
        else:
            k = int(params["k"])
            arranged = values[k:] + values[:k]
        return ",".join(map(str, arranged))
    if op == "sum_first":
        values = [int(v) for v in str(step["input"]).split(",") if v]
        return str(sum(values[: int(params["take"])]))
    if op in {"first_letters", "last_letters", "reverse_words", "sort_words"}:
        words = [str(word).lower() for word in params["words"]]
        if any(not re.fullmatch(r"[a-z]+", word) for word in words):
            raise ValueError("bad_word")
        if op == "first_letters":
            return "".join(word[0] for word in words).upper()
        if op == "last_letters":
            return "".join(word[-1] for word in words).upper()
        if op == "reverse_words":
            return "".join(word[0] for word in reversed(words)).upper()
        return "".join(word[0] for word in sorted(words)).upper()
    if op == "weight":
        return str(int(params["m"]) * int(params["g"]))
    if op == "distance":
        return str(int(params["speed"]) * int(params["time"]))
    if op == "multiply_height":
        return str(int(params["base"]) * int(params["h"]))
    raise ValueError(f"unsupported_step_op:{op}")


def _styled_prompt(style: str, title: str, body: str) -> str:
    if style == "minimal":
        return f"[{title}] {body} Answer only after solving."
    if style == "examples_query":
        return f"[{title}] Example format: do each step exactly. Query: {body}"
    if style == "table":
        return f"[{title}] Table-style task | rule: deterministic | query: {body}"
    if style == "story":
        return f"[{title}] A contest note gives this exact rule. {body}"
    return f"[{title}] {body}"


def _encode_base(value: int, alphabet: list[str]) -> str:
    base = len(alphabet)
    if value == 0:
        return alphabet[0]
    digits: list[str] = []
    current = value
    while current:
        current, rem = divmod(current, base)
        digits.append(alphabet[rem])
    return "".join(reversed(digits))


def _decode_base(text: str, alphabet: list[str]) -> int:
    base = len(alphabet)
    mapping = {symbol: index for index, symbol in enumerate(alphabet)}
    if base < 2 or len(mapping) != base:
        raise ValueError("bad_base")
    value = 0
    for char in text:
        if char not in mapping:
            raise ValueError("unknown_symbol")
        value = value * base + mapping[char]
    return value


def _caesar(text: str, shift: int) -> str:
    if not re.fullmatch(r"[A-Z]+", text):
        raise ValueError("case_or_character_ambiguity")
    return "".join(chr((ord(ch) - 65 + shift) % 26 + 65) for ch in text)


def _atbash(text: str) -> str:
    if not re.fullmatch(r"[A-Z]+", text):
        raise ValueError("case_or_character_ambiguity")
    return "".join(chr(90 - (ord(ch) - 65)) for ch in text)


def _sequence_next(rule: str, seq: list[int]) -> int:
    if rule == "arithmetic":
        if len(seq) < 5:
            raise ValueError("too_short_sequence")
        step = seq[1] - seq[0]
        if not all(seq[i] - seq[i - 1] == step for i in range(1, len(seq))):
            raise ValueError("not_arithmetic")
        return seq[-1] + step
    if rule == "geometric":
        if len(seq) < 5 or seq[0] == 0 or seq[1] % seq[0] != 0:
            raise ValueError("bad_geometric")
        ratio = seq[1] // seq[0]
        if ratio in {0, 1, -1} or not all(seq[i] == seq[i - 1] * ratio for i in range(1, len(seq))):
            raise ValueError("ambiguous_geometric")
        return seq[-1] * ratio
    if rule == "second_order":
        if len(seq) < 6:
            raise ValueError("too_short_second_order")
        diffs = [seq[i] - seq[i - 1] for i in range(1, len(seq))]
        second = diffs[1] - diffs[0]
        if second == 0 or not all(diffs[i] - diffs[i - 1] == second for i in range(1, len(diffs))):
            raise ValueError("ambiguous_second_order")
        return seq[-1] + diffs[-1] + second
    raise ValueError("unsupported_sequence_rule")


def _stable_id(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:24]


def _family_offset(family: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(family))


def _wrong_answer(answer: str, rng: random.Random) -> str:
    if re.fullmatch(r"-?\d+", answer):
        return str(int(answer) + rng.choice([-3, -2, -1, 1, 2, 3]))
    return answer[::-1] if len(answer) > 1 else f"{answer}x"


COMPOSED_TEACHER_REGISTRY: dict[str, ComposedTeacher] = {
    "composed_custom_numeral_arithmetic": ComposedTeacher(
        "composed_custom_numeral_arithmetic",
        ("custom_numeral", "arithmetic"),
        "composed_custom_numeral_arithmetic_v1",
        _build_custom_numeral_arithmetic,
    ),
    "composed_symbol_equation": ComposedTeacher(
        "composed_symbol_equation",
        ("symbol_mapping", "equation_operator"),
        "composed_symbol_equation_v1",
        _build_symbol_equation,
    ),
    "composed_cipher_mapping": ComposedTeacher(
        "composed_cipher_mapping",
        ("char_cipher", "symbol_mapping"),
        "composed_cipher_mapping_v1",
        _build_cipher_mapping,
    ),
    "composed_unit_formula": ComposedTeacher(
        "composed_unit_formula",
        ("unit_conversion", "numeric_formula_safe"),
        "composed_unit_formula_v1",
        _build_unit_formula,
    ),
    "composed_bit_conversion": ComposedTeacher(
        "composed_bit_conversion",
        ("bit_manipulation", "unit_conversion"),
        "composed_bit_conversion_v1",
        _build_bit_conversion,
    ),
    "composed_sequence_operator": ComposedTeacher(
        "composed_sequence_operator",
        ("sequence_pattern", "equation_operator"),
        "composed_sequence_operator_v1",
        _build_sequence_operator,
    ),
    "composed_permutation_mapping": ComposedTeacher(
        "composed_permutation_mapping",
        ("permutation_sorting", "symbol_mapping"),
        "composed_permutation_mapping_v1",
        _build_permutation_mapping,
    ),
    "composed_word_cipher": ComposedTeacher(
        "composed_word_cipher",
        ("word_cipher", "char_cipher"),
        "composed_word_cipher_v1",
        _build_word_cipher,
    ),
    "composed_gravity_unit": ComposedTeacher(
        "composed_gravity_unit",
        ("gravity_numeric", "unit_conversion"),
        "composed_gravity_unit_v1",
        _build_gravity_unit,
    ),
}
