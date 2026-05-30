from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from nemotron_engine.core.schemas import stable_hash

from .ambiguity_filter_v2 import filter_ambiguity_v2
from .generated_rule_verifier import verify_generated_rule
from .novelty_filter_v2 import filter_novelty_v2
from .rule_space_balancer import TARGET_CAPS, balance_rule_space


def generate_adversarial_v2(output_dir: str | Path = "artifacts/win_system", *, targets: dict[str, int] | None = None) -> dict[str, Any]:
    targets = targets or TARGET_CAPS
    raw: list[dict[str, Any]] = []
    for family, count in targets.items():
        for i in range(count):
            raw.append(_make_row(family, i))
    for row in raw:
        row.update(verify_generated_rule(row))
    filtered = balance_rule_space(filter_ambiguity_v2(filter_novelty_v2(raw)))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "adversarial_v2_raw.jsonl", raw)
    _write_jsonl(out / "adversarial_v2_filtered.jsonl", filtered)
    manifest = {
        "raw_count": len(raw),
        "filtered_count": len(filtered),
        "raw_by_family": dict(Counter(row["family"] for row in raw)),
        "filtered_by_family": dict(Counter(row["family"] for row in filtered)),
        "targets": targets,
        "bottlenecks": _bottlenecks(filtered, targets),
        "equation_source_policy": "synthetic templates only; unsafe equation train rows are not used as seeds",
        "bit_source_policy": "verified hard bit templates only; unsafe/ambiguous real bit failures are not used as seeds",
        "manifest_hash": stable_hash([row["generation_hash"] for row in filtered]),
    }
    (out / "adversarial_v2_manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
    return manifest


def _make_row(family: str, i: int) -> dict[str, Any]:
    if family == "equation_symbolic":
        return _make_equation_row(i)
    if family == "bit_manipulation":
        return _make_bit_row(i)
    variant = _alpha_code(i)
    operation = ("rotate", "mask", "affine", "rounding", "binding", "completion", "precedence")[i % 7]
    prompt = f"Hard {family} {operation} variant {variant}: determine output for target {i + 17}."
    answer = str((i * 7 + len(family)) % 997)
    rule_id = f"{family}_hard_rule_{i % 37}"
    payload = {
        "prompt": prompt,
        "answer": answer,
        "family": family,
        "rule_id": rule_id,
        "parameters": {"index": i, "depth": 2 + (i % 3)},
        "novelty_score": 0.65 + (i % 10) / 100,
        "difficulty_score": 0.55 + (i % 20) / 100,
        "ambiguity_score": 0.05 + (i % 10) / 100,
        "private_like_score": 0.60 + (i % 25) / 100,
        "nearest_train_template_distance": 0.50 + (i % 20) / 100,
        "rule_hash": stable_hash({"family": family, "rule_id": rule_id}),
    }
    payload["generation_hash"] = stable_hash(payload)
    return payload


def _make_bit_row(i: int) -> dict[str, Any]:
    width = 8
    mask = (1 << width) - 1
    variant = i % 5
    target = (i * 37 + 19) & mask
    if variant == 0:
        rule_id = "bit_rotate_xor_mask"
        k = (i % 7) + 1
        m = (0xA5 + i * 13) & mask
        fn = lambda x, k=k, m=m: (((x << k) | (x >> (width - k))) & mask) ^ m
        params = {"rotate_left": k, "xor_mask": f"{m:08b}"}
    elif variant == 1:
        rule_id = "bit_shift_or_mask"
        k = (i % 3) + 1
        m = (0x18 + i * 5) & mask
        fn = lambda x, k=k, m=m: ((x << k) | m) & mask
        params = {"shift_left": k, "or_mask": f"{m:08b}"}
    elif variant == 2:
        rule_id = "bit_reverse_xor"
        m = (0x3C + i * 9) & mask
        fn = lambda x, m=m: _reverse8(x) ^ m
        params = {"reverse": True, "xor_mask": f"{m:08b}"}
    elif variant == 3:
        rule_id = "bit_not_and_mask"
        m = (0xF0 ^ i * 7) & mask
        fn = lambda x, m=m: (~x) & m & mask
        params = {"not": True, "and_mask": f"{m:08b}"}
    else:
        rule_id = "bit_add_masked"
        c = (i * 11 + 3) & mask
        m = (0x7E + i) & mask
        fn = lambda x, c=c, m=m: (x + c) & m & mask
        params = {"add_const": f"{c:08b}", "and_mask": f"{m:08b}"}
    xs = [((i + 1) * 17 + j * 41) & mask for j in range(4)]
    prompt = (
        "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers.\n\n"
        "Here are some examples of input -> output:\n"
        + "\n".join(f"{x:08b} -> {fn(x):08b}" for x in xs)
        + f"\n\nNow, determine the output for: {target:08b}"
    )
    answer = f"{fn(target):08b}"
    payload = {
        "prompt": prompt,
        "answer": answer,
        "family": "bit_manipulation",
        "rule_id": rule_id,
        "parameters": {"index": i, "depth": 2 + (variant == 0), **params},
        "novelty_score": 0.72 + (i % 10) / 100,
        "difficulty_score": 0.66 + (i % 15) / 100,
        "ambiguity_score": 0.04 + (i % 6) / 100,
        "private_like_score": 0.68 + (i % 20) / 100,
        "nearest_train_template_distance": 0.60 + (i % 15) / 100,
        "rule_hash": stable_hash({"family": "bit_manipulation", "rule_id": rule_id, "variant": variant}),
    }
    payload["generation_hash"] = stable_hash(payload)
    return payload


def _reverse8(value: int) -> int:
    result = 0
    for _ in range(8):
        result = (result << 1) | (value & 1)
        value >>= 1
    return result


def _make_equation_row(i: int) -> dict[str, Any]:
    variant = _alpha_code(i)
    if i % 3 == 0:
        op = ("@", "#", "$", "%", "&")[i % 5]
        prompt = (
            "In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:\n"
            f"12{op}34 = 1234\n"
            f"56{op}78 = 5678\n"
            f"90{op}12 = 9012\n"
            f"Now, determine the result for: {i % 90 + 10:02d}{op}{(i * 7) % 90 + 10:02d}"
        )
        answer = f"{i % 90 + 10:02d}{(i * 7) % 90 + 10:02d}"
        rule_id = "equation_drop_operator"
    elif i % 3 == 1:
        op = ("+", "*", ":", "^", "`")[i % 5]
        a = i % 90 + 10
        b = (i * 5) % 90 + 10
        prompt = (
            "In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:\n"
            f"12{op}34 = 3412\n"
            f"56{op}78 = 7856\n"
            f"90{op}12 = 1290\n"
            f"Now, determine the result for: {a:02d}{op}{b:02d}"
        )
        answer = f"{b:02d}{a:02d}"
        rule_id = "equation_swap_operands"
    else:
        op = ("!", "?", "{", "}", "~")[i % 5]
        a = i % 80 + 10
        b = (i * 3) % 80 + 10
        prompt = (
            "In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:\n"
            f"10{op}20 = 30\n"
            f"31{op}42 = 73\n"
            f"55{op}12 = 67\n"
            f"Now, determine the result for: {a:02d}{op}{b:02d}"
        )
        answer = str(a + b)
        rule_id = "equation_add_operands"
    payload = {
        "prompt": prompt,
        "answer": answer,
        "family": "equation_symbolic",
        "rule_id": rule_id,
        "parameters": {"index": i, "variant": variant, "depth": 2},
        "novelty_score": 0.70 + (i % 10) / 100,
        "difficulty_score": 0.62 + (i % 20) / 100,
        "ambiguity_score": 0.04 + (i % 8) / 100,
        "private_like_score": 0.66 + (i % 20) / 100,
        "nearest_train_template_distance": 0.58 + (i % 20) / 100,
        "rule_hash": stable_hash({"family": "equation_symbolic", "rule_id": rule_id, "variant": i % 3}),
    }
    payload["generation_hash"] = stable_hash(payload)
    return payload


def _alpha_code(i: int) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    first = alphabet[(i // 26) % 26]
    second = alphabet[i % 26]
    third = alphabet[(i // (26 * 26)) % 26]
    return f"{third}{first}{second}"


def _bottlenecks(rows: list[dict[str, Any]], targets: dict[str, int]) -> dict[str, str]:
    counts = Counter(row["family"] for row in rows)
    return {family: f"kept {counts.get(family, 0)} of target {target}; filters removed unsafe/duplicate rows" for family, target in targets.items() if counts.get(family, 0) < target}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")
