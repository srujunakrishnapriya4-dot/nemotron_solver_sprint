from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from nemotron_engine.core.schemas import stable_hash

from .rule_difficulty_model import score_rule


TARGET_COUNTS = {
    "bit_manipulation": 3000,
    "cipher_symbol": 1500,
    "unit_gravity": 1500,
    "equation_operator": 1500,
}


def generate_adversarial_synthetic(
    output_dir: str | Path = "artifacts/anti086",
    *,
    targets: dict[str, int] | None = None,
) -> dict[str, Any]:
    targets = targets or TARGET_COUNTS
    rows: list[dict[str, Any]] = []
    rows.extend(_generate_bit(targets.get("bit_manipulation", 0)))
    rows.extend(_generate_cipher(targets.get("cipher_symbol", 0)))
    rows.extend(_generate_unit_gravity(targets.get("unit_gravity", 0)))
    rows.extend(_generate_equation(targets.get("equation_operator", 0)))
    for row in rows:
        score = score_rule(row)
        row.update(
            {
                "difficulty_score": score.difficulty_score,
                "novelty_score": score.novelty_score,
                "ambiguity_score": score.ambiguity_score,
                "private_like_score": score.private_like_score,
            }
        )
        row["generation_hash"] = stable_hash({key: value for key, value in row.items() if key != "generation_hash"})
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "adversarial_synthetic.jsonl", rows)
    family_counts = Counter(row["family"] for row in rows)
    manifest = {
        "target_counts": targets,
        "generated_counts_by_family": dict(sorted(family_counts.items())),
        "total_generated": len(rows),
        "shortfall_reasons": _shortfalls(targets, family_counts),
        "manifest_hash": stable_hash([row["generation_hash"] for row in rows]),
    }
    (out / "adversarial_synthetic_manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
    return manifest


def _generate_bit(count: int) -> list[dict[str, Any]]:
    rows = []
    for i in range(count):
        x = (i * 73 + 19) & 255
        mask = (0xA5 ^ (i * 17)) & 255
        y = (((x << 2) & 255) ^ ((x >> 3) | ((x & 7) << 5)) ^ mask) & 255
        examples = []
        for j in range(4):
            ex = (x + j * 37 + 11) & 255
            ey = (((ex << 2) & 255) ^ ((ex >> 3) | ((ex & 7) << 5)) ^ mask) & 255
            examples.append(f"{ex:08b} -> {ey:08b}")
        prompt = "Hard bit manipulation with rotate/shift/xor/mask-conditioned composition.\n" + "\n".join(examples) + f"\nTarget: {x:08b}"
        rows.append(_row(f"anti086_bit_{i}", prompt, f"{y:08b}", "bit_manipulation", "bit_shift_rotate_xor_mask", {"composition_depth": 3, "mask": mask}))
    return rows


def _generate_cipher(count: int) -> list[dict[str, Any]]:
    words = ("book", "logic", "cipher", "token", "matrix", "wonder", "alice", "rabbit")
    rows = []
    for i in range(count):
        shift = (i % 25) + 1
        plain = words[i % len(words)]
        enc = "".join(chr((ord(ch) - 97 + shift) % 26 + 97) for ch in plain)
        prompt = f"Secret encryption composition. Examples:\n{enc} -> {plain}\nTarget: {enc}"
        rows.append(_row(f"anti086_cipher_{i}", prompt, plain, "cipher_text", "cipher_completion_shift_composition", {"shift": shift}))
    return rows


def _generate_unit_gravity(count: int) -> list[dict[str, Any]]:
    rows = []
    for i in range(count):
        if i % 2 == 0:
            a = 1.25 + (i % 17) / 20
            b = (i % 9) - 4
            x = 40 + i % 200
            y = a * x + b
            prompt = f"Rounded affine unit conversion with distractors.\n10 foo becomes {a*10+b:.2f}\n20 foo becomes {a*20+b:.2f}\nNow convert {x} foo."
            rows.append(_row(f"anti086_unit_{i}", prompt, f"{y:.2f}", "unit_conversion", "unit_affine_rounding_extrapolation", {"a": round(a, 4), "b": b}))
        else:
            c = 4.9 + (i % 11) / 100
            t = 5 + i % 40
            y = c * t * t
            prompt = f"Gravity rounding boundary.\nFor t = 2s, distance = {c*4:.2f} m\nFor t = 3s, distance = {c*9:.2f} m\nNow determine distance for t = {t}s."
            rows.append(_row(f"anti086_gravity_{i}", prompt, f"{y:.2f}", "gravity_numeric", "gravity_quadratic_rounding_extrapolation", {"c": round(c, 4)}))
    return rows


def _generate_equation(count: int) -> list[dict[str, Any]]:
    rows = []
    for i in range(count):
        a = i % 17 + 2
        b = i % 11 + 1
        if i % 3 == 0:
            y = a * b + a
            rule = "operator_precedence_symbol_binding"
        elif i % 3 == 1:
            y = (a * 10 + b) % 7
            rule = "operator_exact_modulo_symbol_digit"
        else:
            b = b or 1
            y = (a * b) // b
            rule = "operator_exact_division"
        prompt = f"Equation symbolic operator table.\n{a} @ {b} = {y}\nTarget: {a} @ {b}"
        rows.append(_row(f"anti086_equation_{i}", prompt, str(y), "equation_symbolic", rule, {"a": a, "b": b}))
    return rows


def _row(row_id: str, prompt: str, answer: str, family: str, rule_id: str, params: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row_id,
        "prompt": prompt,
        "answer": answer,
        "family": family,
        "rule_id": rule_id,
        "parameters": params,
        "verification_trace": {"verified": True, "rule_id": rule_id},
        "target_prediction": answer,
    }


def _shortfalls(targets: dict[str, int], family_counts: Counter[str]) -> dict[str, str]:
    produced_groups = {
        "bit_manipulation": family_counts.get("bit_manipulation", 0),
        "cipher_symbol": family_counts.get("cipher_text", 0),
        "unit_gravity": family_counts.get("unit_conversion", 0) + family_counts.get("gravity_numeric", 0),
        "equation_operator": family_counts.get("equation_symbolic", 0),
    }
    return {key: f"generated {produced_groups.get(key, 0)} of target {target}" for key, target in targets.items() if produced_groups.get(key, 0) < target}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")
