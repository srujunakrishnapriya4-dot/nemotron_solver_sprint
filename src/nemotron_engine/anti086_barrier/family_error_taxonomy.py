from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nemotron_engine.core.schemas import stable_hash


TAXONOMY = {
    "equation_symbolic": {
        "known_easy_cases": ("exact lookup",),
        "known_hard_cases": ("operator remapping", "precedence ambiguity", "symbol-digit binding", "exact division/modulo"),
        "likely_hidden_traps": ("nonstandard operator semantics", "mixed symbol/digit alphabets"),
        "synthetic_generation_strategy": "high-diversity operator tables and precedence traps",
        "validation_strategy": "rule-holdout and family-hard",
        "training_examples_to_include": ("verified operator traces", "contrastive wrong-rule rejections"),
        "examples_to_reject": ("non-exact division under integer answer", "underdetermined one-example operators"),
        "answer_format_risks": ("symbol strings", "integers", "operator output normalization"),
        "weight": "boost",
    },
    "bit_manipulation": {
        "known_easy_cases": ("not", "reverse", "single rotate"),
        "known_hard_cases": ("depth 3 boolean compositions", "choice/majority", "mask-conditioned endian traps"),
        "likely_hidden_traps": ("ambiguous rotations", "position-dependent masks"),
        "synthetic_generation_strategy": "low-overlap composition templates only",
        "validation_strategy": "composition shift",
        "training_examples_to_include": ("verified hard bit rows",),
        "examples_to_reject": ("single-transform duplicates", "ambiguous target outputs"),
        "answer_format_risks": ("leading-zero bitstrings",),
        "weight": "boost",
    },
    "gravity_numeric": {"known_easy_cases": ("ratio fit",), "known_hard_cases": ("rounding boundary", "range extrapolation"), "likely_hidden_traps": ("decimal precision shifts",), "synthetic_generation_strategy": "rounding-edge cases", "validation_strategy": "family-hard", "training_examples_to_include": ("rounded verified examples",), "examples_to_reject": ("unstable decimal formatting",), "answer_format_risks": ("decimal strings",), "weight": "boost"},
    "unit_conversion": {"known_easy_cases": ("ratio",), "known_hard_cases": ("affine ambiguity", "distractor units"), "likely_hidden_traps": ("format precision",), "synthetic_generation_strategy": "ratio/affine contrast", "validation_strategy": "rule-holdout", "training_examples_to_include": ("affine and ratio verified rows",), "examples_to_reject": ("ambiguous ratio-affine disagreements",), "answer_format_risks": ("trailing zeros",), "weight": "boost"},
    "cipher_symbol_hard": {"known_easy_cases": ("seen char bijection",), "known_hard_cases": ("unseen-token completion", "symbol-digit composition"), "likely_hidden_traps": ("collision risk",), "synthetic_generation_strategy": "collision-safe bijections", "validation_strategy": "surface shift", "training_examples_to_include": ("only unique completions",), "examples_to_reject": ("many-to-one collisions",), "answer_format_risks": ("lowercase text and symbols",), "weight": "moderate"},
    "roman_numeral": {"known_easy_cases": ("standard roman conversion",), "known_hard_cases": ("none dominant",), "likely_hidden_traps": ("format only",), "synthetic_generation_strategy": "do not add volume", "validation_strategy": "public-like sanity", "training_examples_to_include": ("small retention set",), "examples_to_reject": ("easy duplicates",), "answer_format_risks": ("uppercase roman",), "weight": "downweight"},
}


def build_family_error_taxonomy(output_json: str | Path = "artifacts/anti086/family_error_taxonomy.json", output_md: str | Path = "artifacts/anti086/FAMILY_ERROR_TAXONOMY.md") -> dict[str, Any]:
    payload = {"families": TAXONOMY, "roman_and_easy_cipher_downweighted": True}
    payload["taxonomy_hash"] = stable_hash(payload)
    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    Path(output_md).write_text(_markdown(payload), encoding="utf-8")
    return payload


def _markdown(payload: dict[str, Any]) -> str:
    lines = ["# Family Error Taxonomy", ""]
    for family, spec in payload["families"].items():
        lines.append(f"## {family}")
        lines.append(f"Weight: {spec['weight']}")
        lines.append("Hard cases: " + ", ".join(spec["known_hard_cases"]))
        lines.append("")
    return "\n".join(lines)
