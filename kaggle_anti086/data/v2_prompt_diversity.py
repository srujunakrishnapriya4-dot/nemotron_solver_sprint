from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.v2_corpus_io import read_jsonl, write_json_checked


MAJOR_FAMILIES = {
    "bit_manipulation",
    "char_cipher",
    "gravity_numeric",
    "numeric_formula",
    "symbol_mapping",
    "unit_conversion",
    "word_cipher",
    "roman_numeral",
    "format_only",
}


def build_prompt_diversity_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    template_counts = Counter(prompt_template_signature(str(row.get("prompt", ""))) for row in rows)
    generator_counts = Counter(_metadata(row).get("generator_id") or _metadata(row).get("original_metadata", {}).get("generator_id", "unknown") for row in rows)
    family_counts = Counter(str(row.get("family", "unknown")) for row in rows)
    rule_by_family: dict[str, Counter[str]] = defaultdict(Counter)
    templates_by_family: dict[str, set[str]] = defaultdict(set)
    noise_by_family: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        family = str(row.get("family", "unknown"))
        rule = _rule_signature(row)
        rule_by_family[family][rule] += 1
        templates_by_family[family].add(prompt_template_signature(str(row.get("prompt", ""))))
        noise_by_family[family].add(_noise_profile(row))
    max_template_share = max(template_counts.values(), default=0) / total if total else 0.0
    max_generator_share = max(generator_counts.values(), default=0) / total if total else 0.0
    max_rule_share = {
        family: (max(counter.values(), default=0) / family_counts[family] if family_counts[family] else 0.0)
        for family, counter in sorted(rule_by_family.items())
    }
    violations = []
    warnings = []
    if max_template_share > 0.20:
        violations.append("prompt_template_signature_share_above_20_percent")
    if max_generator_share > 0.35:
        violations.append("generator_share_above_35_percent")
    for family in MAJOR_FAMILIES:
        count = family_counts.get(family, 0)
        if count >= 50 and len(templates_by_family[family]) < 3:
            violations.append(f"{family}_has_fewer_than_3_prompt_templates")
        if count >= 50 and len(noise_by_family[family]) < 2:
            warnings.append(f"{family}_has_fewer_than_2_surface_noise_profiles")
        if count >= 50 and max_rule_share.get(family, 0.0) > 0.10:
            warnings.append(f"{family}_rule_signature_share_above_10_percent")
    return {
        "status": "FAIL" if violations else ("WARN" if warnings else "PASS"),
        "direct_rows": total,
        "max_prompt_template_share": max_template_share,
        "max_generator_share": max_generator_share,
        "max_rule_signature_share_by_family": max_rule_share,
        "templates_per_family": {family: len(values) for family, values in sorted(templates_by_family.items())},
        "surface_noise_profiles_per_family": {family: len(values) for family, values in sorted(noise_by_family.items())},
        "violations": violations,
        "warnings": warnings,
    }


def prompt_template_signature(prompt: str) -> str:
    text = prompt.lower()
    text = re.sub(r"\b[01]{4,}\b", "<BIN>", text)
    text = re.sub(r"\b[ivxlcdm]{2,}\b", "<ROMAN>", text)
    text = re.sub(r"`[^`]+`|\"[^\"]+\"|'[^']+'", "<PHRASE>", text)
    text = re.sub(r"[-+]?\d+(?:\.\d+)?", "<NUM>", text)
    text = re.sub(r"[^\w\s<>/:-]+", "<SYM>", text)
    text = re.sub(r"\s+", " ", text).strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _rule_signature(row: dict[str, Any]) -> str:
    metadata = _metadata(row)
    if metadata.get("rule_signature"):
        return str(metadata["rule_signature"])
    original = metadata.get("original_metadata", {})
    if isinstance(original, dict) and original.get("rule_signature"):
        return str(original["rule_signature"])
    source_rule = str(metadata.get("source_rule_id", ""))
    if source_rule:
        return re.sub(r"_[0-9]{2,}\b", "_<N>", source_rule)
    return f"{row.get('family')}::{row.get('subfamily')}::{row.get('solver_name')}"


def _noise_profile(row: dict[str, Any]) -> str:
    metadata = _metadata(row)
    value = metadata.get("surface_noise_profile") or metadata.get("original_metadata", {}).get("noise_profile")
    if value:
        return str(value)
    prompt = str(row.get("prompt", "")).lower()
    if "\\boxed" in prompt:
        return "boxed"
    if "answer:" in prompt:
        return "answer_prefix"
    if "explanation" in prompt:
        return "explanation_noise"
    if "```" in prompt:
        return "code_fence"
    if '"' in prompt or "`" in prompt:
        return "quoted_examples"
    if ";" in prompt:
        return "semicolon_examples"
    return "clean"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct", default="artifacts/sprint11/train_v2_verified_direct_answer.jsonl")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = build_prompt_diversity_report(read_jsonl(args.direct))
    write_json_checked(args.out, report, field_name="train_v2_prompt_diversity_report")
    print(json.dumps({"status": report["status"], "out": args.out}, sort_keys=True))
    return 0 if report["status"] != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
