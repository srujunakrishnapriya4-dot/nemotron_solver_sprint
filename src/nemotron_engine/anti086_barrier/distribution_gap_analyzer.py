from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from typing import Any

from nemotron_engine.core.schemas import stable_hash


FAMILIES = (
    "bit_manipulation",
    "cipher_text",
    "roman_numeral",
    "unit_conversion",
    "gravity_numeric",
    "equation_symbolic",
    "modular_numeric",
    "sequence_numeric",
    "string_transform",
)


def analyze_distribution_gap(
    train_csv_path: str | Path = "data/nemotron_competition/train.csv",
    verified_rules_path: str | Path = "artifacts/vex_progress/verified_rules.jsonl",
    corpus_manifest_path: str | Path = "artifacts/vex_progress/corpus_manifest.json",
    adapter_manifest_path: str | Path = "artifacts/adapter_training/dataset_manifest.json",
    output_path: str | Path = "artifacts/anti086/distribution_gap_report.json",
) -> dict[str, Any]:
    train_rows = _read_train(Path(train_csv_path))
    verified = _read_jsonl(Path(verified_rules_path))
    corpus_manifest = _read_json(Path(corpus_manifest_path))
    adapter_manifest = _read_json(Path(adapter_manifest_path))
    by_family = {family: _family_stats(family, train_rows, verified, corpus_manifest, adapter_manifest) for family in FAMILIES}
    hard_blockers = [family for family, stats in by_family.items() if stats["private_shift_risk"] >= 0.65]
    report = {
        "families": by_family,
        "suspected_ceiling_causes": (
            "train/public-like validation overweights memorized surface templates",
            "easy synthetic volume increases overlap without improving rule holdout",
            "hard bit/equation/unit/gravity/cipher cases need adversarial verified curricula",
            "random validation misses composition and surface-shift failure modes",
        ),
        "families_most_likely_to_block_0_95": hard_blockers,
        "families_where_more_easy_synthetic_is_useless": tuple(f for f in ("roman_numeral", "cipher_text", "bit_manipulation") if by_family[f]["likely_memorization_risk"] >= 0.55),
        "families_needing_adversarial_rule_generation": tuple(f for f in hard_blockers if f != "roman_numeral"),
        "families_needing_contrastive_negative_examples": tuple(f for f in ("bit_manipulation", "equation_symbolic", "unit_conversion", "gravity_numeric") if by_family[f]["abstained"] or by_family[f]["solver_verified_wrong"]),
        "expected_conclusion": "Do not blindly add 20K easy binary examples; generate high-difficulty, low-overlap, solver-verified examples only.",
    }
    report["report_hash"] = stable_hash(report)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, sort_keys=True, indent=2), encoding="utf-8")
    return report


def _family_stats(family: str, train_rows: list[dict[str, str]], verified: list[dict[str, Any]], corpus_manifest: dict[str, Any], adapter_manifest: dict[str, Any]) -> dict[str, Any]:
    train_family = [_detect_family(row.get("prompt", "")) for row in train_rows]
    train_count = sum(1 for item in train_family if item == family)
    vf = [row for row in verified if row.get("family") == family]
    status = Counter(row.get("verified_status") for row in vf)
    prompts = [row.get("prompt", "") for row in train_rows if _detect_family(row.get("prompt", "")) == family]
    answers = [row.get("answer", "") for row in train_rows if _detect_family(row.get("prompt", "")) == family]
    template_div = len({_surface_template(prompt) for prompt in prompts})
    answer_div = len(set(answers))
    rule_div = len(set(row.get("rule_id") or row.get("rejection_reason") for row in vf))
    generated_count = int(corpus_manifest.get("family_counts", {}).get(family, 0))
    overlap = min(1.0, train_count / max(1, generated_count)) if generated_count else 0.0
    memorization = min(1.0, (train_count / max(1, template_div + 1)) / 100.0 + overlap * 0.4)
    private_shift = min(1.0, 0.25 + (status.get("abstained", 0) / max(1, len(vf))) * 0.45 + (status.get("verified_wrong", 0) / max(1, len(vf))) * 0.3)
    return {
        "train_count": train_count,
        "solver_verified_correct": status.get("verified_correct", 0),
        "solver_verified_wrong": status.get("verified_wrong", 0),
        "abstained": status.get("abstained", 0),
        "rule_diversity": rule_div,
        "surface_template_diversity": template_div,
        "target_answer_diversity": answer_div,
        "train_vs_generated_distribution_overlap": round(overlap, 6),
        "likely_memorization_risk": round(memorization, 6),
        "private_shift_risk": round(private_shift, 6),
    }


def _detect_family(prompt: str) -> str:
    lowered = prompt.lower()
    if "bit manipulation" in lowered:
        return "bit_manipulation"
    if "secret encryption" in lowered:
        return "cipher_text"
    if "numeral system" in lowered:
        return "roman_numeral"
    if "unit conversion" in lowered:
        return "unit_conversion"
    if "gravitational" in lowered:
        return "gravity_numeric"
    if "transformation rules" in lowered and "equation" in lowered:
        return "equation_symbolic"
    return "unknown"


def _surface_template(prompt: str) -> str:
    return "".join("0" if ch.isdigit() else ("a" if ch.isalpha() else ch) for ch in prompt[:300])


def _read_train(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
