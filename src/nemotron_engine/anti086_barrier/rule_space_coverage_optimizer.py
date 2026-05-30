from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from nemotron_engine.core.schemas import stable_hash


HARD_FAMILIES = {"bit_manipulation", "equation_symbolic", "gravity_numeric", "unit_conversion", "cipher_text"}


def optimize_rule_space_coverage(
    filtered_synthetic_path: str | Path = "artifacts/anti086/adversarial_synthetic_filtered.jsonl",
    verified_rules_path: str | Path = "artifacts/vex_progress/verified_rules.jsonl",
    private_manifest_path: str | Path = "artifacts/anti086/private_like_manifest.json",
    output_dir: str | Path = "artifacts/anti086",
    *,
    train_limit: int = 4000,
    holdout_limit: int = 600,
    per_rule_cap: int = 24,
    per_template_cap: int = 8,
) -> dict[str, Any]:
    """Select a compact, high-diversity synthetic subset under a token budget."""

    rows = _read_jsonl(Path(filtered_synthetic_path))
    validation_hashes = _validation_hashes(Path(private_manifest_path))
    verified_rule_ids = _verified_rule_ids(Path(verified_rules_path))
    candidates = [
        row
        for row in rows
        if _row_hash(row) not in validation_hashes
        and float(row.get("ambiguity_score", 1.0)) <= 0.35
        and float(row.get("novelty_score", 0.0)) >= 0.45
        and float(row.get("difficulty_score", 0.0)) >= 0.45
    ]
    ordered = sorted(
        candidates,
        key=lambda row: (
            str(row.get("family")) not in HARD_FAMILIES,
            -float(row.get("private_like_score", 0.0)),
            -float(row.get("novelty_score", 0.0)),
            str(row.get("family")),
            str(row.get("rule_id")),
            str(row.get("id")),
        ),
    )
    unique_candidate_rules = {str(row.get("rule_id", "unknown")) for row in ordered}
    target_holdout_rule_count = min(
        max(1, holdout_limit // max(1, per_rule_cap)),
        max(1, len(unique_candidate_rules) // 4),
        max(1, len(unique_candidate_rules) - 1),
    )
    selected: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    train_rules: set[str] = set()
    holdout_rules: set[str] = set()
    rule_counts: Counter[str] = Counter()
    train_template_counts: Counter[str] = Counter()
    holdout_template_counts: Counter[str] = Counter()
    rejected = Counter()
    for row in ordered:
        rule_id = str(row.get("rule_id", "unknown"))
        template = _surface_template(row)
        template_key = f"{rule_id}|{template}"
        if rule_counts[rule_id] >= per_rule_cap:
            rejected["rule_cap"] += 1
            continue
        if len(holdout) < holdout_limit and rule_id not in train_rules and len(holdout_rules) < target_holdout_rule_count:
            if holdout_template_counts[template_key] >= max(2, per_template_cap // 2):
                rejected["holdout_template_cap"] += 1
                continue
            holdout.append(row)
            holdout_rules.add(rule_id)
            holdout_template_counts[template_key] += 1
            continue
        if rule_id in holdout_rules:
            rejected["heldout_rule_excluded_from_train"] += 1
            continue
        if train_template_counts[template_key] >= per_template_cap:
            rejected["near_duplicate_template_cap"] += 1
            continue
        if len(selected) < train_limit:
            selected.append(row)
            train_rules.add(rule_id)
            rule_counts[rule_id] += 1
            train_template_counts[template_key] += 1
        else:
            rejected["train_limit"] += 1
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "rule_space_selected_train.jsonl", selected)
    _write_jsonl(out / "rule_space_holdout.jsonl", holdout)
    report = {
        "selected_count": len(selected),
        "holdout_count": len(holdout),
        "selected_by_family": dict(sorted(Counter(row.get("family", "unknown") for row in selected).items())),
        "holdout_by_family": dict(sorted(Counter(row.get("family", "unknown") for row in holdout).items())),
        "train_rule_count": len(train_rules),
        "holdout_rule_count": len(holdout_rules),
        "rule_holdout_disjoint": train_rules.isdisjoint(holdout_rules),
        "verified_rule_ids_seen": len(verified_rule_ids),
        "rejected_counts": dict(sorted(rejected.items())),
        "optimizer_policy": {
            "train_limit": train_limit,
            "holdout_limit": holdout_limit,
            "per_rule_cap": per_rule_cap,
            "per_template_cap": per_template_cap,
            "hard_family_priority": sorted(HARD_FAMILIES),
        },
        "report_hash": stable_hash(
            {
                "selected": [_row_hash(row) for row in selected],
                "holdout": [_row_hash(row) for row in holdout],
                "policy": [train_limit, holdout_limit, per_rule_cap, per_template_cap],
            }
        ),
    }
    (out / "rule_space_coverage_report.json").write_text(json.dumps(report, sort_keys=True, indent=2), encoding="utf-8")
    return report


def _surface_template(row: dict[str, Any]) -> str:
    prompt = str(row.get("prompt", "")).lower()
    return " ".join("0" if token.replace(".", "", 1).isdigit() else token for token in prompt.split()[:24])


def _row_hash(row: dict[str, Any]) -> str:
    return str(row.get("generation_hash") or row.get("row_hash") or row.get("id"))


def _validation_hashes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return set(payload.get("validation_hashes", []))


def _verified_rule_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row.get("rule_id")) for row in _read_jsonl(path) if row.get("verified_status") == "verified_correct"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")
