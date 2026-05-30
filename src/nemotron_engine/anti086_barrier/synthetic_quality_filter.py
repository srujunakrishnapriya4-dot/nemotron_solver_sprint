from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from nemotron_engine.core.schemas import stable_hash

from .rule_difficulty_model import reject_reason, score_rule


def filter_synthetic_quality(
    input_path: str | Path = "artifacts/anti086/adversarial_synthetic.jsonl",
    output_dir: str | Path = "artifacts/anti086",
    *,
    max_per_rule: int = 1200,
) -> dict[str, Any]:
    rows = _read_jsonl(Path(input_path))
    kept = []
    rejected = Counter()
    seen_prompt_templates: set[str] = set()
    per_rule = Counter()
    for row in rows:
        score = score_rule(row)
        reason = reject_reason(score)
        template = _template(row.get("prompt", ""))
        if template in seen_prompt_templates:
            reason = reason or "near_duplicate_surface_template"
        if per_rule[row.get("rule_id")] >= max_per_rule:
            reason = reason or "rule_overrepresented"
        if reason:
            rejected[reason] += 1
            continue
        out = dict(row)
        out.update(
            {
                "near_duplicate_score": 0.0,
                "rule_overlap_score": per_rule[row.get("rule_id")] / max_per_rule,
                "answer_prior_score": 0.2,
                "format_stability_score": 1.0,
                "verifier_confidence": 1.0,
                "quality_hash": stable_hash(row),
            }
        )
        kept.append(out)
        seen_prompt_templates.add(template)
        per_rule[row.get("rule_id")] += 1
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "adversarial_synthetic_filtered.jsonl", kept)
    report = {
        "input_count": len(rows),
        "kept_count": len(kept),
        "rejected_counts": dict(sorted(rejected.items())),
        "kept_by_family": dict(sorted(Counter(row["family"] for row in kept).items())),
        "quality_hash": stable_hash([row["quality_hash"] for row in kept]),
    }
    (out_dir / "synthetic_quality_report.json").write_text(json.dumps(report, sort_keys=True, indent=2), encoding="utf-8")
    return report


def _template(prompt: str) -> str:
    return " ".join(prompt.lower().split())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")
