from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from nemotron_engine.core.schemas import stable_hash


HARD_PRIORITY = {"bit_manipulation", "equation_symbolic", "gravity_numeric", "unit_conversion", "cipher_text"}


def build_winmode_curricula(
    vex_corpus_path: str | Path = "artifacts/vex_progress/corpus_mixed_main.jsonl",
    selected_synthetic_path: str | Path = "artifacts/anti086/rule_space_selected_train.jsonl",
    contrastive_path: str | Path = "artifacts/anti086/contrastive_error_corpus.jsonl",
    private_manifest_path: str | Path = "artifacts/anti086/private_like_manifest.json",
    output_dir: str | Path = "artifacts/anti086",
) -> dict[str, Any]:
    base = _exclude(_read_jsonl(Path(vex_corpus_path)), _validation_hashes(Path(private_manifest_path)))
    synthetic = _exclude(_to_corpus_rows(_read_jsonl(Path(selected_synthetic_path)), "synthetic_verified"), _validation_hashes(Path(private_manifest_path)))
    contrastive = _exclude(_read_jsonl(Path(contrastive_path)), _validation_hashes(Path(private_manifest_path)))
    corpora = {
        "winmode_micro.jsonl": _balanced(base, max_rows=64, allowed_types={"direct_raw"}),
        "winmode_v1.jsonl": _balanced(base, max_rows=6000, allowed_types={"direct_raw", "family_tagged", "short_rule_trace"}),
    }
    v2_base = list(corpora["winmode_v1.jsonl"])
    synth_cap = int(len(v2_base) * 0.25 / 0.75)
    corpora["winmode_v2.jsonl"] = _balanced(v2_base + synthetic[:synth_cap], max_rows=len(v2_base) + synth_cap)
    v3_base = list(corpora["winmode_v2.jsonl"])
    contrast_cap = int(len(v3_base) * 0.10 / 0.90)
    corpora["winmode_v3.jsonl"] = _balanced(v3_base + contrastive[:contrast_cap], max_rows=len(v3_base) + contrast_cap)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for filename, rows in corpora.items():
        _write_jsonl(out / filename, rows)
    plan = {
        "selection_policy": "not_automatically_largest",
        "candidate_stages": ("v1", "v2", "v3"),
        "selected_stage": "pending_private_like_gates",
        "fallback": "parent_adapter_if_no_child_beats_parent",
        "required_evidence": ("public_like_no_collapse", "family_hard_improvement", "rule_holdout_improvement"),
        "plan_hash": stable_hash({name: len(rows) for name, rows in corpora.items()}),
    }
    (out / "winmode_final_candidate_plan.json").write_text(json.dumps(plan, sort_keys=True, indent=2), encoding="utf-8")
    report = {
        "file_counts": {filename: len(rows) for filename, rows in corpora.items()},
        "family_counts": {filename: dict(sorted(Counter(row.get("family", "unknown") for row in rows).items())) for filename, rows in corpora.items()},
        "synthetic_ratio_v2": _ratio(corpora["winmode_v2.jsonl"], "synthetic_verified"),
        "contrastive_ratio_v3": _ratio(corpora["winmode_v3.jsonl"], "contrastive_error"),
        "validation_leakage_excluded": True,
        "hard_family_priority": sorted(HARD_PRIORITY),
        "report_hash": stable_hash({filename: [_row_hash(row) for row in rows] for filename, rows in corpora.items()}),
    }
    (out / "winmode_curriculum_report.json").write_text(json.dumps(report, sort_keys=True, indent=2), encoding="utf-8")
    return report


def _to_corpus_rows(rows: list[dict[str, Any]], corpus_type: str) -> list[dict[str, Any]]:
    return [
        {
            "id": row["id"],
            "source_id": row["id"],
            "family": row.get("family", "unknown"),
            "corpus_type": corpus_type,
            "text": f"User:\n{row.get('prompt', '')}\nAssistant:\n{row.get('answer', '')}",
            "answer": row.get("answer", ""),
            "generation_hash": row.get("generation_hash"),
            "rule_id": row.get("rule_id"),
        }
        for row in rows
    ]


def _balanced(rows: list[dict[str, Any]], *, max_rows: int, allowed_types: set[str] | None = None) -> list[dict[str, Any]]:
    if allowed_types is not None:
        rows = [row for row in rows if row.get("corpus_type") in allowed_types]
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(rows, key=lambda item: (str(item.get("family")), str(item.get("id", item.get("source_id", ""))))):
        family = str(row.get("family", "unknown"))
        if family == "roman_numeral" and len(buckets[family]) >= 700:
            continue
        if family == "cipher_text" and row.get("corpus_type") in {"direct_raw", "family_tagged"} and len(buckets[family]) >= 900:
            continue
        buckets[family].append(row)
    output: list[dict[str, Any]] = []
    order = sorted(buckets, key=lambda f: (f not in HARD_PRIORITY, f))
    while len(output) < max_rows and any(buckets.values()):
        for family in order:
            if buckets[family] and len(output) < max_rows:
                output.append(buckets[family].pop(0))
    return output


def _ratio(rows: list[dict[str, Any]], corpus_type: str) -> float:
    return sum(1 for row in rows if row.get("corpus_type") == corpus_type) / max(1, len(rows))


def _row_hash(row: dict[str, Any]) -> str:
    return str(row.get("generation_hash") or row.get("row_hash") or row.get("id") or row.get("source_id"))


def _validation_hashes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return set(payload.get("validation_hashes", []))


def _exclude(rows: list[dict[str, Any]], hashes: set[str]) -> list[dict[str, Any]]:
    return [row for row in rows if _row_hash(row) not in hashes]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")
