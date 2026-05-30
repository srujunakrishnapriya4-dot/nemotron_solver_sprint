from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from .answer_format_policy import normalize_training_answer
from .corpus_deduplicator import deduplicate_corpus
from .corpus_quality_manifest import build_corpus_quality_manifest


def build_win_corpora(
    base_path: str | Path = "artifacts/vex_progress/corpus_mixed_main.jsonl",
    adversarial_path: str | Path = "artifacts/win_system/adversarial_v2_filtered.jsonl",
    contrastive_path: str | Path = "artifacts/anti086/contrastive_error_corpus.jsonl",
    output_dir: str | Path = "artifacts/win_system",
    equation_verified_path: str | Path = "artifacts/win_system/solver_verified_correct.jsonl",
    equation_quarantine_dir: str | Path = "artifacts/win_system",
    equation_direct_cap: int = 500,
    bit_verified_path: str | Path = "artifacts/win_system/solver_verified_correct.jsonl",
) -> dict[str, Any]:
    base = _read_jsonl(Path(base_path))
    quarantine_root = Path(equation_quarantine_dir)
    equation_traces = _equation_trace_rows_from_quarantine(quarantine_root)
    if not equation_traces:
        equation_traces = _equation_trace_rows(Path(equation_verified_path))
    equation_direct = _equation_direct_rows(quarantine_root, cap=equation_direct_cap)
    bit_traces = _bit_trace_rows(Path(bit_verified_path))
    adversarial = [_adversarial_to_row(row) for row in _read_jsonl(Path(adversarial_path))]
    contrastive = _read_jsonl(Path(contrastive_path))
    base_with_traces = base + equation_traces + equation_direct + bit_traces
    corpora = {
        "win_micro.jsonl": _balanced([row for row in base_with_traces if row.get("corpus_type") == "direct_raw"], 64),
        "win_v1.jsonl": _balanced([row for row in base_with_traces if row.get("corpus_type") in {"direct_raw", "family_tagged", "short_rule_trace"}], 6000),
    }
    v2_cap = int(len(corpora["win_v1.jsonl"]) * 0.25 / 0.75)
    corpora["win_v2.jsonl"] = _balanced(corpora["win_v1.jsonl"] + adversarial[:v2_cap], len(corpora["win_v1.jsonl"]) + v2_cap)
    v3_cap = int(len(corpora["win_v2.jsonl"]) * 0.10 / 0.90)
    corpora["win_v3.jsonl"] = _balanced(corpora["win_v2.jsonl"] + contrastive[:v3_cap], len(corpora["win_v2.jsonl"]) + v3_cap)
    corpora = {name: deduplicate_corpus(rows) for name, rows in corpora.items()}
    corpora["win_v2.jsonl"] = _cap_type_ratio(corpora["win_v2.jsonl"], "synthetic_verified", 0.25)
    corpora["win_v3.jsonl"] = _cap_type_ratio(corpora["win_v3.jsonl"], "contrastive_error", 0.10)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in corpora.items():
        _write_jsonl(out / name, rows)
    manifest = build_corpus_quality_manifest(corpora)
    quarantine_manifest = _read_json(quarantine_root / "equation_quarantine_manifest.json")
    manifest.update(
        {
            "equation_safe_trace_count": int(quarantine_manifest.get("equation_safe_trace_count", len(equation_traces))),
            "equation_direct_answer_only_count": int(quarantine_manifest.get("equation_direct_answer_only_count", len(equation_direct))),
            "equation_unsafe_excluded_count": int(quarantine_manifest.get("equation_unsafe_excluded_count", 0)),
            "equation_offline_synthesis_candidate_count": int(quarantine_manifest.get("equation_offline_synthesis_candidate_count", 0)),
            "equation_status": quarantine_manifest.get("equation_status", "ACTIVE" if equation_traces else "QUARANTINED"),
            "bit_safe_trace_count": len(bit_traces),
            "bit_verified_trace_count": len(bit_traces),
            "bit_truth_table_trace_count": sum(1 for row in bit_traces if row.get("rule_id") == "truth_table_bit_mining"),
            "bit_status": "ACTIVE" if bit_traces else "NEEDS_BREAKTHROUGH",
        }
    )
    (out / "win_corpus_manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
    (out / "win_final_candidate_plan.json").write_text(json.dumps({"selection": "gate_based_not_largest", "candidate_stages": ["v1", "v2", "v3"]}, sort_keys=True, indent=2), encoding="utf-8")
    return manifest


def _equation_trace_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in _read_jsonl(path):
        if row.get("family") != "equation_symbolic" or not row.get("prediction"):
            continue
        rule = row.get("reason", "verified_equation_program")
        answer = normalize_training_answer(row["prediction"])
        rows.append(
            {
                "id": f"equation_trace_{row['id']}",
                "source_id": row["id"],
                "family": "equation_symbolic",
                "corpus_type": "short_rule_trace",
                "text": f"User:\nFamily: equation_symbolic\nApply the verified equation rule.\nAssistant:\nRule: {rule}\nAnswer: {answer}",
                "answer": answer,
                "rule_id": rule,
            }
        )
    return rows


def _equation_trace_rows_from_quarantine(root: Path) -> list[dict[str, Any]]:
    rows = []
    for row in _read_jsonl(root / "equation_safe_verified_trace.jsonl"):
        answer = normalize_training_answer(row.get("prediction") or row.get("answer"))
        rule = row.get("rule_id", "verified_equation_program")
        rows.append(
            {
                "id": f"equation_trace_{row['id']}",
                "source_id": row["id"],
                "family": "equation_symbolic",
                "corpus_type": "short_rule_trace",
                "text": f"User:\nFamily: equation_symbolic\nApply the verified equation rule.\nAssistant:\nRule: {rule}\nAnswer: {answer}",
                "answer": answer,
                "rule_id": rule,
            }
        )
    return rows


def _equation_direct_rows(root: Path, *, cap: int) -> list[dict[str, Any]]:
    rows = []
    for row in _read_jsonl(root / "equation_direct_answer_only.jsonl")[:cap]:
        answer = normalize_training_answer(row.get("answer"))
        rows.append(
            {
                "id": f"equation_direct_{row['id']}",
                "source_id": row["id"],
                "family": "equation_symbolic",
                "corpus_type": "direct_raw",
                "text": f"User:\nFamily: equation_symbolic\nReturn only the final answer.\nAssistant:\n{answer}",
                "answer": answer,
                "trace_allowed": False,
            }
        )
    return rows


def _bit_trace_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in _read_jsonl(path):
        if row.get("family") != "bit_manipulation" or not row.get("prediction"):
            continue
        diagnostics = row.get("diagnostics", {}) if isinstance(row.get("diagnostics"), dict) else {}
        rule = diagnostics.get("expr_id") or row.get("reason", "verified_bit_expression")
        answer = normalize_training_answer(row["prediction"])
        rows.append(
            {
                "id": f"bit_trace_{row['id']}",
                "source_id": row["id"],
                "family": "bit_manipulation",
                "corpus_type": "short_rule_trace",
                "text": f"User:\nFamily: bit_manipulation\nApply the verified bit expression.\nAssistant:\nRule: {rule}\nAnswer: {answer}",
                "answer": answer,
                "rule_id": rule,
            }
        )
    return rows


def _adversarial_to_row(row: dict[str, Any]) -> dict[str, Any]:
    answer = normalize_training_answer(row["answer"])
    return {
        "id": row["generation_hash"],
        "source_id": row["generation_hash"],
        "family": row["family"],
        "corpus_type": "synthetic_verified",
        "text": f"User:\n{row['prompt']}\nAssistant:\n{answer}",
        "answer": answer,
        "generation_hash": row["generation_hash"],
        "rule_id": row["rule_id"],
    }


def _balanced(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get("family", "unknown"))].append(row)
    out = []
    while len(out) < limit and any(buckets.values()):
        for family in sorted(buckets):
            if buckets[family] and len(out) < limit:
                out.append(buckets[family].pop(0))
    return out


def _cap_type_ratio(rows: list[dict[str, Any]], corpus_type: str, max_ratio: float) -> list[dict[str, Any]]:
    protected = [row for row in rows if row.get("corpus_type") != corpus_type]
    capped = [row for row in rows if row.get("corpus_type") == corpus_type]
    allowed = int(len(protected) * max_ratio / max(0.0001, 1 - max_ratio))
    return protected + capped[:allowed]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")
