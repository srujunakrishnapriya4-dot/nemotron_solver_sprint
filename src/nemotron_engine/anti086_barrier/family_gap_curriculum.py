from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any
import zipfile

from nemotron_engine.core.schemas import stable_hash


def build_family_gap_curricula(
    vex_corpus_path: str | Path = "artifacts/vex_progress/corpus_mixed_main.jsonl",
    filtered_synthetic_path: str | Path = "artifacts/anti086/adversarial_synthetic_filtered.jsonl",
    contrastive_path: str | Path = "artifacts/anti086/contrastive_error_corpus.jsonl",
    private_manifest_path: str | Path = "artifacts/anti086/private_like_manifest.json",
    output_dir: str | Path = "artifacts/anti086",
) -> dict[str, Any]:
    base = _read_jsonl(Path(vex_corpus_path))
    synthetic = _to_corpus_rows(_read_jsonl(Path(filtered_synthetic_path)), "synthetic_verified")
    contrastive = _read_jsonl(Path(contrastive_path))
    val_hashes = _validation_hashes(Path(private_manifest_path))
    base = _exclude(base, val_hashes)
    synthetic = _exclude(synthetic, val_hashes)
    contrastive = _exclude(contrastive, val_hashes)
    corpora = {
        "corpus_anti086_micro.jsonl": _balanced(base, max_rows=64, allowed_types={"direct_raw"}),
        "corpus_anti086_v1.jsonl": _balanced(base, max_rows=6000, allowed_types={"direct_raw", "family_tagged", "short_rule_trace"}),
    }
    v2_base = list(corpora["corpus_anti086_v1.jsonl"])
    v2_syn_cap = int(len(v2_base) * 0.25 / 0.75)
    corpora["corpus_anti086_v2.jsonl"] = _balanced(v2_base + synthetic[:v2_syn_cap], max_rows=len(v2_base) + v2_syn_cap)
    v3_base = list(corpora["corpus_anti086_v2.jsonl"])
    contrastive_cap = int(len(v3_base) * 0.10 / 0.90)
    corpora["corpus_anti086_v3.jsonl"] = _balanced(v3_base + contrastive[:contrastive_cap], max_rows=len(v3_base) + contrastive_cap)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for filename, rows in corpora.items():
        _write_jsonl(out / filename, rows)
    manifest = {
        "file_counts": {filename: len(rows) for filename, rows in corpora.items()},
        "family_counts": {filename: dict(sorted(Counter(row.get("family") for row in rows).items())) for filename, rows in corpora.items()},
        "synthetic_ratio_v2": _ratio(corpora["corpus_anti086_v2.jsonl"], "synthetic_verified"),
        "contrastive_ratio_v3": _ratio(corpora["corpus_anti086_v3.jsonl"], "contrastive_error"),
        "validation_leakage_excluded": True,
        "manifest_hash": stable_hash({filename: [row.get("row_hash") or row.get("generation_hash") or row.get("id") for row in rows] for filename, rows in corpora.items()}),
    }
    (out / "curriculum_manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
    _zip_input(out, out / "anti086_kaggle_input.zip")
    return manifest


def _to_corpus_rows(rows: list[dict[str, Any]], corpus_type: str) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        output.append(
            {
                "id": row["id"],
                "source_id": row["id"],
                "family": row["family"],
                "corpus_type": corpus_type,
                "text": f"User:\n{row['prompt']}\nAssistant:\n{row['answer']}",
                "answer": row["answer"],
                "generation_hash": row.get("generation_hash"),
                "rule_id": row.get("rule_id"),
            }
        )
    return output


def _balanced(rows: list[dict[str, Any]], *, max_rows: int, allowed_types: set[str] | None = None) -> list[dict[str, Any]]:
    if allowed_types is not None:
        rows = [row for row in rows if row.get("corpus_type") in allowed_types]
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        family = str(row.get("family", "unknown"))
        if family == "roman_numeral" and len(buckets[family]) > 800:
            continue
        if family == "cipher_text" and row.get("rule_id") in {None, "solved"} and len(buckets[family]) > 900:
            continue
        buckets[family].append(row)
    output = []
    while len(output) < max_rows and any(buckets.values()):
        for family in sorted(buckets, key=lambda f: (f in {"roman_numeral", "cipher_text"}, f)):
            if buckets[family] and len(output) < max_rows:
                output.append(buckets[family].pop(0))
    return output


def _ratio(rows: list[dict[str, Any]], corpus_type: str) -> float:
    return sum(1 for row in rows if row.get("corpus_type") == corpus_type) / max(1, len(rows))


def _validation_hashes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    hashes = set(payload.get("validation_hashes", []))
    return hashes


def _exclude(rows: list[dict[str, Any]], hashes: set[str]) -> list[dict[str, Any]]:
    return [row for row in rows if (row.get("generation_hash") or row.get("row_hash") or row.get("id")) not in hashes]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")


def _zip_input(source: Path, target: Path) -> None:
    names = [
        "distribution_gap_report.json",
        "adversarial_synthetic_filtered.jsonl",
        "synthetic_quality_report.json",
        "private_like_manifest.json",
        "contrastive_error_corpus.jsonl",
        "contrastive_manifest.json",
        "curriculum_manifest.json",
        "corpus_anti086_micro.jsonl",
        "corpus_anti086_v1.jsonl",
        "corpus_anti086_v2.jsonl",
        "corpus_anti086_v3.jsonl",
    ]
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            path = source / name
            if path.exists():
                archive.write(path, name)
