from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any
import zipfile

from nemotron_engine.core.schemas import stable_hash

from .token_corpus_schema import VexTokenCorpusRow, row_to_json


def build_vex_text_corpus(
    verified_rules_path: str | Path = "artifacts/vex_progress/verified_rules.jsonl",
    output_dir: str | Path = "artifacts/vex_progress",
    *,
    raw_answer_style: bool = True,
) -> dict[str, Any]:
    rows = _read_jsonl(Path(verified_rules_path))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    direct: list[VexTokenCorpusRow] = []
    tagged: list[VexTokenCorpusRow] = []
    traces: list[VexTokenCorpusRow] = []
    for row in rows:
        if not row.get("answer") or row.get("verified_status") == "parse_failed":
            continue
        answer = str(row["answer"])
        direct.append(_corpus_row(row, "direct_raw", f"User:\n{row['prompt']}\nAssistant:\n{answer}", answer))
        tagged.append(_corpus_row(row, "family_tagged", f"User:\nFamily: {row['family']}.\n{row['prompt']}\nAnswer only.\nAssistant:\n{answer}", answer))
        if row.get("verified_status") == "verified_correct":
            trace = f"User:\n{row['prompt']}\nAssistant:\nRule: {row.get('rule_id') or 'verified symbolic rule'}\nAnswer: {answer}"
            traces.append(_corpus_row(row, "short_rule_trace", trace, answer))
    mixed = list(direct) + list(tagged) + list(traces)
    files = {
        "corpus_direct_raw.jsonl": direct,
        "corpus_family_tagged.jsonl": tagged,
        "corpus_short_rule_trace.jsonl": traces,
        "corpus_mixed_main.jsonl": mixed,
    }
    hashes: dict[str, str] = {}
    counts: dict[str, int] = {}
    family_counts: Counter[str] = Counter()
    for filename, corpus_rows in files.items():
        path = out / filename
        _write_corpus(path, corpus_rows)
        hashes[filename] = stable_hash([row.row_hash for row in corpus_rows])
        counts[filename] = len(corpus_rows)
        if filename == "corpus_mixed_main.jsonl":
            family_counts.update(row.family for row in corpus_rows)
    manifest = {
        "file_counts": counts,
        "corpus_rows_by_type": {
            "direct_raw": len(direct),
            "family_tagged": len(tagged),
            "short_rule_trace": len(traces),
            "synthetic_verified": 0,
        },
        "family_counts": dict(sorted(family_counts.items())),
        "artifact_hashes": hashes,
        "raw_answer_style": raw_answer_style,
        "no_test_labels_or_ids": True,
        "manifest_hash": stable_hash({"counts": counts, "hashes": hashes}),
    }
    (out / "corpus_manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
    _zip_vex_input(out, out / "vex_kaggle_input.zip")
    return manifest


def _corpus_row(source: dict[str, Any], corpus_type: str, text: str, answer: str) -> VexTokenCorpusRow:
    row_id = f"{source['id']}:{corpus_type}"
    return VexTokenCorpusRow(
        row_id=row_id,
        source_id=str(source["id"]),
        family=str(source["family"]),
        corpus_type=corpus_type,
        text=text,
        answer=answer,
        metadata={"verified_status": source.get("verified_status"), "rule_id": source.get("rule_id")},
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def _write_corpus(path: Path, rows: list[VexTokenCorpusRow]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row_to_json(row), sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")


def _zip_vex_input(source_dir: Path, zip_path: Path) -> None:
    names = (
        "notebook_audit.json",
        "vex_recipe.json",
        "verified_rules.jsonl",
        "verified_rule_manifest.json",
        "corpus_direct_raw.jsonl",
        "corpus_family_tagged.jsonl",
        "corpus_short_rule_trace.jsonl",
        "corpus_mixed_main.jsonl",
        "corpus_manifest.json",
    )
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            path = source_dir / name
            if path.exists():
                archive.write(path, name)
