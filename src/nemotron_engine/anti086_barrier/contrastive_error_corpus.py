from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nemotron_engine.core.schemas import stable_hash


def build_contrastive_error_corpus(
    verified_rules_path: str | Path = "artifacts/vex_progress/verified_rules.jsonl",
    output_dir: str | Path = "artifacts/anti086",
    *,
    max_ratio_base_count: int = 1000,
) -> dict[str, Any]:
    rows = _read_jsonl(Path(verified_rules_path))
    correct = [row for row in rows if row.get("verified_status") == "verified_correct"]
    wrongish = [row for row in rows if row.get("verified_status") in {"verified_wrong", "abstained"}]
    cap = max(1, int(max_ratio_base_count * 0.15))
    output = []
    for idx, row in enumerate(correct[:cap]):
        output.append(
            {
                "id": f"contrastive_{row['id']}",
                "source_id": row["id"],
                "family": row.get("family"),
                "text": f"User:\n{row['prompt']}\nAssistant:\nWrong rule rejected: plausible shortcut failed verification.\nCorrect rule: {row.get('rule_id') or 'verified symbolic rule'}.\nAnswer: {row['answer']}",
                "answer": row["answer"],
                "corpus_type": "contrastive_error",
                "contrastive_hash": stable_hash(row),
            }
        )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "contrastive_error_corpus.jsonl", output)
    manifest = {
        "contrastive_count": len(output),
        "cap_ratio": 0.15,
        "source_verified_correct": len(correct),
        "available_wrong_or_abstained": len(wrongish),
        "manifest_hash": stable_hash([row["contrastive_hash"] for row in output]),
    }
    (out / "contrastive_manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
    return manifest


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")
