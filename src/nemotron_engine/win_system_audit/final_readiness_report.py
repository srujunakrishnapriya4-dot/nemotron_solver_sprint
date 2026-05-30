from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .repo_readiness_audit import run_repo_readiness_audit


def build_final_readiness_report(
    audit_path: str | Path = "artifacts/win_system/readiness_audit.json",
    coverage_path: str | Path = "artifacts/win_system/solver_coverage_report.json",
    corpus_manifest_path: str | Path = "artifacts/win_system/win_corpus_manifest.json",
    output_path: str | Path = "artifacts/win_system/final_readiness_report.json",
) -> dict[str, Any]:
    audit = json.loads(Path(audit_path).read_text(encoding="utf-8")) if Path(audit_path).exists() else run_repo_readiness_audit()
    coverage = json.loads(Path(coverage_path).read_text(encoding="utf-8")) if Path(coverage_path).exists() else {}
    corpus = json.loads(Path(corpus_manifest_path).read_text(encoding="utf-8")) if Path(corpus_manifest_path).exists() else {}
    status = audit["verdict"]
    if status == "MAIN_READY" and corpus.get("file_counts", {}).get("win_micro.jsonl", 0) > 0:
        status = "MICRO_READY"
    payload = {
        "status": status,
        "audit_verdict": audit["verdict"],
        "solver_verified_correct_total": sum(coverage.get("verified_correct_by_family", {}).values()),
        "corpus_counts": corpus.get("file_counts", {}),
        "next_action": "run Kaggle Day-2 validate/prepare_micro only" if status in {"MICRO_READY", "MAIN_READY"} else "fix critical gaps before GPU",
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    return payload
