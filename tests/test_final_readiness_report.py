from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.win_system_audit.final_readiness_report import build_final_readiness_report  # noqa: E402


def test_final_readiness_report_uses_audit_and_corpus(tmp_path: Path) -> None:
    audit = {"verdict": "MAIN_READY"}
    coverage = {"verified_correct_by_family": {"bit": 2}}
    corpus = {"file_counts": {"win_micro.jsonl": 64}}
    (tmp_path / "audit.json").write_text(json.dumps(audit), encoding="utf-8")
    (tmp_path / "coverage.json").write_text(json.dumps(coverage), encoding="utf-8")
    (tmp_path / "corpus.json").write_text(json.dumps(corpus), encoding="utf-8")
    report = build_final_readiness_report(tmp_path / "audit.json", tmp_path / "coverage.json", tmp_path / "corpus.json", tmp_path / "out.json")
    assert report["status"] == "MICRO_READY"
    assert report["solver_verified_correct_total"] == 2
