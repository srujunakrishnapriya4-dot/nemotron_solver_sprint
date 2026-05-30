from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.win_system_audit.equation_quarantine_policy import build_equation_quarantine  # noqa: E402


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_equation_quarantine_exports_safe_direct_unsafe_and_offline(tmp_path: Path) -> None:
    _write(tmp_path / "solver_verified_correct.jsonl", [{"id": "v", "family": "equation_symbolic", "prediction": "12", "answer": "12", "reason": "rule"}])
    _write(tmp_path / "solver_verified_wrong.jsonl", [{"id": "w", "family": "equation_symbolic", "answer": "9", "reason": "answer_mismatch"}])
    _write(tmp_path / "solver_abstained.jsonl", [{"id": "a", "family": "equation_symbolic", "answer": "7", "reason": "unsupported_equation_transform"}])
    _write(tmp_path / "digit_symbol_failure_examples.jsonl", [{"id": "a", "cluster": "digit_symbol_arithmetic", "enough_constraints_for_unique_induction": True}])
    manifest = build_equation_quarantine(tmp_path, tmp_path, direct_answer_cap=1)
    assert manifest["equation_safe_trace_count"] == 1
    assert manifest["equation_direct_answer_only_count"] == 1
    assert manifest["equation_unsafe_excluded_count"] == 2
    assert manifest["equation_offline_synthesis_candidate_count"] == 1
