from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.anti086_barrier.win_corpus_builder import build_win_corpora  # noqa: E402


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_win_corpus_builder_caps_synthetic(tmp_path: Path) -> None:
    base = [{"id": f"b{i}", "family": "bit_manipulation", "corpus_type": "direct_raw", "text": "User:\np\nAssistant:\n1", "answer": "1"} for i in range(80)]
    syn = [{"generation_hash": f"g{i}", "family": "equation_symbolic", "rule_id": "r", "prompt": "p", "answer": "2"} for i in range(40)]
    _write(tmp_path / "base.jsonl", base)
    _write(tmp_path / "syn.jsonl", syn)
    _write(tmp_path / "contrast.jsonl", [])
    manifest = build_win_corpora(tmp_path / "base.jsonl", tmp_path / "syn.jsonl", tmp_path / "contrast.jsonl", tmp_path)
    assert manifest["file_counts"]["win_micro.jsonl"] <= 64
    assert manifest["synthetic_ratio_v2"] <= 0.25


def test_win_corpus_builder_uses_quarantine_policy(tmp_path: Path) -> None:
    base = [{"id": f"b{i}", "family": "bit_manipulation", "corpus_type": "direct_raw", "text": "User:\np\nAssistant:\n1", "answer": "1"} for i in range(20)]
    safe = [{"id": "eq1", "prediction": "42", "answer": "42", "rule_id": "verified_rule"}]
    direct = [{"id": "eq_bad", "answer": "99"}]
    unsafe = [{"id": "eq_bad", "reason": "unsupported_equation_transform"}]
    _write(tmp_path / "base.jsonl", base)
    _write(tmp_path / "syn.jsonl", [])
    _write(tmp_path / "contrast.jsonl", [])
    _write(tmp_path / "equation_safe_verified_trace.jsonl", safe)
    _write(tmp_path / "equation_direct_answer_only.jsonl", direct)
    _write(tmp_path / "equation_unsafe_trace_excluded.jsonl", unsafe)
    (tmp_path / "equation_quarantine_manifest.json").write_text(json.dumps({"equation_safe_trace_count": 1, "equation_direct_answer_only_count": 1, "equation_unsafe_excluded_count": 1, "equation_offline_synthesis_candidate_count": 0, "equation_status": "QUARANTINED"}), encoding="utf-8")

    manifest = build_win_corpora(tmp_path / "base.jsonl", tmp_path / "syn.jsonl", tmp_path / "contrast.jsonl", tmp_path, equation_quarantine_dir=tmp_path)

    assert manifest["equation_safe_trace_count"] == 1
    assert manifest["equation_unsafe_excluded_count"] == 1
    assert manifest["equation_status"] == "QUARANTINED"
    traces = (tmp_path / "win_v1.jsonl").read_text(encoding="utf-8")
    assert "verified_rule" in traces


def test_win_corpus_builder_counts_truth_table_bit_traces(tmp_path: Path) -> None:
    base = [{"id": "b0", "family": "bit_manipulation", "corpus_type": "direct_raw", "text": "User:\np\nAssistant:\n1", "answer": "1"}]
    bit = [{"id": "bit1", "family": "bit_manipulation", "prediction": "10101010", "diagnostics": {"expr_id": "truth_table_bit_mining"}}]
    _write(tmp_path / "base.jsonl", base)
    _write(tmp_path / "syn.jsonl", [])
    _write(tmp_path / "contrast.jsonl", [])
    _write(tmp_path / "bits.jsonl", bit)
    (tmp_path / "equation_quarantine_manifest.json").write_text(json.dumps({"equation_status": "OFFLINE_SYNTHESIS_REQUIRED"}), encoding="utf-8")
    manifest = build_win_corpora(tmp_path / "base.jsonl", tmp_path / "syn.jsonl", tmp_path / "contrast.jsonl", tmp_path, equation_quarantine_dir=tmp_path, bit_verified_path=tmp_path / "bits.jsonl")
    assert manifest["bit_truth_table_trace_count"] == 1
    assert manifest["equation_status"] == "OFFLINE_SYNTHESIS_REQUIRED"
