from pathlib import Path

from kaggle_anti086.data.v2_corpus_io import write_jsonl_checked
from kaggle_anti086.data.v2_corpus_manifest import build_corpus_manifest, validate_corpus_manifest


def test_manifest_contains_hashes_and_row_counts(tmp_path):
    direct_path = tmp_path / "direct.jsonl"
    write_jsonl_checked(direct_path, [{"family": "format_only", "rule_id": "r", "leakage_group": "lg", "source_eval": "unit", "loss_scope": "assistant_only"}], field_name="direct")
    manifest = build_corpus_manifest(
        files={"verified_direct_answer": direct_path},
        direct_rows=[{"family": "format_only", "rule_id": "r", "leakage_group": "lg", "source_eval": "unit", "loss_scope": "assistant_only"}],
        solver_corrected_rows=[],
        abstain_rows=[],
        hard_negative_rows=[],
        leakage_report={"status": "PASS"},
        mixture_report={"status": "PASS"},
        quality_gate={"decision": "ALLOW_DAY7_TRAINING_CONFIG_PREP", "quality_gates": {}, "remaining_blockers": []},
    )
    record = manifest["files"]["verified_direct_answer"]
    assert record["row_count"] == 1
    assert len(record["sha256"]) == 64
    assert manifest["training_allowed"] is False
    assert manifest["no_training_performed"] is True
    assert validate_corpus_manifest(manifest)["status"] == "PASS"


def test_manifest_blocks_missing_required_corpus(tmp_path):
    manifest = build_corpus_manifest(
        files={"verified_direct_answer": tmp_path / "missing.jsonl"},
        direct_rows=[],
        solver_corrected_rows=[],
        abstain_rows=[],
        hard_negative_rows=[],
        leakage_report={"status": "PASS"},
        mixture_report={"status": "PASS"},
        quality_gate={"decision": "BLOCK_DAY7_TRAINING_CONFIG_PREP", "quality_gates": {}, "remaining_blockers": []},
    )
    assert validate_corpus_manifest(manifest)["status"] == "FAIL"
