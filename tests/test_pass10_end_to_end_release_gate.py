from __future__ import annotations

from contextlib import contextmanager
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.release.artifact_safety import ArtifactSafetyConfig, scan_release_artifacts
from nemotron_engine.release.import_safety import ImportSafetyConfig, audit_import_safety
from nemotron_engine.release.locked_pass_registry import build_locked_pass_registry, validate_locked_pass_registry
from nemotron_engine.release.release_candidate import ReleaseCandidateConfig, ReleaseCandidateError, build_release_candidate_report
from nemotron_engine.release.system_audit import SystemAuditConfig, run_system_audit


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = "494 passed, 1 skipped"


@contextmanager
def temp_root(name: str):
    root = Path(__file__).resolve().parent / ".tmp_pass10_end_to_end" / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_end_to_end_release_gate_deterministic() -> None:
    with temp_root("green") as root:
        source = root / "src/nemotron_engine"
        source.mkdir(parents=True)
        (source / "safe.py").write_text("VALUE = 'peft_config is metadata only'\n", encoding="utf-8")
        release_tree = root / "release_tree"
        release_tree.mkdir()
        (release_tree / "README.txt").write_text("clean\n", encoding="utf-8")
        registry = build_locked_pass_registry(repository_root=ROOT, suite_result_summary=SUMMARY)
        validate_locked_pass_registry(registry)
        import_report = audit_import_safety(ImportSafetyConfig(repository_root=root, scan_roots=("src/nemotron_engine",)))
        artifact_report = scan_release_artifacts(ArtifactSafetyConfig(root_path=release_tree))
        audit = run_system_audit(
            locked_pass_registry=registry,
            import_safety_report=import_report,
            artifact_safety_report=artifact_report,
            scoped_suite_result_summary=SUMMARY,
            config=SystemAuditConfig(accepted_suite_summaries=(SUMMARY,)),
        )
        candidate = build_release_candidate_report(
            ReleaseCandidateConfig(system_audit_report=audit, repository_root=str(ROOT), scoped_suite_result_summary=SUMMARY, accepted_suite_summaries=(SUMMARY,))
        )
        again = build_release_candidate_report(
            ReleaseCandidateConfig(system_audit_report=audit, repository_root=str(ROOT), scoped_suite_result_summary=SUMMARY, accepted_suite_summaries=(SUMMARY,))
        )
        assert import_report.passed
        assert artifact_report.passed
        assert audit.passed
        assert candidate.ready
        assert candidate.report_hash == again.report_hash


def test_end_to_end_fake_kaggle_leaderboard_metadata_blocks() -> None:
    with temp_root("fake_metadata") as root:
        source = root / "src/nemotron_engine"
        source.mkdir(parents=True)
        (source / "safe.py").write_text("x = 1\n", encoding="utf-8")
        release_tree = root / "release_tree"
        release_tree.mkdir()
        registry = build_locked_pass_registry(repository_root=ROOT, suite_result_summary=SUMMARY)
        audit = run_system_audit(
            locked_pass_registry=registry,
            import_safety_report=audit_import_safety(ImportSafetyConfig(repository_root=root, scan_roots=("src/nemotron_engine",))),
            artifact_safety_report=scan_release_artifacts(ArtifactSafetyConfig(root_path=release_tree)),
            scoped_suite_result_summary=SUMMARY,
            config=SystemAuditConfig(accepted_suite_summaries=(SUMMARY,)),
        )
        with pytest.raises(ReleaseCandidateError):
            ReleaseCandidateConfig(system_audit_report=audit, metadata={"leaderboard_readiness": True})


def test_end_to_end_trained_adapter_claim_blocks_without_evidence() -> None:
    with temp_root("adapter_claim") as root:
        source = root / "src/nemotron_engine"
        source.mkdir(parents=True)
        (source / "safe.py").write_text("x = 1\n", encoding="utf-8")
        release_tree = root / "release_tree"
        release_tree.mkdir()
        registry = build_locked_pass_registry(repository_root=ROOT, suite_result_summary=SUMMARY)
        audit = run_system_audit(
            locked_pass_registry=registry,
            import_safety_report=audit_import_safety(ImportSafetyConfig(repository_root=root, scan_roots=("src/nemotron_engine",))),
            artifact_safety_report=scan_release_artifacts(ArtifactSafetyConfig(root_path=release_tree)),
            scoped_suite_result_summary=SUMMARY,
            config=SystemAuditConfig(accepted_suite_summaries=(SUMMARY,)),
        )
        with pytest.raises(ReleaseCandidateError):
            ReleaseCandidateConfig(system_audit_report=audit, metadata={"note": "trained adapter available"})


def test_end_to_end_forbidden_artifact_blocks() -> None:
    with temp_root("forbidden_artifact") as root:
        source = root / "src/nemotron_engine"
        source.mkdir(parents=True)
        (source / "safe.py").write_text("x = 1\n", encoding="utf-8")
        release_tree = root / "release_tree"
        release_tree.mkdir()
        (release_tree / "model.safetensors").write_text("bad\n", encoding="utf-8")
        registry = build_locked_pass_registry(repository_root=ROOT, suite_result_summary=SUMMARY)
        audit = run_system_audit(
            locked_pass_registry=registry,
            import_safety_report=audit_import_safety(ImportSafetyConfig(repository_root=root, scan_roots=("src/nemotron_engine",))),
            artifact_safety_report=scan_release_artifacts(ArtifactSafetyConfig(root_path=release_tree)),
            scoped_suite_result_summary=SUMMARY,
            config=SystemAuditConfig(accepted_suite_summaries=(SUMMARY,)),
        )
        candidate = build_release_candidate_report(
            ReleaseCandidateConfig(system_audit_report=audit, repository_root=str(ROOT), scoped_suite_result_summary=SUMMARY, accepted_suite_summaries=(SUMMARY,))
        )
        assert not audit.passed
        assert not candidate.ready
