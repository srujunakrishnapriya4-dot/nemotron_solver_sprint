from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.release.locked_pass_registry import (
    LockedPassRecord,
    LockedPassRegistry,
    LockedPassRegistryError,
    build_locked_pass_registry,
    validate_locked_pass_registry,
)


@contextmanager
def temp_root(name: str):
    root = Path(__file__).resolve().parent / ".tmp_pass10_registry" / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_builds_registry_with_all_locked_pass_records() -> None:
    registry = build_locked_pass_registry(repository_root=Path(__file__).resolve().parents[1], suite_result_summary="494 passed, 1 skipped")
    assert len(registry.records) == 10
    assert all(record.status == "LOCKED" for record in registry.records)
    assert validate_locked_pass_registry(registry) is registry


def test_registry_rejects_missing_required_locked_pass() -> None:
    registry = build_locked_pass_registry(repository_root=Path(__file__).resolve().parents[1])
    with pytest.raises(LockedPassRegistryError):
        LockedPassRegistry(records=registry.records[:-1], suite_result_summary=None, scoped_suite_command="pytest tests -q")


def test_registry_rejects_duplicate_pass_id() -> None:
    registry = build_locked_pass_registry(repository_root=Path(__file__).resolve().parents[1])
    with pytest.raises(LockedPassRegistryError):
        LockedPassRegistry(records=registry.records + (registry.records[0],), suite_result_summary=None, scoped_suite_command="pytest tests -q")


def test_record_rejects_unlocked_status() -> None:
    with pytest.raises(LockedPassRegistryError):
        LockedPassRecord(
            pass_id="pass_x",
            name="bad",
            status="OPEN",
            required_modules=("nemotron_engine.core.schemas",),
            required_tests=("tests/test_schemas.py",),
            representative_imports=("nemotron_engine.core.schemas:stable_hash",),
        )


def test_registry_rejects_missing_required_module() -> None:
    registry = build_locked_pass_registry(repository_root=Path(__file__).resolve().parents[1])
    bad_record = replace(registry.records[0], required_modules=("nemotron_engine.nope",), record_hash="")
    bad_registry = LockedPassRegistry(records=(bad_record,) + registry.records[1:], suite_result_summary=None, scoped_suite_command="pytest tests -q")
    with pytest.raises(LockedPassRegistryError):
        validate_locked_pass_registry(bad_registry)


def test_registry_rejects_missing_required_test() -> None:
    registry = build_locked_pass_registry(repository_root=Path(__file__).resolve().parents[1])
    bad_record = replace(registry.records[0], required_tests=("tests/nope.py",), record_hash="")
    bad_registry = LockedPassRegistry(records=(bad_record,) + registry.records[1:], suite_result_summary=None, scoped_suite_command="pytest tests -q")
    with pytest.raises(LockedPassRegistryError):
        validate_locked_pass_registry(bad_registry)


def test_forged_record_hash_rejected() -> None:
    registry = build_locked_pass_registry(repository_root=Path(__file__).resolve().parents[1])
    with pytest.raises(LockedPassRegistryError):
        replace(registry.records[0], record_hash="forged")


def test_forged_registry_hash_rejected() -> None:
    registry = build_locked_pass_registry(repository_root=Path(__file__).resolve().parents[1])
    with pytest.raises(LockedPassRegistryError):
        replace(registry, registry_hash="forged")


def test_registry_detects_wrong_wrapper_root() -> None:
    with temp_root("wrong_root") as root:
        registry = build_locked_pass_registry(repository_root=root)
        with pytest.raises(LockedPassRegistryError):
            validate_locked_pass_registry(registry)


def test_registry_rejects_fake_root_with_src_tests_but_no_locked_modules() -> None:
    with temp_root("fake_root") as root:
        (root / "src").mkdir()
        (root / "tests").mkdir()
        registry = build_locked_pass_registry(repository_root=root)
        with pytest.raises(LockedPassRegistryError, match="module_origin_outside_repo"):
            validate_locked_pass_registry(registry)


def test_registry_rejects_required_module_resolved_outside_supplied_src() -> None:
    with temp_root("module_outside") as root:
        (root / "src").mkdir()
        (root / "tests").mkdir()
        registry = build_locked_pass_registry(repository_root=root)
        bad_record = replace(registry.records[0], required_modules=("nemotron_engine.core.schemas",), required_tests=(), representative_imports=(), record_hash="")
        bad_registry = LockedPassRegistry(records=(bad_record,) + registry.records[1:], suite_result_summary=None, scoped_suite_command="pytest tests -q", repository_root=str(root))
        with pytest.raises(LockedPassRegistryError, match="module_origin_outside_repo"):
            validate_locked_pass_registry(bad_registry)


def test_registry_rejects_representative_import_resolved_outside_supplied_src() -> None:
    with temp_root("representative_outside") as root:
        (root / "src").mkdir()
        (root / "tests").mkdir()
        registry = build_locked_pass_registry(repository_root=root)
        bad_record = replace(
            registry.records[0],
            required_modules=(),
            required_tests=(),
            representative_imports=("nemotron_engine.core.schemas:stable_hash",),
            record_hash="",
        )
        bad_registry = LockedPassRegistry(records=(bad_record,) + registry.records[1:], suite_result_summary=None, scoped_suite_command="pytest tests -q", repository_root=str(root))
        with pytest.raises(LockedPassRegistryError, match="representative_import_origin_outside_repo"):
            validate_locked_pass_registry(bad_registry)


def test_registry_hash_deterministic() -> None:
    first = build_locked_pass_registry(repository_root=Path(__file__).resolve().parents[1])
    second = build_locked_pass_registry(repository_root=Path(__file__).resolve().parents[1])
    assert first.registry_hash == second.registry_hash
