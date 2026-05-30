from __future__ import annotations

from pathlib import Path

from tools.hash_audit import check_manifest, write_manifest


def test_hash_audit_write_check_and_detect_change(tmp_path: Path) -> None:
    root = tmp_path
    sample = root / "sample.txt"
    sample.write_text("alpha", encoding="utf-8")
    manifest_path = root / "manifest.json"
    manifest = write_manifest(["sample.txt", "missing.txt"], manifest_path, root)
    assert manifest["files"][0]["exists"] is True
    assert manifest["files"][1]["exists"] is False
    ok, report = check_manifest(manifest_path, root)
    assert ok is False
    assert any(item["message"] == "missing" for item in report["failures"])
    sample.write_text("beta", encoding="utf-8")
    ok, report = check_manifest(manifest_path, root)
    assert ok is False
    assert any(item["message"] == "changed" for item in report["failures"])


def test_hash_audit_passes_unchanged_file(tmp_path: Path) -> None:
    root = tmp_path
    (root / "sample.txt").write_text("alpha", encoding="utf-8")
    manifest_path = root / "manifest.json"
    write_manifest(["sample.txt"], manifest_path, root)
    ok, report = check_manifest(manifest_path, root)
    assert ok is True
    assert report["failures"] == []
