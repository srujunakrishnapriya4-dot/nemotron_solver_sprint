from __future__ import annotations

import json
from pathlib import Path

from tools.archive_old_notebook_fragments import archive


ROOT = Path(__file__).resolve().parents[1]


def test_baseline_manifest_has_required_fields_and_excludes_weights() -> None:
    manifest_path = ROOT / "clean_sprint10_1_baseline" / "baseline_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["created_by"] == "SPRINT-11_DAY1_FREEZE"
    assert "training" in manifest["forbidden_actions"]
    for record in manifest["files"]:
        assert {"path", "sha256", "size_bytes", "role"} <= set(record)
        assert not record["path"].endswith((".safetensors", ".bin", ".pt", ".pth", ".gguf", "submission.zip"))


def test_archive_old_fragments_dry_run_and_apply(tmp_path: Path) -> None:
    stale = tmp_path / "artifacts" / "old" / "CELL_00_old.py"
    active = tmp_path / "artifacts" / "win_system_kaggle_cells" / "CELL_01_write_configs.py"
    stale.parent.mkdir(parents=True)
    active.parent.mkdir(parents=True)
    stale.write_text("SPRINT-9 permits micro or v1 only\n", encoding="utf-8")
    active.write_text("SPRINT-9 permits micro or v1 only\n", encoding="utf-8")
    dry = archive(tmp_path, apply=False)
    assert len(dry["records"]) == 1
    assert stale.exists()
    applied = archive(tmp_path, apply=True)
    assert len(applied["records"]) == 1
    assert not stale.exists()
    assert active.exists()
