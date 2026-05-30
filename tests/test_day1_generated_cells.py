from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import zipfile

from tools.day1_repo_guard import EXPECTED_CELLS


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_generated_cell_directory_and_zip_match() -> None:
    cell_dir = ROOT / "artifacts" / "win_system_kaggle_cells"
    zip_path = ROOT / "artifacts" / "win_system_kaggle_cells.zip"
    assert cell_dir.exists()
    assert zip_path.exists()
    disk = {name: _sha(cell_dir / name) for name in EXPECTED_CELLS}
    for name in EXPECTED_CELLS:
        assert (cell_dir / name).exists(), name
    with zipfile.ZipFile(zip_path) as archive:
        zipped = {
            Path(info.filename).name: hashlib.sha256(archive.read(info.filename)).hexdigest()
            for info in archive.infolist()
            if not info.is_dir()
        }
    assert zipped == disk


def test_writer_cells_use_checked_writes_only() -> None:
    cell_dir = ROOT / "artifacts" / "win_system_kaggle_cells"
    for path in cell_dir.glob("CELL_*.py"):
        text = path.read_text(encoding="utf-8")
        assert "%%writefile" not in text
        if "write_file_checked(" not in text:
            continue
        assert "size_bytes" in text and "sha256" in text and "path" in text
        tree = ast.parse(text)
        helper_ranges = [
            (node.lineno, getattr(node, "end_lineno", node.lineno))
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "write_file_checked"
        ]
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "write_text":
                line = getattr(node, "lineno", 0)
                assert any(start <= line <= end for start, end in helper_ranges), f"unchecked write_text in {path}:{line}"


def test_cell_08_contains_compile_import_and_path_safety_checks() -> None:
    text = (ROOT / "artifacts" / "win_system_kaggle_cells" / "CELL_08_verify_files.py").read_text(encoding="utf-8")
    assert "py_compile.compile" in text
    assert "nemotron_engine" in text
    assert "/kaggle/input" in text
    assert "output_fields" in text
    assert "sha256" in text


def test_runbook_cell_list_matches_actual_cells() -> None:
    runbook = (ROOT / "factory_reset_runbook.md").read_text(encoding="utf-8")
    listed = re.findall(r"`(CELL_\d+_[^`]+\.py)`", runbook)
    assert listed == EXPECTED_CELLS


def test_generated_cells_manifest_exists_and_matches() -> None:
    manifest_path = ROOT / "artifacts" / "day1" / "day1_generated_cells_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["zip_matches_directory"] is True
    assert [cell["name"] for cell in manifest["cells"]] == EXPECTED_CELLS
