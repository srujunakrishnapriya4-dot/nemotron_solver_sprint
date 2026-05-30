from __future__ import annotations

import json
from pathlib import Path
import zipfile

from tools.day1_repo_guard import EXPECTED_CELLS, REQUIRED_NAMES, build_report


def _write_clean_tree(root: Path) -> None:
    kaggle = root / "kaggle_anti086"
    cells = root / "artifacts" / "win_system_kaggle_cells"
    day1 = root / "artifacts" / "day1"
    kaggle.mkdir(parents=True)
    cells.mkdir(parents=True)
    day1.mkdir(parents=True)
    for name in REQUIRED_NAMES:
        path = kaggle / name
        if name.endswith(".yaml"):
            path.write_text("stage: v1b\nparent_adapter_path: none\n", encoding="utf-8")
        else:
            path.write_text("print('ok')\n", encoding="utf-8")
    cell_text = (
        "from pathlib import Path\nimport hashlib\nimport json\n"
        "def write_file_checked(path, content):\n"
        "    path = Path(path)\n"
        "    path.write_text(content, encoding='utf-8')\n"
        "    print(json.dumps({'path': str(path), 'size_bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}))\n"
        "write_file_checked('x.py', 'ok')\n"
    )
    for name in EXPECTED_CELLS:
        (cells / name).write_text(cell_text if name != "CELL_08_verify_files.py" else "print('path-safety compile no-import sha256')\n", encoding="utf-8")
    with zipfile.ZipFile(root / "artifacts" / "win_system_kaggle_cells.zip", "w") as archive:
        for name in EXPECTED_CELLS:
            archive.write(cells / name, arcname=name)
    (day1 / "day1_hash_manifest.json").write_text(json.dumps({"schema_version": 1, "files": []}), encoding="utf-8")


def test_day1_repo_guard_passes_clean_toy_tree(tmp_path: Path) -> None:
    _write_clean_tree(tmp_path)
    report = build_report(tmp_path, strict=True)
    assert report["status"] == "PASS", report["failures"]


def test_day1_repo_guard_fails_for_forbidden_import(tmp_path: Path) -> None:
    _write_clean_tree(tmp_path)
    (tmp_path / "kaggle_anti086" / "kaggle_prepare_anti086_tokens.py").write_text("from nemotron_engine.core import x\n", encoding="utf-8")
    report = build_report(tmp_path, strict=True)
    assert report["status"] == "FAIL"
    assert any(f["check"] == "forbidden_imports" for f in report["failures"])


def test_day1_repo_guard_fails_for_missing_required_in_strict_mode(tmp_path: Path) -> None:
    _write_clean_tree(tmp_path)
    (tmp_path / "kaggle_anti086" / "kaggle_backend_probe.py").unlink()
    report = build_report(tmp_path, strict=True)
    assert any(f["check"] == "required_files" for f in report["failures"])


def test_day1_repo_guard_catches_indirect_input_overlay_bug(tmp_path: Path) -> None:
    _write_clean_tree(tmp_path)
    bad = tmp_path / "kaggle_anti086" / "kaggle_prepare_anti086_tokens.py"
    bad.write_text(
        "from pathlib import Path\n"
        "input_base = Path('/kaggle/input')\n"
        "for path in input_base.rglob('curriculum_manifest.json'):\n"
        "    _write_conservative_quarantine_overlay(path.parent)\n",
        encoding="utf-8",
    )
    report = build_report(tmp_path, strict=True)
    assert any(f["check"] == "indirect_readonly_input_write_risk" for f in report["failures"])


def test_day1_repo_guard_allows_fixed_input_overlay_pattern(tmp_path: Path) -> None:
    _write_clean_tree(tmp_path)
    fixed = tmp_path / "kaggle_anti086" / "kaggle_prepare_anti086_tokens.py"
    fixed.write_text(
        "from pathlib import Path\n"
        "input_base = Path('/kaggle/input')\n"
        "overlay = Path('/kaggle/working/anti086_input_overlay')\n"
        "for path in input_base.rglob('curriculum_manifest.json'):\n"
        "    copy_minimal_input_to_overlay(path.parent, overlay)\n"
        "    _write_conservative_quarantine_overlay(overlay)\n",
        encoding="utf-8",
    )
    report = build_report(tmp_path, strict=True)
    assert report["status"] == "PASS", report["failures"]


def test_day1_repo_guard_rejects_unchecked_writer_cell(tmp_path: Path) -> None:
    _write_clean_tree(tmp_path)
    cell = tmp_path / "artifacts" / "win_system_kaggle_cells" / "CELL_01_write_configs.py"
    cell.write_text("from pathlib import Path\nPath('x.py').write_text('bad')\nprint('sha256 size bytes')\n", encoding="utf-8")
    report = build_report(tmp_path, strict=True)
    assert any(f["check"] == "generated_hash_reporting" for f in report["failures"])


def test_day1_repo_guard_rejects_bad_output_config(tmp_path: Path) -> None:
    _write_clean_tree(tmp_path)
    (tmp_path / "kaggle_anti086" / "anti086_winmode_v1b.yaml").write_text(
        "stage: v1b\nparent_adapter_path: none\ntoken_output: /kaggle/input/bad/train_tokens.jsonl\n",
        encoding="utf-8",
    )
    report = build_report(tmp_path, strict=True)
    assert any(f["check"] == "output_path_safety_static" for f in report["failures"])


def test_day1_repo_guard_accepts_checked_writer_cell(tmp_path: Path) -> None:
    _write_clean_tree(tmp_path)
    report = build_report(tmp_path, strict=True)
    assert report["status"] == "PASS", report["failures"]
