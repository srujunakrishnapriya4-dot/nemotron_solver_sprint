from __future__ import annotations

from pathlib import Path

from tools.day1_repo_guard import readonly_write_failures


ROOT = Path(__file__).resolve().parents[1]


def test_readonly_input_write_detector_flags_write_text(tmp_path: Path) -> None:
    path = tmp_path / "kaggle_bad.py"
    path.write_text("from pathlib import Path\nPath('/kaggle/input/x').write_text('bad')\n", encoding="utf-8")
    failures = readonly_write_failures(path, tmp_path)
    assert failures


def test_active_kaggle_files_do_not_write_to_kaggle_input() -> None:
    failures = []
    for path in (ROOT / "kaggle_anti086").glob("kaggle_*.py"):
        failures.extend(readonly_write_failures(path, ROOT))
    for path in (ROOT / "artifacts" / "win_system_kaggle_cells").glob("CELL_*.py"):
        failures.extend(readonly_write_failures(path, ROOT))
    assert failures == []
