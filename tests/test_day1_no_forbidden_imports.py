from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_active_kaggle_files_do_not_import_nemotron_engine() -> None:
    hits = []
    for base in (ROOT / "kaggle_anti086", ROOT / "artifacts" / "win_system_kaggle_cells"):
        for path in base.glob("kaggle_*.py") if base.name == "kaggle_anti086" else base.glob("CELL_*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "from nemotron_engine" in text or "import nemotron_engine" in text:
                hits.append(str(path.relative_to(ROOT)))
    assert hits == []
