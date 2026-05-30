from __future__ import annotations

from pathlib import Path
import sys

from tools.day1_repo_guard import readonly_write_failures


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle_anti086"))

from kaggle_prepare_anti086_tokens import resolve_anti086_input_root  # noqa: E402


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


def test_resolver_does_not_write_quarantine_manifest_under_input_like_source(tmp_path: Path) -> None:
    source = tmp_path / "kaggle" / "input" / "datasets" / "surtr19" / "anti086-kaggle-input"
    working = tmp_path / "kaggle" / "working"
    source.mkdir(parents=True)
    working.mkdir(parents=True)
    (source / "corpus_anti086_v1.jsonl").write_text('{"id":"r1","text":"User:\\nQ\\nAssistant:\\nA","answer":"A","family":"cipher_text"}\n', encoding="utf-8")
    (source / "curriculum_manifest.json").write_text("{}", encoding="utf-8")

    resolved = resolve_anti086_input_root({}, input_base=tmp_path / "kaggle" / "input", working_base=working)

    assert resolved == working / "anti086_input_overlay"
    assert not (source / "equation_quarantine_manifest.json").exists()
    assert (resolved / "equation_quarantine_manifest.json").exists()
    assert (resolved / "corpus_anti086_v1.jsonl").exists()
