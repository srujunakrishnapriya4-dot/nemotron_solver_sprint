from __future__ import annotations

from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
CELL_DIR = ROOT / "artifacts" / "win_system_kaggle_cells"


def test_micro_stack_cells_exist_and_are_copy_safe() -> None:
    expected = [
        "CELL_01_write_configs.py",
        "CELL_02_write_runtime_patches.py",
        "CELL_03_write_prepare_tokens.py",
        "CELL_04_write_train.py",
        "CELL_05_write_eval.py",
        "CELL_06_write_parent_calibrated_eval.py",
        "CELL_07_write_orchestrator.py",
        "CELL_08_verify_files.py",
        "CELL_09_v1b_commands.py",
    ]
    for name in expected:
        path = CELL_DIR / name
        assert path.exists(), name
        assert path.read_text(encoding="utf-8").strip()


def test_cells_do_not_run_v2_v3_or_package() -> None:
    commands = (CELL_DIR / "CELL_09_v1b_commands.py").read_text(encoding="utf-8")
    forbidden = ["train_v2", "eval_v2", "train_v3", "eval_v3", "package_final", "submission.zip"]
    assert not any(token in commands for token in forbidden)
    assert "kaggle_build_parent_calibrated_eval.py --config anti086_winmode_v1b.yaml" in commands
    assert "kaggle_train_stage.py --config anti086_winmode_v1b.yaml" in commands
    assert "kaggle_eval_stage.py --config anti086_winmode_v1b.yaml --stage eval_v1b" in commands


def test_micro_cells_have_no_placeholder_eval_or_fake_pass() -> None:
    blob = "\n".join(path.read_text(encoding="utf-8") for path in CELL_DIR.glob("CELL_*.py"))
    assert '"status": "EVAL_SCRIPT_PLACEHOLDER"' not in blob
    assert "'status': 'EVAL_SCRIPT_PLACEHOLDER'" not in blob
    assert "fake PASS" not in blob
    assert "NO V2. NO PACKAGE. NO SUBMISSION" in blob


def test_micro_cells_zip_contains_cells() -> None:
    zip_path = ROOT / "artifacts" / "win_system_kaggle_cells.zip"
    assert zip_path.exists()
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
    assert "CELL_09_v1b_commands.py" in names
