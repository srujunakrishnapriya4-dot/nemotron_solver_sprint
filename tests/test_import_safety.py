from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.release.import_safety import (
    ImportSafetyConfig,
    ImportSafetyError,
    ImportSafetyFinding,
    ImportSafetyReport,
    audit_import_safety,
)


@contextmanager
def temp_source(name: str):
    root = Path(__file__).resolve().parent / ".tmp_pass10_import_safety" / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def write_py(root: Path, text: str) -> None:
    (root / "module.py").write_text(text, encoding="utf-8")


def scan(root: Path):
    return audit_import_safety(ImportSafetyConfig(repository_root=root, scan_roots=(".",)))


def test_clean_source_scan_passes() -> None:
    with temp_source("clean") as root:
        write_py(root, "VALUE = 'peft_config is just a string'\n")
        assert scan(root).passed


def test_actual_import_torch_fails() -> None:
    with temp_source("torch") as root:
        write_py(root, "import torch\n")
        report = scan(root)
        assert not report.passed
        assert any("import torch" in finding.pattern for finding in report.findings)


def test_from_peft_import_fails() -> None:
    with temp_source("peft") as root:
        write_py(root, "from peft import LoraConfig\n")
        report = scan(root)
        assert not report.passed
        assert any("from peft" in finding.pattern for finding in report.findings)


def test_string_peft_config_does_not_fail() -> None:
    with temp_source("peft_string") as root:
        write_py(root, "payload = {'peft_config': {'r': 8}}\n")
        assert scan(root).passed


def test_harmless_submission_zip_string_does_not_fail() -> None:
    with temp_source("submission_string") as root:
        write_py(root, "note = 'submission.zip appears in docs only'\n")
        assert scan(root).passed


def test_forbidden_call_subprocess_run_fails() -> None:
    with temp_source("subprocess") as root:
        write_py(root, "import pathlib\nsubprocess.run(['echo', 'x'])\n")
        report = scan(root)
        assert not report.passed
        assert any("subprocess.run" == finding.pattern for finding in report.findings)


def test_forbidden_call_eval_fails() -> None:
    with temp_source("eval") as root:
        write_py(root, "value = eval('1 + 1')\n")
        report = scan(root)
        assert not report.passed
        assert any("eval" == finding.pattern for finding in report.findings)


@pytest.mark.parametrize(
    ("name", "source", "pattern"),
    [
        ("popen", "subprocess.Popen(['echo', 'x'])\n", "subprocess.Popen"),
        ("os_system", "os.system('echo x')\n", "os.system"),
        ("exec", "exec('x = 1')\n", "exec"),
        ("requests", "requests.post('https://example.invalid')\n", "requests.post"),
        ("openai", "openai.chat.completions.create()\n", "openai."),
        ("kaggle", "kaggle.api.competition_submit()\n", "kaggle.api"),
        ("torch_save", "torch.save({}, 'x.pt')\n", "torch.save"),
        ("trainer_train", "trainer.train()\n", "trainer.train"),
        ("save_pretrained", "model.save_pretrained('out')\n", "model.save_pretrained"),
        ("push_to_hub", "push_to_hub('repo')\n", "push_to_hub"),
    ],
)
def test_forbidden_call_patterns_fail(name: str, source: str, pattern: str) -> None:
    with temp_source(name) as root:
        write_py(root, source)
        report = scan(root)
        assert not report.passed
        assert any(pattern == finding.pattern or finding.pattern.startswith(pattern) for finding in report.findings)


@pytest.mark.parametrize(
    ("name", "source"),
    [
        ("path_write_text", "from pathlib import Path\nPath('submission.zip').write_text('x')\n"),
        ("path_write_bytes", "from pathlib import Path\nPath('submission.zip').write_bytes(b'x')\n"),
        ("pathlib_write_text", "import pathlib\npathlib.Path('submission.zip').write_text('x')\n"),
        ("pathlib_write_bytes", "import pathlib\npathlib.Path('submission.zip').write_bytes(b'x')\n"),
        ("zipfile_w", "import zipfile\nzipfile.ZipFile('submission.zip', 'w')\n"),
        ("zipfile_a", "import zipfile\nzipfile.ZipFile('submission.zip', 'a')\n"),
        ("zipfile_x", "import zipfile\nzipfile.ZipFile('submission.zip', 'x')\n"),
    ],
)
def test_forbidden_submission_zip_write_patterns_fail(name: str, source: str) -> None:
    with temp_source(name) as root:
        write_py(root, source)
        report = scan(root)
        assert not report.passed
        assert any("submission.zip write" == finding.pattern for finding in report.findings)


def test_scanner_does_not_execute_source_code() -> None:
    with temp_source("no_execute") as root:
        write_py(root, "raise RuntimeError('would execute if imported')\n")
        report = scan(root)
        assert report.passed


def test_forged_import_safety_report_hash_rejected() -> None:
    with temp_source("forged") as root:
        write_py(root, "x = 1\n")
        report = scan(root)
        with pytest.raises(ImportSafetyError):
            replace(report, report_hash="forged")


def test_passed_true_with_errors_rejected() -> None:
    finding = ImportSafetyFinding(path="module.py", line_number=1, pattern="eval", severity="error", message="bad")
    with pytest.raises(ImportSafetyError):
        ImportSafetyReport(passed=True, scanned_files=("module.py",), findings=(finding,), errors=("module.py:1:eval",))


def test_report_hash_deterministic() -> None:
    with temp_source("deterministic") as root:
        write_py(root, "x = 1\n")
        assert scan(root).report_hash == scan(root).report_hash
