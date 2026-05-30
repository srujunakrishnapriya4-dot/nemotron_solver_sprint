from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle_anti086"))

import kaggle_winmode_orchestrator as orch  # noqa: E402


def test_orchestrator_allows_v1_but_not_v2_v3_package() -> None:
    assert "prepare_v1" in orch.STAGES
    assert "train_v1" in orch.STAGES
    assert "eval_v1" in orch.STAGES
    assert "prepare_v2" not in orch.STAGES
    assert "train_v3" not in orch.STAGES
    assert "package_final" not in orch.STAGES


def test_orchestrator_v1_requires_prepare_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(orch, "gate_dir", lambda: tmp_path)
    with pytest.raises(SystemExit):
        orch.require_previous("train_v1")
