from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.vex_progress_bridge.vex_training_manifest import VexTrainingManifest, VexTrainingManifestError  # noqa: E402


def test_training_manifest_rejects_full_prompt_loss() -> None:
    with pytest.raises(VexTrainingManifestError):
        VexTrainingManifest("t", "c", 1, 1, ("q_proj",), "parent", full_prompt_loss=True)


def test_training_manifest_accepts_masked_manifest() -> None:
    assert VexTrainingManifest("t", "c", 1, 10, ("q_proj",), "parent").manifest_hash

