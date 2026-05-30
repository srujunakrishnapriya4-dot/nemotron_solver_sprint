from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.anti086_barrier.training_recipe_selector import select_training_recipe  # noqa: E402


def test_training_selector_prefers_vex_style_backend() -> None:
    decision = select_training_recipe({"vex_style_available": True, "vllm_available": True, "token_corpus_available": True})

    assert decision["recommended_mode"] == "vex_style_available"


def test_training_selector_refuses_kaggle_heavy_when_backend_inadequate() -> None:
    decision = select_training_recipe({"kaggle_gpu_available": True})

    assert decision["recommended_mode"] == "kaggle_micro_only"
    assert "main_training" in decision["forbidden_actions"]


def test_training_selector_no_viable_training() -> None:
    assert select_training_recipe({})["recommended_mode"] == "no_viable_training"
