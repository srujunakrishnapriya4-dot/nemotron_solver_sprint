from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.vex_progress_bridge.notebook_audit import audit_vex_notebooks  # noqa: E402
from nemotron_engine.vex_progress_bridge.vex_recipe import VexRecipeError, VexTrainingRecipe, recipes_from_audit  # noqa: E402


def test_vex_recipe_builds_from_real_notebook_audit(tmp_path: Path) -> None:
    audit = audit_vex_notebooks(
        Path(r"C:\Users\ADMIN\OneDrive\Desktop\vex506-reinst-kaggle-replica.ipynb"),
        Path(r"C:\Users\ADMIN\OneDrive\Desktop\vex-eval-vex506-reinst.ipynb"),
        tmp_path / "audit.json",
    )
    train_recipe, eval_recipe = recipes_from_audit(audit)

    assert train_recipe.lora_rank == 32
    assert train_recipe.reset_weights is True
    assert "lm_head" in train_recipe.target_modules
    assert train_recipe.heldout_file
    assert eval_recipe.max_new_tokens == 7680
    assert abs(sum(eval_recipe.category_weights.values()) - 1.0) < 1e-12


def test_vex_recipe_rejects_lm_head_without_audit_extraction() -> None:
    with pytest.raises(VexRecipeError):
        VexTrainingRecipe(
            lora_rank=32,
            lora_alpha=32,
            lora_dropout=0.0,
            max_seq_len=8192,
            num_steps=1,
            batch_size=1,
            micro_batch_size=1,
            learning_rate=2e-4,
            reset_weights=True,
            in_proj_only=False,
            moe_tie_weights=False,
            shuffle_dataset=True,
            target_modules=("lm_head",),
            corpus_path="/x",
            train_order_path="/x",
            swap_cryptarithm_thk_path="/x",
            swap_v6_path="/x",
            swap_binary_fmt_path="/x",
            extra_hyprevise_synth_path="/x",
            heldout_file="/x",
            source_notebook_hash="h",
            lm_head_extracted=False,
        )


def test_vex_recipe_blocks_replica_without_notebook_audit() -> None:
    with pytest.raises(VexRecipeError):
        VexTrainingRecipe(
            lora_rank=32,
            lora_alpha=32,
            lora_dropout=0.0,
            max_seq_len=8192,
            num_steps=1,
            batch_size=1,
            micro_batch_size=1,
            learning_rate=2e-4,
            reset_weights=True,
            in_proj_only=False,
            moe_tie_weights=False,
            shuffle_dataset=True,
            target_modules=("q_proj",),
            corpus_path="/x",
            train_order_path="/x",
            swap_cryptarithm_thk_path="/x",
            swap_v6_path="/x",
            swap_binary_fmt_path="/x",
            extra_hyprevise_synth_path="/x",
            heldout_file="/x",
            source_notebook_hash="h",
            notebook_audit_found=False,
        )
