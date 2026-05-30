from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.vex_progress_bridge.notebook_audit import audit_vex_notebooks  # noqa: E402
from nemotron_engine.vex_progress_bridge.token_swap_recipe import (  # noqa: E402
    TokenSwapRecipeError,
    describe_merge_order,
    token_swap_recipe_from_audit,
    validate_token_swap_paths,
)


def test_token_swap_order_is_preserved_from_audit(tmp_path: Path) -> None:
    audit = audit_vex_notebooks(
        Path(r"C:\Users\ADMIN\OneDrive\Desktop\vex506-reinst-kaggle-replica.ipynb"),
        Path(r"C:\Users\ADMIN\OneDrive\Desktop\vex-eval-vex506-reinst.ipynb"),
        tmp_path / "audit.json",
    )
    recipe = token_swap_recipe_from_audit(audit)

    assert describe_merge_order(recipe) == ("cryptarithm_thk", "v6_cryptarithm", "binary_format", "hyprevise_binary_synth")
    assert recipe.exclude_heldout_ids is True


def test_token_swap_recipe_rejects_missing_private_inputs(tmp_path: Path) -> None:
    audit = audit_vex_notebooks(
        Path(r"C:\Users\ADMIN\OneDrive\Desktop\vex506-reinst-kaggle-replica.ipynb"),
        Path(r"C:\Users\ADMIN\OneDrive\Desktop\vex-eval-vex506-reinst.ipynb"),
        tmp_path / "audit.json",
    )
    recipe = token_swap_recipe_from_audit(audit)

    with pytest.raises(TokenSwapRecipeError):
        validate_token_swap_paths(recipe)

    result = validate_token_swap_paths(recipe, public_safe_fallback=True)
    assert result["public_safe_fallback"] is True
    assert result["missing_private_dependencies"]
