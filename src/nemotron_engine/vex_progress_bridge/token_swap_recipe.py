from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

from nemotron_engine.core.schemas import stable_hash


class TokenSwapRecipeError(ValueError):
    pass


@dataclass(frozen=True)
class TokenSwapRecipe:
    base_corpus: str | None
    ordered_override_swaps: tuple[tuple[str, str | None], ...]
    additive_synth: tuple[tuple[str, str | None], ...]
    heldout_file: str | None
    exclude_heldout_ids: bool
    recipe_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "ordered_override_swaps", tuple((str(k), v) for k, v in self.ordered_override_swaps))
        object.__setattr__(self, "additive_synth", tuple((str(k), v) for k, v in self.additive_synth))
        if not self.exclude_heldout_ids:
            raise TokenSwapRecipeError("heldout exclusion is required in VEX replica mode")
        _set_or_check_hash(self, "recipe_hash")


def token_swap_recipe_from_audit(audit: dict) -> TokenSwapRecipe:
    data = audit["extracted_data_recipe"]
    return TokenSwapRecipe(
        base_corpus=data.get("base_corpus"),
        ordered_override_swaps=(
            ("cryptarithm_thk", data.get("swap_cryptarithm_thk_path")),
            ("v6_cryptarithm", data.get("swap_v6_path")),
            ("binary_format", data.get("swap_binary_fmt_path")),
        ),
        additive_synth=(("hyprevise_binary_synth", data.get("extra_hyprevise_synth_path")),),
        heldout_file=data.get("heldout_file"),
        exclude_heldout_ids=bool(data.get("exclude_heldout_ids")),
    )


def describe_merge_order(recipe: TokenSwapRecipe) -> tuple[str, ...]:
    return tuple(name for name, _ in recipe.ordered_override_swaps) + tuple(name for name, _ in recipe.additive_synth)


def validate_token_swap_paths(
    recipe: TokenSwapRecipe,
    *,
    allow_missing_private_swaps: bool = False,
    public_safe_fallback: bool = False,
) -> dict:
    missing = []
    for label, path in (("base_corpus", recipe.base_corpus), ("heldout_file", recipe.heldout_file)):
        if not path or not Path(path).exists():
            missing.append(label if not path else path)
    for _, path in recipe.ordered_override_swaps + recipe.additive_synth:
        if not path or not Path(path).exists():
            missing.append(path or "missing_swap_path")
    if missing and not allow_missing_private_swaps and not public_safe_fallback:
        raise TokenSwapRecipeError(f"missing VEX private/token-swap inputs: {missing}")
    return {
        "available": not missing,
        "missing_private_dependencies": tuple(missing),
        "public_safe_fallback": public_safe_fallback,
        "merge_order": describe_merge_order(recipe),
    }


def _set_or_check_hash(instance: object, hash_field: str) -> None:
    expected = stable_hash({field.name: getattr(instance, field.name) for field in fields(instance) if field.name != hash_field})
    current = getattr(instance, hash_field)
    if not current:
        object.__setattr__(instance, hash_field, expected)
    elif current != expected:
        raise TokenSwapRecipeError(f"{hash_field} does not match payload")
