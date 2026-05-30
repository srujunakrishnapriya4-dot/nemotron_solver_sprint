from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Mapping

from nemotron_engine.core.schemas import stable_hash


class VexRecipeError(ValueError):
    pass


@dataclass(frozen=True)
class VexTrainingRecipe:
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    max_seq_len: int
    num_steps: int
    batch_size: int
    micro_batch_size: int
    learning_rate: float
    reset_weights: bool | None
    in_proj_only: bool
    moe_tie_weights: bool
    shuffle_dataset: bool
    target_modules: tuple[str, ...]
    corpus_path: str | None
    train_order_path: str | None
    swap_cryptarithm_thk_path: str | None
    swap_v6_path: str | None
    swap_binary_fmt_path: str | None
    extra_hyprevise_synth_path: str | None
    heldout_file: str | None
    source_notebook_hash: str
    notebook_audit_found: bool = True
    lm_head_extracted: bool = False
    replica_mode: bool = True
    recipe_hash: str = ""

    def __post_init__(self) -> None:
        if self.lora_rank <= 0 or self.lora_rank > 32:
            raise VexRecipeError("rank must be <= 32")
        modules = tuple(str(module) for module in self.target_modules)
        if not modules:
            raise VexRecipeError("target_modules must be non-empty")
        if "lm_head" in modules and not self.lm_head_extracted:
            raise VexRecipeError("lm_head is allowed only when notebook audit extracted it")
        if self.reset_weights is None:
            raise VexRecipeError("reset_weights must be explicit")
        if self.replica_mode:
            if not self.notebook_audit_found:
                raise VexRecipeError("VEX replica mode requires real notebook audit")
            if not self.heldout_file:
                raise VexRecipeError("heldout_file is required for VEX replica mode")
        object.__setattr__(self, "target_modules", modules)
        _set_or_check_hash(self, "recipe_hash")


@dataclass(frozen=True)
class VexEvalRecipe:
    max_model_len: int
    max_new_tokens: int
    temperature: float
    top_p: float
    max_num_seqs: int
    gpu_mem_util: float
    category_weights: Mapping[str, float]
    held_variant_primary: str
    held_variant_secondary: str | None
    use_vllm: bool
    use_lora_request: bool
    eval_hash: str = ""

    def __post_init__(self) -> None:
        if self.max_model_len <= 0 or self.max_new_tokens <= 0:
            raise VexRecipeError("eval lengths must be positive")
        if self.max_num_seqs <= 0:
            raise VexRecipeError("max_num_seqs must be positive")
        weights = dict(sorted((str(k), float(v)) for k, v in self.category_weights.items()))
        if weights and abs(sum(weights.values()) - 1.0) > 1e-9:
            raise VexRecipeError("category weights must sum to 1")
        object.__setattr__(self, "category_weights", weights)
        _set_or_check_hash(self, "eval_hash")


def recipes_from_audit(audit: dict) -> tuple[VexTrainingRecipe, VexEvalRecipe]:
    train = audit["extracted_training_config"]
    data = audit["extracted_data_recipe"]
    eval_cfg = audit["extracted_eval_config"]
    training_recipe = VexTrainingRecipe(
        lora_rank=int(train["lora_rank"]),
        lora_alpha=int(train["lora_alpha"]),
        lora_dropout=float(train["lora_dropout"]),
        max_seq_len=int(train["max_seq_len"]),
        num_steps=int(train["num_steps"]),
        batch_size=int(train["batch_size"]),
        micro_batch_size=int(train["micro_batch_size"]),
        learning_rate=float(train["learning_rate"]),
        reset_weights=train["reset_weights"],
        in_proj_only=bool(train["in_proj_only"]),
        moe_tie_weights=bool(train["moe_tie_weights"]),
        shuffle_dataset=bool(train["shuffle_dataset"]),
        target_modules=tuple(train["target_modules"]),
        corpus_path=data.get("base_corpus"),
        train_order_path=data.get("train_order_path"),
        swap_cryptarithm_thk_path=data.get("swap_cryptarithm_thk_path"),
        swap_v6_path=data.get("swap_v6_path"),
        swap_binary_fmt_path=data.get("swap_binary_fmt_path"),
        extra_hyprevise_synth_path=data.get("extra_hyprevise_synth_path"),
        heldout_file=data.get("heldout_file"),
        source_notebook_hash=str(audit["audit_hash"]),
        notebook_audit_found=bool(audit["notebooks_found"]),
        lm_head_extracted="lm_head" in tuple(train["target_modules"]),
        replica_mode=True,
    )
    eval_recipe = VexEvalRecipe(
        max_model_len=int(eval_cfg["max_model_len"]),
        max_new_tokens=int(eval_cfg["max_new_tokens"]),
        temperature=float(eval_cfg["temperature"]),
        top_p=float(eval_cfg["top_p"]),
        max_num_seqs=int(eval_cfg["max_num_seqs"]),
        gpu_mem_util=float(eval_cfg["gpu_mem_util"]),
        category_weights=eval_cfg["category_weights"],
        held_variant_primary=str(eval_cfg["held_variant_primary"]),
        held_variant_secondary=eval_cfg.get("held_variant_secondary"),
        use_vllm=bool(eval_cfg["use_vllm"]),
        use_lora_request=bool(eval_cfg["use_lora_request"]),
    )
    return training_recipe, eval_recipe


def _set_or_check_hash(instance: object, hash_field: str) -> None:
    expected = stable_hash({field.name: getattr(instance, field.name) for field in fields(instance) if field.name != hash_field})
    current = getattr(instance, hash_field)
    if not current:
        object.__setattr__(instance, hash_field, expected)
    elif current != expected:
        raise VexRecipeError(f"{hash_field} does not match payload")
