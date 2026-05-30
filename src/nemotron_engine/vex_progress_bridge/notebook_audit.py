from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from nemotron_engine.core.schemas import stable_hash


TRAIN_NOTEBOOK = "vex506-reinst-kaggle-replica.ipynb"
EVAL_NOTEBOOK = "vex-eval-vex506-reinst.ipynb"
DEFAULT_SEARCH_DIRS = (
    Path("."),
    Path("artifacts"),
    Path("artifacts/vex_progress"),
    Path(r"C:\Users\ADMIN\OneDrive\Desktop"),
)


def audit_vex_notebooks(
    train_notebook: str | Path = TRAIN_NOTEBOOK,
    eval_notebook: str | Path = EVAL_NOTEBOOK,
    output_path: str | Path = "artifacts/vex_progress/notebook_audit.json",
) -> dict[str, Any]:
    train_path = _resolve_notebook(Path(train_notebook))
    eval_path = _resolve_notebook(Path(eval_notebook))
    train_text = _notebook_text(train_path) if train_path else ""
    eval_text = _notebook_text(eval_path) if eval_path else ""
    training_config = _extract_training_config(train_text)
    eval_config = _extract_eval_config(eval_text)
    data_recipe = _extract_data_recipe(train_text, eval_text)
    audit = {
        "training_notebook": str(train_path or train_notebook),
        "eval_notebook": str(eval_path or eval_notebook),
        "training_present": train_path is not None,
        "eval_present": eval_path is not None,
        "notebooks_found": train_path is not None and eval_path is not None,
        "extracted_training_config": training_config,
        "extracted_eval_config": eval_config,
        "extracted_data_recipe": data_recipe,
        "training_hyperparameters": _extract_hparams(train_text),
        "target_modules": tuple(training_config.get("target_modules", ())),
        "features": {
            "cut_cross_entropy": _contains(train_text, "cut_cross_entropy", "cut-cross-entropy", "linear_cross_entropy"),
            "labels_weighted_masked": _contains(train_text, "loss_weights", "target_ids", "weights", "masked"),
            "unsloth_fast_language_model": _contains(train_text, "FastLanguageModel", "unsloth"),
            "adapter_loaded_from_pretrained": _contains(train_text, "PeftModel.from_pretrained", "load_adapter", "from_pretrained"),
            "fresh_reset_training": _contains(train_text, "RESET_WEIGHTS", "reset_lora", "reset_weights"),
            "tinker_moe_tied_weights": _contains(train_text, "tied", "MoE", "tinker"),
            "preprocessed_tokens_loaded": _contains(train_text, "train_tokens", "input_ids", "token_manifest", "loss_weights"),
            "lm_head_lora_added": _contains(train_text, "lm_head"),
        },
        "eval_settings": {
            "vllm_usage": _contains(eval_text, "vllm", "LLM(", "LoRARequest"),
            "lora_request_usage": _contains(eval_text, "LoRARequest"),
            "answer_extraction_function": _contains(eval_text, "extract_answer", "boxed", "answer"),
            "category_weighting_lb_calibration": _contains(eval_text, "category", "weight", "leaderboard", "lb"),
            "heldout_file_usage": _contains(eval_text, "heldout", "private", "validation"),
            "train_slice_usage": _contains(eval_text, "train", "slice"),
            "eval_output_files": _contains(eval_text, "jsonl", "eval_summary", "per_puzzle"),
            "max_model_len": eval_config.get("max_model_len"),
            "max_tokens": eval_config.get("max_new_tokens"),
            "temperature": eval_config.get("temperature"),
            "top_p": eval_config.get("top_p"),
            "max_num_seqs": eval_config.get("max_num_seqs"),
        },
        "reusable_design_points": (
            "pretokenized corpus with explicit target_ids/loss_weights",
            "assistant/answer masking before expensive model load",
            "custom weighted token loss instead of generic full-prompt Trainer loss",
            "explicit LoRA target-module control and rank-32 packaging gate",
            "vLLM/LoRARequest-style fast eval before packaging",
        ),
        "unavailable_private_dependencies": _private_dependencies(train_text + "\n" + eval_text, data_recipe),
        "unsafe_assumptions": (
            "private heldout files cannot be assumed present",
            "leaderboard calibration must not be the only promotion signal",
            "4GB custom adapters are rejected unless known-good structure is proven",
        ),
        "implementation_requirements": (
            "verified rule exporter",
            "VEX token corpus schema",
            "mask/weight builder with prompt-leak diagnostics",
            "custom weighted SFT loop",
            "vLLM-like smoke/family eval",
            "rank/size/eval submission gate",
        ),
        "required_kaggle_inputs": _required_kaggle_inputs(data_recipe, eval_config),
        "missing_instruction": None
        if train_path and eval_path
        else "Copy vex506-reinst-kaggle-replica.ipynb and vex-eval-vex506-reinst.ipynb into C:\\Users\\ADMIN\\nemotron_solver_sprint and rerun.",
    }
    audit["audit_hash"] = stable_hash(audit)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, sort_keys=True, indent=2), encoding="utf-8")
    return audit


def _resolve_notebook(path: Path) -> Path | None:
    if path.exists():
        return path
    for directory in DEFAULT_SEARCH_DIRS:
        candidate = directory / path.name
        if candidate.exists():
            return candidate
    return None


def _notebook_text(path: Path) -> str:
    if not path.exists():
        return ""
    payload = json.loads(path.read_text(encoding="utf-8"))
    chunks: list[str] = []
    for cell in payload.get("cells", []):
        source = cell.get("source", "")
        chunks.append("".join(source) if isinstance(source, list) else str(source))
    return "\n".join(chunks)


def _extract_hparams(text: str) -> dict[str, Any]:
    keys = (
        "LORA_RANK",
        "LORA_ALPHA",
        "LORA_DROPOUT",
        "MAX_SEQ_LEN",
        "NUM_STEPS",
        "BATCH_SIZE",
        "MICRO_BATCH_SIZE",
        "LEARNING_RATE",
        "RESET_WEIGHTS",
        "IN_PROJ_ONLY",
        "MOE_TIE_WEIGHTS",
        "ORIGINAL_PROBLEMS_ONLY",
        "SHUFFLE_DATASET",
    )
    return {key: _extract_assignment(text, key) for key in keys}


def _extract_training_config(text: str) -> dict[str, Any]:
    return {
        "lora_rank": _as_int(_extract_assignment(text, "LORA_RANK")),
        "lora_alpha": _as_int(_extract_assignment(text, "LORA_ALPHA")),
        "lora_dropout": _as_float(_extract_assignment(text, "LORA_DROPOUT")),
        "max_seq_len": _as_int(_extract_assignment(text, "MAX_SEQ_LEN")),
        "num_steps": _as_int(_extract_assignment(text, "NUM_STEPS")),
        "batch_size": _as_int(_extract_assignment(text, "BATCH_SIZE")),
        "micro_batch_size": _as_int(_extract_assignment(text, "MICRO_BATCH_SIZE")),
        "learning_rate": _as_float(_extract_assignment(text, "LEARNING_RATE")),
        "reset_weights": _as_bool(_extract_assignment(text, "RESET_WEIGHTS")),
        "in_proj_only": _as_bool(_extract_assignment(text, "IN_PROJ_ONLY")),
        "moe_tie_weights": _as_bool(_extract_assignment(text, "MOE_TIE_WEIGHTS")),
        "original_problems_only": _as_bool(_extract_assignment(text, "ORIGINAL_PROBLEMS_ONLY")),
        "shuffle_dataset": _as_bool(_extract_assignment(text, "SHUFFLE_DATASET")),
        "target_modules": _extract_target_modules(text),
        "uses_unsloth": _contains(text, "FastLanguageModel"),
        "uses_cut_cross_entropy": _contains(text, "linear_cross_entropy"),
        "uses_generic_trainer": _contains(text, "Trainer("),
        "manually_adds_lm_head_lora": _contains(text, "Manually add lm_head LoRA", "lm_head LoRA"),
        "uses_token_level_weights": _contains(text, "loss_weights", "weights"),
    }


def _extract_eval_config(text: str) -> dict[str, Any]:
    return {
        "use_vllm": _contains(text, "from vllm import LLM", "LLM("),
        "use_lora_request": _contains(text, "LoRARequest"),
        "sampling_params": _contains(text, "SamplingParams"),
        "temperature": _as_float(_extract_assignment(text, "TEMPERATURE")),
        "top_p": _as_float(_extract_assignment(text, "TOP_P")),
        "max_new_tokens": _as_int(_extract_assignment(text, "MAX_NEW_TOKENS")),
        "max_model_len": _as_int(_extract_assignment(text, "MAX_MODEL_LEN")),
        "max_num_seqs": _as_int(_extract_assignment(text, "MAX_NUM_SEQS")),
        "gpu_mem_util": _as_float(_extract_assignment(text, "GPU_MEM_UTIL")),
        "enable_lora": _contains(text, "enable_lora=True"),
        "max_lora_rank": _as_int(_extract_number(text, "max_lora_rank")),
        "enable_prefix_caching": _contains(text, "enable_prefix_caching=True"),
        "enable_chunked_prefill": _contains(text, "enable_chunked_prefill=True"),
        "boxed_answer_first": _contains(text, r"\\boxed", "boxed"),
        "numeric_fallback": _contains(text, "re.findall", r"\\d"),
        "category_weights": _extract_category_weights(text),
        "held_variant_primary": _strip_quotes(_extract_assignment(text, "HELD_VARIANT")),
        "held_variant_secondary": _strip_quotes(_extract_assignment(text, "HELD_VARIANT_GEN")),
        "train_excludes_heldout_ids": _contains(text, "all_held_pids", "excluded from train"),
    }


def _extract_data_recipe(train_text: str, eval_text: str) -> dict[str, Any]:
    return {
        "base_corpus": _strip_quotes(_extract_assignment(train_text, "CORPUS_PATH")),
        "train_order_path": _strip_quotes(_extract_assignment(train_text, "TRAIN_ORDER_PATH")),
        "swap_cryptarithm_thk_path": _strip_quotes(_extract_assignment(train_text, "SWAP_CRYPTARITHM_THK")),
        "swap_v6_path": _strip_quotes(_extract_assignment(train_text, "SWAP_V6")),
        "swap_binary_fmt_path": _strip_quotes(_extract_assignment(train_text, "SWAP_BINARY_FMT")),
        "extra_hyprevise_synth_path": _strip_quotes(_extract_assignment(train_text, "EXTRA_HYPREVISE_SYNTH")),
        "heldout_file": _strip_quotes(_extract_assignment(train_text, "HELDOUT_FILE")),
        "exclude_heldout_ids": _contains(train_text + "\n" + eval_text, "heldout", "excluded from train"),
        "shuffle_dataset": _as_bool(_extract_assignment(train_text, "SHUFFLE_DATASET")),
        "token_level_corpus_weights": _contains(train_text, "input_ids", "target_ids", "weights"),
        "swap_order": ("cryptarithm_thk", "v6_cryptarithm", "binary_fmt"),
        "additive_synth": "hyprevise_synth" if _contains(train_text, "EXTRA_HYPREVISE_SYNTH") else None,
    }


def _extract_target_modules(text: str) -> tuple[str, ...]:
    block = re.search(r"\bTARGET_MODULES\s*=\s*\[(?P<body>.*?)\]", text, re.DOTALL)
    source = block.group("body") if block else text
    modules = sorted(
        set(
            re.findall(
                r"['\"]([a-z_]*(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj|out_proj|lm_head)[a-z_]*)['\"]",
                source,
            )
        )
    )
    preferred = ("q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "in_proj", "out_proj", "lm_head")
    modules = [module for module in preferred if module in modules] + [module for module in modules if module not in preferred]
    return tuple(modules)


def _extract_assignment(text: str, key: str) -> str | None:
    match = re.search(rf"\b{re.escape(key)}\s*=\s*(.+?)(?:\n[A-Z_]+\s*=|\n#|$)", text, re.DOTALL)
    if not match:
        return None
    value = match.group(1).strip()
    if value.startswith("("):
        bool_match = re.search(r"\b(True|False)\b", value)
        if bool_match:
            return bool_match.group(1)
    return value.splitlines()[0].strip()


def _extract_number(text: str, key: str) -> str | None:
    match = re.search(rf"\b{re.escape(key)}\s*=?\s*([0-9.]+)", text, re.IGNORECASE)
    return match.group(1) if match else None


def _extract_category_weights(text: str) -> dict[str, float]:
    match = re.search(r"LB_W\s*=\s*\{(?P<body>.*?)\}", text, re.DOTALL)
    if not match:
        return {}
    weights: dict[str, float] = {}
    for key, numerator, denominator in re.findall(r"['\"]([^'\"]+)['\"]\s*:\s*(\d+)\s*/\s*(\d+)", match.group("body")):
        weights[key] = int(numerator) / int(denominator)
    return dict(sorted(weights.items()))


def _contains(text: str, *needles: str) -> bool:
    lower = text.lower()
    return any(needle.lower() in lower for needle in needles)


def _private_dependencies(text: str, data_recipe: dict[str, Any] | None = None) -> tuple[str, ...]:
    lower = text.lower()
    deps = []
    for needle in ("heldout", "private", "vex506", "preprocessed", "tinker"):
        if needle in lower:
            deps.append(needle)
    for key, value in (data_recipe or {}).items():
        if isinstance(value, str) and value.startswith("/kaggle/input/"):
            deps.append(f"{key}:{value}")
    return tuple(sorted(set(deps)))


def _required_kaggle_inputs(data_recipe: dict[str, Any], eval_config: dict[str, Any]) -> tuple[str, ...]:
    required = []
    for key in (
        "base_corpus",
        "train_order_path",
        "swap_cryptarithm_thk_path",
        "swap_v6_path",
        "swap_binary_fmt_path",
        "extra_hyprevise_synth_path",
        "heldout_file",
    ):
        value = data_recipe.get(key)
        if value:
            required.append(str(value))
    if eval_config.get("held_variant_primary"):
        required.append(f"eval_holdout_{eval_config['held_variant_primary']}.jsonl")
    if eval_config.get("held_variant_secondary"):
        required.append(f"eval_holdout_{eval_config['held_variant_secondary']}.jsonl")
    return tuple(required)


def _strip_quotes(value: str | None) -> str | None:
    if value is None:
        return None
    return value.split("#", 1)[0].strip().strip(",").strip('"').strip("'")


def _as_int(value: str | None) -> int | None:
    if value is None:
        return None
    match = re.search(r"-?\d+", value)
    return int(match.group(0)) if match else None


def _as_float(value: str | None) -> float | None:
    if value is None:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", value, re.IGNORECASE)
    return float(match.group(0)) if match else None


def _as_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    match = re.search(r"\b(True|False)\b", value)
    if match:
        return match.group(1) == "True"
    return None
