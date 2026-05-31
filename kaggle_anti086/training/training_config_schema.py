from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kaggle_anti086.kaggle_path_safety import require_writable_output_dir


REQUIRED_FIELDS = {
    "stage",
    "base_model_path",
    "parent_adapter_path",
    "train_direct_path",
    "train_solver_corrected_path",
    "train_abstain_safety_path",
    "train_hard_negative_path",
    "output_adapter_dir",
    "rank",
    "lora_alpha",
    "target_modules",
    "learning_rate",
    "max_seq_len",
    "micro_batch_size",
    "gradient_accumulation",
    "num_steps",
    "assistant_only_loss",
    "full_prompt_loss",
    "train_on_user",
    "direct_answer_weight",
    "solver_corrected_weight",
    "abstain_safety_weight",
    "hard_negative_weight",
    "adapter_size_limit_mb",
    "not_submission_ready",
}
STAGES = {"v2a_base_lora", "v2b_tinker_parent_lora", "v2c_best_parent_lora", "v3_hard_family_repair"}
V2_STAGES = {"v2a_base_lora", "v2b_tinker_parent_lora", "v2c_best_parent_lora"}
ALLOWED_TARGET_MODULES = {"q_proj", "v_proj", "o_proj"}
STATUS_PASS_RUNNABLE = "PASS_RUNNABLE"
STATUS_PARENT_MISSING = "PASS_BLOCKED_PARENT_MISSING"
STATUS_NOT_RUNNABLE = "PASS_NOT_RUNNABLE_YET"
STATUS_FAIL = "FAIL"


@dataclass(frozen=True)
class TrainingConfigValidation:
    path: str
    stage: str
    status: str
    runnable: bool
    warnings: list[str]
    failures: list[str]
    config: dict[str, Any]


def load_training_config(path: str | Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise ValueError(f"{path}: invalid config line: {line}")
        key, value = stripped.split(":", 1)
        data[key.strip()] = _parse_value(value.strip())
    return data


def validate_training_config(config: dict[str, Any], *, path: str | Path = "<memory>") -> TrainingConfigValidation:
    failures: list[str] = []
    warnings: list[str] = []
    missing = sorted(field for field in REQUIRED_FIELDS if field not in config)
    failures.extend(f"missing_{field}" for field in missing)
    stage = str(config.get("stage", ""))
    if stage not in STAGES:
        failures.append("unknown_stage")
    _check_bool(config, "assistant_only_loss", True, failures)
    _check_bool(config, "full_prompt_loss", False, failures)
    _check_bool(config, "train_on_user", False, failures)
    _check_bool(config, "not_submission_ready", True, failures)
    _check_numeric(config, "rank", max_value=32, failures=failures, integer=True)
    _check_numeric(config, "learning_rate", min_value=5e-8, max_value=2e-7, failures=failures)
    _check_numeric(config, "adapter_size_limit_mb", min_value=1, max_value=1500, failures=failures)
    if int(config.get("max_seq_len", 0) or 0) not in {1024, 1536}:
        failures.append("max_seq_len_must_be_1024_or_1536")
    steps = int(config.get("num_steps", 0) or 0)
    if stage in V2_STAGES and not (300 <= steps <= 800):
        failures.append("v2_num_steps_out_of_range")
    if stage == "v3_hard_family_repair" and bool(config.get("runnable", True)):
        failures.append("v3_must_be_non_runnable_day7")
    if float(config.get("direct_answer_weight", -1)) != 1.0:
        failures.append("direct_answer_weight_must_be_1")
    if float(config.get("solver_corrected_weight", 99)) > 0.5:
        failures.append("solver_corrected_weight_gt_0_5")
    if float(config.get("abstain_safety_weight", 99)) != 0.0:
        failures.append("abstain_safety_weight_must_be_0")
    if float(config.get("hard_negative_weight", 99)) != 0.0:
        failures.append("hard_negative_weight_must_be_0")
    targets = _target_modules(config)
    if "lm_head" in targets:
        failures.append("lm_head_forbidden")
    extras = sorted(set(targets) - ALLOWED_TARGET_MODULES)
    if extras:
        failures.append(f"forbidden_target_modules:{','.join(extras)}")
    if set(targets) != ALLOWED_TARGET_MODULES:
        failures.append("target_modules_must_be_q_proj_v_proj_o_proj")
    try:
        require_writable_output_dir(str(config.get("output_adapter_dir", "")), field_name="output_adapter_dir")
    except SystemExit as exc:
        failures.append(f"unsafe_output_adapter_dir:{exc}")
    parent = str(config.get("parent_adapter_path", ""))
    parent_missing = False
    if parent == "auto_or_none":
        failures.append("auto_or_none_parent_forbidden")
    elif parent not in {"none", ""}:
        if parent.startswith("REQUIRED_") or not Path(parent).exists():
            parent_missing = True
            warnings.append("parent_adapter_missing_blocks_config")
    base_model = str(config.get("base_model_path", ""))
    if base_model.startswith("/kaggle/input"):
        warnings.append("base_model_path_is_kaggle_placeholder")
    elif base_model and not Path(base_model).exists():
        warnings.append("base_model_path_missing_local")
    if stage == "v3_hard_family_repair":
        warnings.append("v3_blocked_until_v2_eval_failure_mining_and_repair_corpus")
    if failures:
        status = STATUS_FAIL
        runnable = False
    elif stage == "v3_hard_family_repair" or ("runnable" in config and bool(config.get("runnable")) is False):
        status = STATUS_NOT_RUNNABLE
        runnable = False
    elif parent_missing:
        status = STATUS_PARENT_MISSING
        runnable = False
    else:
        status = STATUS_PASS_RUNNABLE
        runnable = True
    return TrainingConfigValidation(str(path), stage, status, runnable, warnings, failures, dict(config))


def _parse_value(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if "," in value:
        return [item.strip() for item in value.split(",") if item.strip()]
    try:
        if any(ch in value for ch in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _target_modules(config: dict[str, Any]) -> list[str]:
    value = config.get("target_modules", [])
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value]


def _check_bool(config: dict[str, Any], field: str, expected: bool, failures: list[str]) -> None:
    if config.get(field) is not expected:
        failures.append(f"{field}_must_be_{str(expected).lower()}")


def _check_numeric(config: dict[str, Any], field: str, *, failures: list[str], min_value: float | None = None, max_value: float | None = None, integer: bool = False) -> None:
    try:
        value = int(config[field]) if integer else float(config[field])
    except (KeyError, TypeError, ValueError):
        failures.append(f"{field}_invalid")
        return
    if min_value is not None and value < min_value:
        failures.append(f"{field}_below_min")
    if max_value is not None and value > max_value:
        failures.append(f"{field}_above_max")
