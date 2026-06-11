from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
import shutil
import stat
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PHASE = "PHASE9_CORE_SFT_ADAPTER_TRAINING_MATRIX"

DEFAULT_MODEL_PATH = "/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1"

PTXAS_BLACKWELL_SOURCE = (
    "/kaggle/usr/lib/notebooks/ryanholbrook/"
    "nvidia-utility-script/triton/backends/nvidia/bin/ptxas-blackwell"
)

ALLOWED_TARGET_MODULES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj",
    "out_proj",
    "up_proj",
    "down_proj",
}

FIRST_WAVE_ORDER = [
    "A1_direct_qv_r32",
    "A3_mixed_curriculum_qv_r32",
    "A2_short_trace_qv_r32",
    "A4_format_heavy_qv_r32",
    "A5_composed_heavy_qv_r32",
]


@dataclass(frozen=True)
class Phase9Variant:
    adapter_name: str
    dataset_variant: str
    dataset_filename: str
    target_modules: Tuple[str, ...]
    rank: int
    alpha: int
    dropout: float
    learning_rate: float
    max_seq_len: int
    train_rows: int
    eval_rows: int
    max_steps: int
    warmup_ratio: float
    weight_decay: float
    gradient_accumulation_steps: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int


VARIANTS: Dict[str, Phase9Variant] = {
    "A1_direct_qv_r32": Phase9Variant(
        adapter_name="A1_direct_qv_r32",
        dataset_variant="direct",
        dataset_filename="day1x_sft_direct.jsonl",
        target_modules=("q_proj", "v_proj"),
        rank=32,
        alpha=64,
        dropout=0.05,
        learning_rate=2e-5,
        max_seq_len=2048,
        train_rows=50000,
        eval_rows=5000,
        max_steps=0,
        warmup_ratio=0.03,
        weight_decay=0.0,
        gradient_accumulation_steps=8,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
    ),
    "A2_short_trace_qv_r32": Phase9Variant(
        adapter_name="A2_short_trace_qv_r32",
        dataset_variant="short_trace",
        dataset_filename="day1x_sft_short_trace.jsonl",
        target_modules=("q_proj", "v_proj"),
        rank=32,
        alpha=64,
        dropout=0.05,
        learning_rate=2e-5,
        max_seq_len=2048,
        train_rows=50000,
        eval_rows=5000,
        max_steps=0,
        warmup_ratio=0.03,
        weight_decay=0.0,
        gradient_accumulation_steps=8,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
    ),
    "A3_mixed_curriculum_qv_r32": Phase9Variant(
        adapter_name="A3_mixed_curriculum_qv_r32",
        dataset_variant="mixed_curriculum",
        dataset_filename="day1x_sft_mixed_curriculum.jsonl",
        target_modules=("q_proj", "v_proj"),
        rank=32,
        alpha=64,
        dropout=0.05,
        learning_rate=2e-5,
        max_seq_len=2048,
        train_rows=50000,
        eval_rows=5000,
        max_steps=0,
        warmup_ratio=0.03,
        weight_decay=0.0,
        gradient_accumulation_steps=8,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
    ),
    "A4_format_heavy_qv_r32": Phase9Variant(
        adapter_name="A4_format_heavy_qv_r32",
        dataset_variant="format_heavy",
        dataset_filename="day1x_sft_format_heavy.jsonl",
        target_modules=("q_proj", "v_proj"),
        rank=32,
        alpha=64,
        dropout=0.05,
        learning_rate=2e-5,
        max_seq_len=2048,
        train_rows=50000,
        eval_rows=5000,
        max_steps=0,
        warmup_ratio=0.03,
        weight_decay=0.0,
        gradient_accumulation_steps=8,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
    ),
    "A5_composed_heavy_qv_r32": Phase9Variant(
        adapter_name="A5_composed_heavy_qv_r32",
        dataset_variant="composed_heavy",
        dataset_filename="day1x_sft_composed_heavy.jsonl",
        target_modules=("q_proj", "v_proj"),
        rank=32,
        alpha=64,
        dropout=0.05,
        learning_rate=2e-5,
        max_seq_len=2048,
        train_rows=50000,
        eval_rows=5000,
        max_steps=0,
        warmup_ratio=0.03,
        weight_decay=0.0,
        gradient_accumulation_steps=8,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
    ),
}


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return obj


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"bad_jsonl:{path}:{i}:{e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"jsonl_row_not_object:{path}:{i}")
            rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")
            count += 1
    tmp.replace(path)
    return count


def boxed_count(text: str) -> int:
    return text.count("\\boxed{")


def row_prompt(row: Dict[str, Any]) -> str:
    for key in ("prompt", "input", "question", "problem"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("row_missing_prompt")


def row_target(row: Dict[str, Any]) -> str:
    for key in ("target", "completion", "response", "assistant"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("row_missing_target")


def validate_sft_row(row: Dict[str, Any], idx: int) -> Optional[str]:
    try:
        prompt = row_prompt(row)
        target = row_target(row)
    except ValueError as e:
        return f"{idx}:{e}"

    if "ABSTAIN" in target.upper():
        return f"{idx}:target_contains_abstain"

    if boxed_count(target) != 1:
        return f"{idx}:target_must_have_exactly_one_boxed"

    answer = row.get("answer")
    if isinstance(answer, str) and answer.strip():
        if answer.strip() not in target:
            return f"{idx}:answer_not_present_in_target"

    verification_status = row.get("verification_status")
    if verification_status not in (None, "PASS", True):
        return f"{idx}:verification_status_not_pass"

    ambiguity_count = row.get("ambiguity_count")
    if ambiguity_count not in (None, 0):
        return f"{idx}:ambiguity_count_nonzero"

    return None


def validate_variant_config(v: Phase9Variant) -> List[str]:
    reasons: List[str] = []

    if v.rank <= 0 or v.rank > 32:
        reasons.append("rank_must_be_1_to_32")

    if len(set(v.target_modules)) != len(v.target_modules):
        reasons.append("duplicate_target_modules")

    unknown = [m for m in v.target_modules if m not in ALLOWED_TARGET_MODULES]
    if unknown:
        reasons.append("unknown_target_modules:" + ",".join(unknown))

    bad = [m for m in v.target_modules if m.lower() in {"all-linear", "all_linear", "*"}]
    if bad:
        reasons.append("unsafe_all_linear_target_modules")

    if not (1e-6 <= v.learning_rate <= 8e-5):
        reasons.append("learning_rate_outside_safe_band")

    if v.max_seq_len not in {1024, 1536, 2048}:
        reasons.append("max_seq_len_not_allowed")

    if not (0.0 <= v.dropout <= 0.2):
        reasons.append("dropout_outside_safe_band")

    if v.train_rows <= 0:
        reasons.append("train_rows_nonpositive")

    if v.eval_rows < 0:
        reasons.append("eval_rows_negative")

    if v.gradient_accumulation_steps <= 0:
        reasons.append("gradient_accumulation_steps_nonpositive")

    return reasons


def phase8_gate(out_dir: Path) -> List[str]:
    reasons: List[str] = []

    smoke_train = out_dir / "day2_smoke_train_report.json"
    delta_eval = out_dir / "day2_adapter_delta_smoke_report.json"

    if not smoke_train.exists():
        reasons.append("missing_phase8_smoke_train_report")
    else:
        r = read_json(smoke_train)
        if r.get("status") != "PASS":
            reasons.append("phase8_smoke_train_not_pass")
        if r.get("package_authorized") is not False:
            reasons.append("phase8_smoke_package_flag_not_false")
        if r.get("submission_authorized") is not False:
            reasons.append("phase8_smoke_submission_flag_not_false")
        if r.get("leaderboard_claim") is not False:
            reasons.append("phase8_smoke_leaderboard_flag_not_false")

    if not delta_eval.exists():
        reasons.append("missing_phase8_adapter_delta_smoke_report")
    else:
        r = read_json(delta_eval)
        if r.get("status") != "PASS":
            reasons.append("phase8_delta_eval_not_pass")
        if r.get("safe_for_full_training_matrix") is not True:
            reasons.append("phase8_not_safe_for_full_training_matrix")
        if r.get("package_authorized") not in (None, False):
            reasons.append("phase8_delta_package_flag_not_false")
        if r.get("submission_authorized") not in (None, False):
            reasons.append("phase8_delta_submission_flag_not_false")
        if r.get("leaderboard_claim") not in (None, False):
            reasons.append("phase8_delta_leaderboard_flag_not_false")

    return reasons


def setup_blackwell_ptxas() -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "ptxas_blackwell_fix": "not_attempted",
        "source": PTXAS_BLACKWELL_SOURCE,
        "target": "/tmp/ptxas-blackwell",
        "target_exists": False,
    }

    src = Path(PTXAS_BLACKWELL_SOURCE)
    dst = Path("/tmp/ptxas-blackwell")

    if not src.exists():
        result["ptxas_blackwell_fix"] = "source_missing"
        return result

    try:
        shutil.copy2(src, dst)
        mode = dst.stat().st_mode
        os.chmod(dst, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        os.environ["TRITON_PTXAS_PATH"] = str(dst)
        os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = str(dst)
        result["ptxas_blackwell_fix"] = "enabled"
        result["target_exists"] = dst.exists()
        result["mode"] = oct(dst.stat().st_mode)
    except Exception as e:
        result["ptxas_blackwell_fix"] = "failed"
        result["error"] = repr(e)

    return result


def source_dataset_path(out_dir: Path, v: Phase9Variant) -> Path:
    return out_dir / v.dataset_filename


def materialized_dataset_paths(out_dir: Path, adapter_name: str) -> Tuple[Path, Path]:
    root = out_dir / "day2_phase9_datasets"
    return root / f"{adapter_name}_train.jsonl", root / f"{adapter_name}_eval.jsonl"


def materialize_variant_dataset(
    out_dir: Path,
    v: Phase9Variant,
    seed: int = 1234,
    train_rows_override: Optional[int] = None,
    eval_rows_override: Optional[int] = None,
) -> Dict[str, Any]:
    src = source_dataset_path(out_dir, v)
    if not src.exists():
        raise FileNotFoundError(f"missing_source_dataset:{src}")

    rows = read_jsonl(src)
    if not rows:
        raise ValueError(f"empty_source_dataset:{src}")

    bad: List[str] = []
    checked_rows = 0
    for idx, row in enumerate(rows[: min(len(rows), 10000)], start=1):
        checked_rows += 1
        reason = validate_sft_row(row, idx)
        if reason:
            bad.append(reason)
            if len(bad) >= 20:
                break

    if bad:
        raise ValueError("sft_row_validation_failed:" + ";".join(bad))

    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)

    train_cap = train_rows_override if train_rows_override is not None else v.train_rows
    eval_cap = eval_rows_override if eval_rows_override is not None else v.eval_rows

    train = shuffled[: min(train_cap, len(shuffled))]
    rest = shuffled[min(train_cap, len(shuffled)) :]
    eval_rows = rest[: min(eval_cap, len(rest))]

    if not eval_rows and eval_cap > 0:
        eval_rows = shuffled[-min(eval_cap, len(shuffled)) :]

    train_path, eval_path = materialized_dataset_paths(out_dir, v.adapter_name)
    train_count = write_jsonl(train_path, train)
    eval_count = write_jsonl(eval_path, eval_rows)

    return {
        "source_path": str(src),
        "source_rows": len(rows),
        "checked_rows": checked_rows,
        "train_path": str(train_path),
        "eval_path": str(eval_path),
        "train_rows": train_count,
        "eval_rows": eval_count,
    }


def adapter_dir_valid(adapter_path: Path) -> bool:
    if not adapter_path.exists() or not adapter_path.is_dir():
        return False
    if not (adapter_path / "adapter_config.json").exists():
        return False
    return (adapter_path / "adapter_model.safetensors").exists() or (adapter_path / "adapter_model.bin").exists()


def finite_loss_stats(log_history: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    losses: List[float] = []
    nan_count = 0
    inf_count = 0

    for item in log_history:
        if not isinstance(item, dict):
            continue
        if "loss" not in item:
            continue
        try:
            value = float(item["loss"])
        except Exception:
            nan_count += 1
            continue

        if math.isnan(value):
            nan_count += 1
        elif math.isinf(value):
            inf_count += 1
        else:
            losses.append(value)

    return {
        "train_loss_start": losses[0] if losses else None,
        "train_loss_end": losses[-1] if losses else None,
        "loss_finite": bool(losses) and nan_count == 0 and inf_count == 0,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "loss_count": len(losses),
    }


def sanitize_no_split_modules(model: Any) -> None:
    value = getattr(model, "_no_split_modules", None)
    if value is None:
        return

    def flatten(x: Any) -> List[str]:
        if x is None:
            return []
        if isinstance(x, str):
            return [x]
        if isinstance(x, (list, tuple, set)):
            out: List[str] = []
            for item in x:
                out.extend(flatten(item))
            return out
        return [str(x)]

    try:
        setattr(model, "_no_split_modules", sorted(set(flatten(value))))
    except Exception:
        pass


def training_arguments_kwargs(**kwargs: Any) -> Dict[str, Any]:
    from transformers import TrainingArguments

    sig = inspect.signature(TrainingArguments.__init__)
    params = set(sig.parameters)

    final = dict(kwargs)

    if "eval_strategy" in params and "evaluation_strategy" in final:
        final["eval_strategy"] = final.pop("evaluation_strategy")
    elif "evaluation_strategy" in params and "eval_strategy" in final:
        final["evaluation_strategy"] = final.pop("eval_strategy")

    return {k: v for k, v in final.items() if k in params}


class SFTDataset:
    def __init__(self, rows: List[Dict[str, Any]], tokenizer: Any, max_seq_len: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        prompt = row_prompt(row)
        target = row_target(row)

        prompt_text = prompt if prompt.endswith("\n") else prompt + "\n"
        eos = self.tokenizer.eos_token or ""
        full_text = prompt_text + target + eos

        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_seq_len,
        )["input_ids"]

        encoded = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_seq_len,
            padding=False,
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        labels = list(input_ids)

        prompt_len = min(len(prompt_ids), len(labels))
        for i in range(prompt_len):
            labels[i] = -100

        if all(x == -100 for x in labels):
            labels[-1] = input_ids[-1]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def train_one_variant(
    *,
    out_dir: Path,
    model_path: str,
    variant_name: str,
    train_rows_override: Optional[int],
    eval_rows_override: Optional[int],
    max_steps_override: Optional[int],
    seed: int,
) -> Dict[str, Any]:
    v = VARIANTS[variant_name]

    config_reasons = validate_variant_config(v)
    if config_reasons:
        raise ValueError("unsafe_variant_config:" + ",".join(config_reasons))

    ptxas_report = setup_blackwell_ptxas()

    dataset_info = materialize_variant_dataset(
        out_dir,
        v,
        seed=seed,
        train_rows_override=train_rows_override,
        eval_rows_override=eval_rows_override,
    )

    train_path = Path(dataset_info["train_path"])
    eval_path = Path(dataset_info["eval_path"])
    train_rows = read_jsonl(train_path)
    eval_rows = read_jsonl(eval_path)

    adapter_path = out_dir / "adapters" / v.adapter_name
    trainer_output_dir = out_dir / "tmp_trainer" / v.adapter_name
    adapter_path.mkdir(parents=True, exist_ok=True)
    trainer_output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    torch.manual_seed(seed)
    random.seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map=None,
    )

    sanitize_no_split_modules(model)

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=v.rank,
        lora_alpha=v.alpha,
        lora_dropout=v.dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=list(v.target_modules),
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = SFTDataset(train_rows, tokenizer=tokenizer, max_seq_len=v.max_seq_len)
    eval_dataset = SFTDataset(eval_rows, tokenizer=tokenizer, max_seq_len=v.max_seq_len) if eval_rows else None

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        import torch

        max_len = max(len(x["input_ids"]) for x in batch)
        pad_id = tokenizer.pad_token_id

        input_ids = []
        attention_mask = []
        labels = []

        for item in batch:
            n = len(item["input_ids"])
            pad = max_len - n
            input_ids.append(item["input_ids"]
 + [pad_id] * pad)
            attention_mask.append(item["attention_mask"] + [0] * pad)
            labels.append(item["labels"] + [-100] * pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    max_steps = max_steps_override if max_steps_override is not None else v.max_steps

    base_training_kwargs: Dict[str, Any] = {
        "output_dir": str(trainer_output_dir),
        "overwrite_output_dir": True,
        "per_device_train_batch_size": v.per_device_train_batch_size,
        "per_device_eval_batch_size": v.per_device_eval_batch_size,
        "gradient_accumulation_steps": v.gradient_accumulation_steps,
        "learning_rate": v.learning_rate,
        "weight_decay": v.weight_decay,
        "warmup_ratio": v.warmup_ratio,
        "num_train_epochs": 1.0,
        "max_steps": max_steps if max_steps and max_steps > 0 else -1,
        "bf16": True,
        "fp16": False,
        "logging_steps": 5,
        "save_strategy": "no",
        "evaluation_strategy": "steps" if eval_dataset is not None else "no",
        "eval_steps": 100 if eval_dataset is not None else None,
        "report_to": [],
        "remove_unused_columns": False,
        "gradient_checkpointing": True,
        "optim": "paged_adamw_8bit",
        "seed": seed,
    }

    if base_training_kwargs["eval_steps"] is None:
        base_training_kwargs.pop("eval_steps", None)

    training_args = TrainingArguments(
        **training_arguments_kwargs(**base_training_kwargs)
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate,
        tokenizer=tokenizer,
    )

    trainer.train()
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))

    stats = finite_loss_stats(trainer.state.log_history)

    adapter_valid = adapter_dir_valid(adapter_path)
    safe_for_eval = (
        adapter_valid
        and stats["loss_finite"] is True
        and stats["nan_count"] == 0
        and stats["inf_count"] == 0
        and v.rank <= 32
    )

    report = {
        "schema_version": 1,
        "phase": PHASE,
        "status": "PASS" if safe_for_eval else "FAIL",
        "adapter_name": v.adapter_name,
        "dataset_variant": v.dataset_variant,
        "dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "target_modules": list(v.target_modules),
        "rank": v.rank,
        "alpha": v.alpha,
        "dropout": v.dropout,
        "learning_rate": v.learning_rate,
        "max_seq_len": v.max_seq_len,
        "train_rows": int(dataset_info["train_rows"]),
        "eval_rows": int(dataset_info["eval_rows"]),
        "train_loss_start": stats["train_loss_start"],
        "train_loss_end": stats["train_loss_end"],
        "loss_finite": stats["loss_finite"],
        "nan_count": stats["nan_count"],
        "inf_count": stats["inf_count"],
        "loss_count": stats["loss_count"],
        "adapter_path": str(adapter_path),
        "adapter_dir_valid": adapter_valid,
        "safe_for_eval": safe_for_eval,
        "package_authorized": False,
        "submission_authorized": False,
        "leaderboard_claim": False,
        "ptxas_report": ptxas_report,
        "dataset_info": dataset_info,
        "blocked_reasons": [] if safe_for_eval else ["adapter_not_safe_for_eval"],
    }

    report_path = out_dir / f"day2_train_report_{v.adapter_name}.json"
    write_json(report_path, report)
    return report


def prepare_plan(out_dir: Path, variants: Sequence[str]) -> Dict[str, Any]:
    blocked_reasons = phase8_gate(out_dir)

    variant_entries = []
    for name in variants:
        if name not in VARIANTS:
            blocked_reasons.append(f"unknown_variant:{name}")
            continue
        v = VARIANTS[name]
        reasons = validate_variant_config(v)
        if reasons:
            blocked_reasons.append(f"{name}:unsafe_config:" + ",".join(reasons))
        variant_entries.append(asdict(v))

    plan = {
        "schema_version": 1,
        "phase": PHASE,
        "status": "READY_FOR_RUNTIME" if not blocked_reasons else "BLOCKED",
        "variants": variant_entries,
        "variant_order": list(variants),
        "single_process_per_adapter_required": True,
        "execute_train_required_for_pass": True,
        "dpo_authorized": False,
        "package_authorized": False,
        "submission_authorized": False,
        "leaderboard_claim": False,
        "blocked_reasons": blocked_reasons,
    }

    write_json(out_dir / "day2_lora_training_matrix_plan.json", plan)
    return plan


def run_subprocess_for_variant(
    *,
    out_dir: Path,
    model_path: str,
    variant_name: str,
    train_rows: Optional[int],
    eval_rows: Optional[int],
    max_steps: Optional[int],
    seed: int,
) -> int:
    cmd = [
        sys.executable,
        "-m",
        "kaggle_anti086.training.day2_lora_train_matrix",
        "--out-dir",
        str(out_dir),
        "--model-path",
        model_path,
        "--variant",
        variant_name,
        "--execute-train",
        "--seed",
        str(seed),
    ]

    if train_rows is not None:
        cmd += ["--train-rows", str(train_rows)]
    if eval_rows is not None:
        cmd += ["--eval-rows", str(eval_rows)]
    if max_steps is not None:
        cmd += ["--max-steps", str(max_steps)]

    print("RUNNING:", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def aggregate_reports(out_dir: Path, require_adapter_files: bool = True) -> Dict[str, Any]:
    from kaggle_anti086.training.day2_lora_train_report import build_matrix_report, read_json as read_report_json

    report_paths = sorted(out_dir.glob("day2_train_report_A*.json"))
    reports = [read_report_json(p) for p in report_paths]
    matrix = build_matrix_report(
        reports,
        out_dir=out_dir,
        require_adapter_files=require_adapter_files,
    )
    write_json(out_dir / "day2_lora_training_matrix_report.json", matrix)
    return matrix


def parse_variant_list(value: str) -> List[str]:
    if value.strip().lower() in {"first_wave", "all"}:
        return list(FIRST_WAVE_ORDER)
    return [x.strip() for x in value.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="artifacts/sprint11")
    ap.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--variants", default="first_wave")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--execute-train", action="store_true")
    ap.add_argument("--launch-subprocesses", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--train-rows", type=int, default=None)
    ap.add_argument("--eval-rows", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        matrix = aggregate_reports(out_dir)
        print(json.dumps(matrix, indent=2, sort_keys=True))
        return 0 if matrix["status"] == "PASS" else 2

    variants = [args.variant] if args.variant else parse_variant_list(args.variants)

    plan = prepare_plan(out_dir, variants)
    print(json.dumps(plan, indent=2, sort_keys=True))

    if plan["status"] == "BLOCKED":
        return 2

    if args.prepare_only:
        return 0

    if args.launch_subprocesses:
        failures: List[str] = []
        for name in variants:
            code = run_subprocess_for_variant(
                out_dir=out_dir,
                model_path=args.model_path,
                variant_name=name,
                train_rows=args.train_rows,
                eval_rows=args.eval_rows,
                max_steps=args.max_steps,
                seed=args.seed,
            )
            if code != 0:
                failures.append(f"{name}:exit_{code}")

        matrix = aggregate_reports(out_dir)
        print(json.dumps(matrix, indent=2, sort_keys=True))

        if failures:
            print("SUBPROCESS_FAILURES:", failures, file=sys.stderr)
            return 2

        return 0 if matrix["status"] == "PASS" else 2

    if args.execute_train:
        if not args.variant:
            raise SystemExit("--execute-train requires --variant for clean single-adapter process")

        report = train_one_variant(
            out_dir=out_dir,
            model_path=args.model_path,
            variant_name=args.variant,
            train_rows_override=args.train_rows,
            eval_rows_override=args.eval_rows,
            max_steps_override=args.max_steps,
            seed=args.seed,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["status"] == "PASS" else 2

    print(
        "No training executed. Use --execute-train with --variant, "
        "or --launch-subprocesses for isolated sequential launches.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
