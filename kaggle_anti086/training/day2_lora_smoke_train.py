from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


BOX_RE = re.compile(r"\\boxed\{([^{}]*)\}")


SMOKE_VARIANTS_REQUIRED = [
    "smoke_direct_qv_r16",
    "smoke_mixed_qv_r16",
    "smoke_short_trace_qv_r16",
]

DATASET_BY_VARIANT = {
    "smoke_direct_qv_r16": "day1x_sft_direct.jsonl",
    "smoke_mixed_qv_r16": "day1x_sft_mixed_curriculum.jsonl",
    "smoke_short_trace_qv_r16": "day1x_sft_short_trace.jsonl",
}

DEFAULT_SMOKE_CONFIGS = {
    "smoke_direct_qv_r16": {
        "adapter_name": "smoke_direct_qv_r16",
        "dataset_variant": "direct",
        "target_modules": ["q_proj", "v_proj"],
        "rank": 16,
        "alpha": 32,
        "dropout": 0.05,
        "learning_rate": 2e-5,
        "max_seq_len": 1024,
        "train_rows_cap": 2000,
        "max_steps": 30,
    },
    "smoke_mixed_qv_r16": {
        "adapter_name": "smoke_mixed_qv_r16",
        "dataset_variant": "mixed_curriculum",
        "target_modules": ["q_proj", "v_proj"],
        "rank": 16,
        "alpha": 32,
        "dropout": 0.05,
        "learning_rate": 2e-5,
        "max_seq_len": 1024,
        "train_rows_cap": 2000,
        "max_steps": 30,
    },
    "smoke_short_trace_qv_r16": {
        "adapter_name": "smoke_short_trace_qv_r16",
        "dataset_variant": "short_trace",
        "target_modules": ["q_proj", "v_proj"],
        "rank": 16,
        "alpha": 32,
        "dropout": 0.05,
        "learning_rate": 2e-5,
        "max_seq_len": 1024,
        "train_rows_cap": 2000,
        "max_steps": 30,
    },
}


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"JSON is not object: {path}")
    return obj


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                yield {
                    "__parse_error__": True,
                    "__line_no__": line_no,
                    "__error__": repr(exc),
                }
                continue
            if isinstance(row, dict):
                yield row
            else:
                yield {
                    "__parse_error__": True,
                    "__line_no__": line_no,
                    "__error__": "jsonl row is not object",
                }


def norm_ws(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "")).strip()


def boxed_values(text: str) -> List[str]:
    return [norm_ws(x) for x in BOX_RE.findall(text or "")]


def ends_with_final_box(text: str) -> bool:
    text = norm_ws(text)
    matches = list(BOX_RE.finditer(text))
    return bool(matches) and matches[-1].end() == len(text)


def get_user_prompt(row: Dict[str, Any]) -> str:
    if "prompt" in row:
        return norm_ws(row.get("prompt"))
    msgs = row.get("messages")
    if isinstance(msgs, list):
        for msg in msgs:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return norm_ws(msg.get("content"))
    return ""


def get_assistant_target(row: Dict[str, Any]) -> str:
    if "target_text" in row:
        return norm_ws(row.get("target_text"))
    msgs = row.get("messages")
    if isinstance(msgs, list):
        for msg in msgs:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return norm_ws(msg.get("content"))
    return ""


def get_answer(row: Dict[str, Any]) -> str:
    return norm_ws(row.get("answer"))


def validate_sft_row(row: Dict[str, Any]) -> List[str]:
    problems: List[str] = []
    if row.get("__parse_error__"):
        return [f"parse_error:{row.get('__error__')}"]

    prompt = get_user_prompt(row)
    target = get_assistant_target(row)
    answer = get_answer(row)

    if not prompt:
        problems.append("empty_prompt")
    if not target:
        problems.append("empty_target")
    if not answer:
        problems.append("empty_answer")

    boxes = boxed_values(target)
    if len(boxes) != 1:
        problems.append(f"box_count_{len(boxes)}")
    elif boxes[0] != answer:
        problems.append("boxed_answer_mismatch")

    if boxes and not ends_with_final_box(target):
        problems.append("text_after_final_box")

    if "ABSTAIN" in target.upper():
        problems.append("abstain_in_target")

    if row.get("verification_status") not in (None, "PASS"):
        problems.append("verification_status_not_pass")

    try:
        ambiguity = int(row.get("ambiguity_count", 0) or 0)
    except Exception:
        ambiguity = 999
    if ambiguity != 0:
        problems.append("ambiguity_nonzero")

    return problems


def make_training_text(row: Dict[str, Any]) -> Dict[str, Any]:
    prompt = get_user_prompt(row)
    target = get_assistant_target(row)

    # Stable training text. This is deliberately simple and extractor-safe.
    text = (
        "You must solve the problem and put only the final answer in exactly one "
        "\\boxed{} at the end.\n\n"
        f"Problem:\n{prompt}\n\n"
        f"Answer:\n{target}"
    )

    return {
        "id": row.get("id"),
        "family": row.get("family"),
        "prompt_style": row.get("prompt_style"),
        "difficulty": row.get("difficulty"),
        "answer": row.get("answer"),
        "prompt": prompt,
        "target_text": target,
        "text": text,
    }


def materialize_smoke_dataset(
    out_dir: Path,
    source_path: Path,
    adapter_name: str,
    train_rows_cap: int,
) -> Dict[str, Any]:
    accepted: List[Dict[str, Any]] = []
    rejected_examples: List[Dict[str, Any]] = []
    scanned = 0
    problem_count = 0

    for row in iter_jsonl(source_path):
        scanned += 1
        problems = validate_sft_row(row)
        if problems:
            problem_count += 1
            if len(rejected_examples) < 10:
                rejected_examples.append({
                    "id": row.get("id"),
                    "family": row.get("family"),
                    "problems": problems,
                })
            continue

        accepted.append(make_training_text(row))
        if len(accepted) >= train_rows_cap:
            break

    path = out_dir / "day2_smoke_datasets" / f"{adapter_name}.jsonl"
    write_jsonl(path, accepted)

    status = "PASS" if len(accepted) == train_rows_cap and problem_count == 0 else "WARN"
    if len(accepted) == 0:
        status = "FAIL"

    return {
        "adapter_name": adapter_name,
        "source_path": str(source_path),
        "smoke_dataset_path": str(path),
        "status": status,
        "scanned_rows": scanned,
        "accepted_rows": len(accepted),
        "problem_count": problem_count,
        "train_rows_cap": train_rows_cap,
        "examples_rejected": rejected_examples,
    }


def validate_phase7(out_dir: Path) -> Dict[str, Any]:
    required = [
        out_dir / "day2_lora_preflight_report.json",
        out_dir / "day2_lora_variant_matrix.json",
        out_dir / "day2_training_config_manifest.json",
    ]
    blocked: List[str] = []
    reports: Dict[str, Any] = {}

    for path in required:
        if not path.exists():
            blocked.append(f"missing_phase7_file:{path.name}")
            continue
        try:
            reports[path.name] = read_json(path)
        except Exception as exc:
            blocked.append(f"broken_phase7_file:{path.name}:{repr(exc)}")

    for name, report in reports.items():
        if report.get("status") != "PASS":
            blocked.append(f"phase7_report_not_pass:{name}:{report.get('status')}")
        if report.get("training_authorized") is not True:
            blocked.append(f"training_not_authorized:{name}")
        if report.get("package_authorized") is True:
            blocked.append(f"package_authorized_true:{name}")
        if report.get("submission_authorized") is True:
            blocked.append(f"submission_authorized_true:{name}")
        if report.get("leaderboard_claim") is True:
            blocked.append(f"leaderboard_claim_true:{name}")

    return {
        "status": "PASS" if not blocked else "FAIL",
        "blocked_reasons": blocked,
        "reports_loaded": sorted(reports.keys()),
    }


def get_smoke_configs_from_phase7(out_dir: Path) -> Dict[str, Dict[str, Any]]:
    matrix_path = out_dir / "day2_lora_variant_matrix.json"
    configs: Dict[str, Dict[str, Any]] = {}

    if matrix_path.exists():
        matrix = read_json(matrix_path)
        for variant in matrix.get("variants", []):
            name = variant.get("adapter_name")
            if name in SMOKE_VARIANTS_REQUIRED:
                configs[name] = dict(variant)

    # Add required missing smoke config, especially smoke_short_trace_qv_r16.
    for name in SMOKE_VARIANTS_REQUIRED:
        if name not in configs:
            configs[name] = dict(DEFAULT_SMOKE_CONFIGS[name])

    # Hard safety override for smoke.
    for name, cfg in configs.items():
        cfg["adapter_name"] = name
        cfg["rank"] = int(cfg.get("rank", 16))
        cfg["target_modules"] = list(cfg.get("target_modules", ["q_proj", "v_proj"]))
        cfg["train_rows_cap"] = int(cfg.get("train_rows_cap", DEFAULT_SMOKE_CONFIGS[name]["train_rows_cap"]))
        cfg["max_steps"] = int(cfg.get("max_steps", DEFAULT_SMOKE_CONFIGS[name]["max_steps"]))
        cfg["safe_for_smoke"] = (
            cfg["rank"] <= 32
            and cfg["target_modules"]
            and not bool(cfg.get("uses_dpo"))
        )

    return configs


def build_runtime_plan(
    out_dir: Path,
    base_model_path: Optional[str],
    execute_train: bool,
    smoke_datasets: Dict[str, Any],
    smoke_configs: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "created_by": "DAY2_SMOKE_RUNTIME_PLAN",
        "execute_train_requested": execute_train,
        "base_model_path": base_model_path,
        "requires_gpu": True,
        "requires_transformers": True,
        "requires_peft": True,
        "requires_torch": True,
        "smoke_variants": list(SMOKE_VARIANTS_REQUIRED),
        "smoke_datasets": smoke_datasets,
        "smoke_configs": smoke_configs,
        "local_windows_note": (
            "If this is run on Windows without the Nemotron model and GPU stack, "
            "the script must stop at READY_FOR_RUNTIME and must not fake adapter training."
        ),
        "next_if_ready": (
            "Run this script in Kaggle/GPU runtime with --execute-train "
            "--base-model-path <Nemotron model path>."
        ),
    }


def check_runtime_imports() -> Dict[str, Any]:
    modules = {}
    for name in ["torch", "transformers", "peft", "datasets"]:
        try:
            __import__(name)
            modules[name] = "OK"
        except Exception as exc:
            modules[name] = f"MISSING:{repr(exc)}"
    return modules


def run_real_training(
    out_dir: Path,
    smoke_configs: Dict[str, Dict[str, Any]],
    smoke_dataset_reports: Dict[str, Any],
    base_model_path: str,
) -> Dict[str, Any]:
    """
    Real training path. This is intentionally conservative.

    If required libraries/model/GPU are missing, this returns FAIL/BLOCKED.
    It must never create fake adapter directories.
    """
    imports = check_runtime_imports()
    missing = [k for k, v in imports.items() if not v == "OK"]

    if missing:
        return {
            "status": "BLOCKED_MISSING_RUNTIME",
            "runtime_imports": imports,
            "blocked_reasons": [f"missing_runtime_module:{m}" for m in missing],
            "adapter_reports": {},
        }

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    if not torch.cuda.is_available():
        return {
            "status": "BLOCKED_NO_CUDA",
            "runtime_imports": imports,
            "blocked_reasons": ["cuda_not_available"],
            "adapter_reports": {},
        }

    if not Path(base_model_path).exists():
        return {
            "status": "BLOCKED_MISSING_BASE_MODEL",
            "runtime_imports": imports,
            "blocked_reasons": [f"base_model_path_missing:{base_model_path}"],
            "adapter_reports": {},
        }

    adapter_reports: Dict[str, Any] = {}
    blocked: List[str] = []

    for name in SMOKE_VARIANTS_REQUIRED:
        cfg = smoke_configs[name]
        ds_report = smoke_dataset_reports[name]
        ds_path = Path(ds_report["smoke_dataset_path"])

        if ds_report.get("accepted_rows", 0) <= 0:
            adapter_reports[name] = {
                "status": "FAIL",
                "blocked_reasons": ["empty_smoke_dataset"],
            }
            blocked.append(f"{name}:empty_smoke_dataset")
            continue

        rows = list(iter_jsonl(ds_path))
        texts = [r["text"] for r in rows]

        tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        lora_cfg = LoraConfig(
            r=int(cfg["rank"]),
            lora_alpha=int(cfg.get("alpha", 32)),
            target_modules=list(cfg["target_modules"]),
            lora_dropout=float(cfg.get("dropout", 0.05)),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)

        def tokenize(batch):
            return tokenizer(
                batch["text"],
                truncation=True,
                max_length=int(cfg.get("max_seq_len", 1024)),
                padding=False,
            )

        dataset = Dataset.from_dict({"text": texts})
        tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

        adapter_dir = out_dir / "adapters" / name
        args = TrainingArguments(
            output_dir=str(out_dir / "tmp_trainer" / name),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=float(cfg.get("learning_rate", 2e-5)),
            max_steps=int(cfg.get("max_steps", 30)),
            logging_steps=1,
            save_steps=int(cfg.get("max_steps", 30)),
            bf16=True,
            fp16=False,
            report_to=[],
            remove_unused_columns=False,
            gradient_checkpointing=True,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        )

        train_started = time.time()
        result = trainer.train()
        train_seconds = time.time() - train_started

        losses = []
        for item in trainer.state.log_history:
            if "loss" in item:
                try:
                    losses.append(float(item["loss"]))
                except Exception:
                    pass

        finite_losses = all(math.isfinite(x) for x in losses) if losses else False
        loss_start = losses[0] if losses else None
        loss_end = losses[-1] if losses else None

        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

        adapter_config = adapter_dir / "adapter_config.json"
        status = "PASS" if adapter_config.exists() and finite_losses else "FAIL"
        if status != "PASS":
            blocked.append(f"{name}:adapter_missing_or_loss_not_finite")

        adapter_reports[name] = {
            "status": status,
            "adapter_name": name,
            "adapter_path": str(adapter_dir),
            "adapter_config_exists": adapter_config.exists(),
            "rank": int(cfg["rank"]),
            "target_modules": cfg["target_modules"],
            "train_rows": len(texts),
            "max_steps": int(cfg.get("max_steps", 30)),
            "train_seconds": train_seconds,
            "loss_values_seen": len(losses),
            "train_loss_start": loss_start,
            "train_loss_end": loss_end,
            "loss_finite": finite_losses,
            "nan_count": sum(1 for x in losses if math.isnan(x)),
            "inf_count": sum(1 for x in losses if math.isinf(x)),
            "safe_for_smoke_eval": status == "PASS",
        }

        del trainer
        del model
        torch.cuda.empty_cache()

    return {
        "status": "PASS" if not blocked else "FAIL",
        "runtime_imports": imports,
        "blocked_reasons": blocked,
        "adapter_reports": adapter_reports,
    }


def run(
    out_dir: Path,
    base_model_path: Optional[str] = None,
    execute_train: bool = False,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    phase7 = validate_phase7(out_dir)
    smoke_configs = get_smoke_configs_from_phase7(out_dir)

    smoke_dataset_reports: Dict[str, Any] = {}
    blocked: List[str] = []

    for name in SMOKE_VARIANTS_REQUIRED:
        cfg = smoke_configs[name]
        if not cfg.get("safe_for_smoke"):
            blocked.append(f"unsafe_smoke_config:{name}")
            continue

        dataset_file = DATASET_BY_VARIANT[name]
        source_path = out_dir / dataset_file
        if not source_path.exists():
            blocked.append(f"missing_source_dataset:{dataset_file}")
            continue

        report = materialize_smoke_dataset(
            out_dir=out_dir,
            source_path=source_path,
            adapter_name=name,
            train_rows_cap=int(cfg["train_rows_cap"]),
        )
        smoke_dataset_reports[name] = report
        if report["status"] == "FAIL":
            blocked.append(f"smoke_dataset_failed:{name}")

    runtime_plan = build_runtime_plan(
        out_dir=out_dir,
        base_model_path=base_model_path,
        execute_train=execute_train,
        smoke_datasets=smoke_dataset_reports,
        smoke_configs=smoke_configs,
    )
    write_json(out_dir / "day2_smoke_runtime_plan.json", runtime_plan)

    training_result: Dict[str, Any]
    if execute_train:
        if not base_model_path:
            training_result = {
                "status": "BLOCKED_MISSING_BASE_MODEL_ARG",
                "blocked_reasons": ["--base-model-path required when --execute-train is set"],
                "adapter_reports": {},
            }
        elif phase7["status"] != "PASS":
            training_result = {
                "status": "BLOCKED_PHASE7_NOT_PASS",
                "blocked_reasons": phase7["blocked_reasons"],
                "adapter_reports": {},
            }
        elif blocked:
            training_result = {
                "status": "BLOCKED_SMOKE_DATASET_OR_CONFIG",
                "blocked_reasons": blocked,
                "adapter_reports": {},
            }
        else:
            training_result = run_real_training(
                out_dir=out_dir,
                smoke_configs=smoke_configs,
                smoke_dataset_reports=smoke_dataset_reports,
                base_model_path=base_model_path,
            )
    else:
        training_result = {
            "status": "READY_FOR_RUNTIME",
            "blocked_reasons": ["execute_train_not_requested"],
            "adapter_reports": {},
        }

    all_adapter_pass = (
        training_result.get("status") == "PASS"
        and all(
            training_result.get("adapter_reports", {}).get(name, {}).get("status") == "PASS"
            for name in SMOKE_VARIANTS_REQUIRED
        )
    )

    final_blocked = []
    if phase7["status"] != "PASS":
        final_blocked.append("phase7_not_pass")
        final_blocked.extend(phase7["blocked_reasons"])
    final_blocked.extend(blocked)
    if training_result.get("status") != "PASS":
        final_blocked.append(f"training_not_pass:{training_result.get('status')}")
        final_blocked.extend(training_result.get("blocked_reasons", []))

    if all_adapter_pass:
        status = "PASS"
    elif not execute_train and not blocked and phase7["status"] == "PASS":
        status = "READY_FOR_RUNTIME"
    else:
        status = "FAIL"

    report = {
        "schema_version": 1,
        "created_by": "PHASE8_ADAPTER_SMOKE_TRAINING",
        "status": status,
        "phase7_status": phase7["status"],
        "smoke_variants_required": SMOKE_VARIANTS_REQUIRED,
        "smoke_configs": smoke_configs,
        "smoke_dataset_reports": smoke_dataset_reports,
        "runtime_plan_file": str(out_dir / "day2_smoke_runtime_plan.json"),
        "execute_train_requested": execute_train,
        "base_model_path": base_model_path,
        "training_result": training_result,
        "all_smoke_runs_complete": all_adapter_pass,
        "loss_finite": all(
            r.get("loss_finite") is True
            for r in training_result.get("adapter_reports", {}).values()
        ) if training_result.get("adapter_reports") else False,
        "training_authorized": phase7["status"] == "PASS",
        "safe_for_adapter_delta_eval": all_adapter_pass,
        "package_authorized": False,
        "submission_authorized": False,
        "leaderboard_claim": False,
        "no_0_93_evidence": True,
        "no_0_95_evidence": True,
        "next_phase": "PHASE8_ADAPTER_DELTA_SMOKE_EVAL" if all_adapter_pass else "RUN_PHASE8_SMOKE_TRAINING_IN_GPU_RUNTIME",
        "blocked_reasons": final_blocked,
    }

    write_json(out_dir / "day2_smoke_train_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="artifacts/sprint11")
    parser.add_argument("--base-model-path", default=None)
    parser.add_argument("--execute-train", action="store_true")
    args = parser.parse_args()

    report = run(
        out_dir=Path(args.out_dir),
        base_model_path=args.base_model_path,
        execute_train=args.execute_train,
    )

    summary = {
        "status": report["status"],
        "phase7_status": report["phase7_status"],
        "execute_train_requested": report["execute_train_requested"],
        "all_smoke_runs_complete": report["all_smoke_runs_complete"],
        "safe_for_adapter_delta_eval": report["safe_for_adapter_delta_eval"],
        "training_authorized": report["training_authorized"],
        "package_authorized": report["package_authorized"],
        "submission_authorized": report["submission_authorized"],
        "leaderboard_claim": report["leaderboard_claim"],
        "next_phase": report["next_phase"],
        "blocked_reasons": report["blocked_reasons"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
