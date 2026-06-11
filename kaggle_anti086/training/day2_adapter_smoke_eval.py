from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


BOX_RE = re.compile(r"\\boxed\{([^{}]*)\}")
SMOKE_VARIANTS_REQUIRED = [
    "smoke_direct_qv_r16",
    "smoke_mixed_qv_r16",
    "smoke_short_trace_qv_r16",
]


DIAGNOSTIC_PROMPTS = [
    {
        "id": "diag_format_1",
        "prompt": "Compute 17 + 25. Return only the final answer in exactly one boxed expression.",
        "expected_answer": "42",
    },
    {
        "id": "diag_format_2",
        "prompt": "What is 9 times 8? Return only the final answer in exactly one boxed expression.",
        "expected_answer": "72",
    },
    {
        "id": "diag_no_abstain",
        "prompt": "Convert 12 centimeters to millimeters. Return only the final answer in exactly one boxed expression.",
        "expected_answer": "120",
    },
]


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


def norm_ws(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "")).strip()


def boxed_values(text: str) -> List[str]:
    return [norm_ws(x) for x in BOX_RE.findall(text or "")]


def ends_with_final_box(text: str) -> bool:
    text = norm_ws(text)
    matches = list(BOX_RE.finditer(text))
    return bool(matches) and matches[-1].end() == len(text)


def validate_generation(text: str) -> List[str]:
    problems: List[str] = []
    text = norm_ws(text)
    if not text:
        problems.append("empty_output")
    if "ABSTAIN" in text.upper():
        problems.append("abstain_output")
    boxes = boxed_values(text)
    if len(boxes) != 1:
        problems.append(f"box_count_{len(boxes)}")
    elif not ends_with_final_box(text):
        problems.append("text_after_final_box")
    return problems


def validate_adapter_dir(path: Path, expected_rank: Optional[int] = None) -> Dict[str, Any]:
    blocked: List[str] = []
    if not path.exists():
        blocked.append("adapter_dir_missing")

    config_path = path / "adapter_config.json"
    if not config_path.exists():
        blocked.append("adapter_config_missing")
        cfg = {}
    else:
        try:
            cfg = read_json(config_path)
        except Exception as exc:
            blocked.append(f"adapter_config_broken:{repr(exc)}")
            cfg = {}

    rank = cfg.get("r") or cfg.get("rank")
    try:
        rank_int = int(rank) if rank is not None else None
    except Exception:
        rank_int = None

    if rank_int is None:
        blocked.append("rank_missing")
    elif rank_int > 32:
        blocked.append(f"rank_gt_32:{rank_int}")

    if expected_rank is not None and rank_int is not None and rank_int != expected_rank:
        blocked.append(f"rank_mismatch:{rank_int}!={expected_rank}")

    target_modules = cfg.get("target_modules")
    if not target_modules:
        blocked.append("target_modules_missing")

    # PEFT adapters usually have adapter_model.safetensors or adapter_model.bin.
    model_files = [
        path / "adapter_model.safetensors",
        path / "adapter_model.bin",
    ]
    if not any(p.exists() for p in model_files):
        blocked.append("adapter_weight_file_missing")

    return {
        "path": str(path),
        "status": "PASS" if not blocked else "FAIL",
        "adapter_config_exists": config_path.exists(),
        "adapter_weight_file_exists": any(p.exists() for p in model_files),
        "rank": rank_int,
        "target_modules": target_modules,
        "blocked_reasons": blocked,
    }


def check_runtime_imports() -> Dict[str, str]:
    modules = {}
    for name in ["torch", "transformers", "peft"]:
        try:
            __import__(name)
            modules[name] = "OK"
        except Exception as exc:
            modules[name] = f"MISSING:{repr(exc)}"
    return modules


def run_real_generation_delta(
    out_dir: Path,
    base_model_path: str,
    adapter_paths: Dict[str, Path],
    max_new_tokens: int = 128,
) -> Dict[str, Any]:
    imports = check_runtime_imports()
    missing = [k for k, v in imports.items() if v != "OK"]
    if missing:
        return {
            "status": "BLOCKED_MISSING_RUNTIME",
            "runtime_imports": imports,
            "blocked_reasons": [f"missing_runtime_module:{m}" for m in missing],
            "generation_reports": {},
        }

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        return {
            "status": "BLOCKED_NO_CUDA",
            "runtime_imports": imports,
            "blocked_reasons": ["cuda_not_available"],
            "generation_reports": {},
        }

    if not Path(base_model_path).exists():
        return {
            "status": "BLOCKED_MISSING_BASE_MODEL",
            "runtime_imports": imports,
            "blocked_reasons": [f"base_model_path_missing:{base_model_path}"],
            "generation_reports": {},
        }

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    base_model.eval()

    def generate(model, prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    base_outputs = {}
    for item in DIAGNOSTIC_PROMPTS:
        base_outputs[item["id"]] = generate(base_model, item["prompt"])

    generation_reports: Dict[str, Any] = {}
    global_blocked: List[str] = []

    for adapter_name, adapter_path in adapter_paths.items():
        if not adapter_path.exists():
            generation_reports[adapter_name] = {
                "status": "FAIL",
                "blocked_reasons": ["adapter_path_missing"],
            }
            global_blocked.append(f"{adapter_name}:adapter_path_missing")
            continue

        adapter_model = PeftModel.from_pretrained(base_model, str(adapter_path))
        adapter_model.eval()

        comparisons = []
        delta_count = 0
        format_problem_count = 0
        abstain_count = 0

        for item in DIAGNOSTIC_PROMPTS:
            adapter_output = generate(adapter_model, item["prompt"])
            base_output = base_outputs[item["id"]]
            delta = norm_ws(adapter_output) != norm_ws(base_output)
            problems = validate_generation(adapter_output)

            if delta:
                delta_count += 1
            if problems:
                format_problem_count += 1
            if "ABSTAIN" in adapter_output.upper():
                abstain_count += 1

            comparisons.append({
                "id": item["id"],
                "prompt": item["prompt"],
                "base_output": base_output,
                "adapter_output": adapter_output,
                "output_differs_from_base": delta,
                "adapter_generation_problems": problems,
            })

        status = "PASS" if delta_count > 0 and format_problem_count == 0 and abstain_count == 0 else "FAIL"
        if status != "PASS":
            global_blocked.append(f"{adapter_name}:delta_or_format_smoke_failed")

        generation_reports[adapter_name] = {
            "status": status,
            "delta_count": delta_count,
            "format_problem_count": format_problem_count,
            "abstain_count": abstain_count,
            "diagnostic_prompt_count": len(DIAGNOSTIC_PROMPTS),
            "comparisons": comparisons,
        }

        del adapter_model

    return {
        "status": "PASS" if not global_blocked else "FAIL",
        "runtime_imports": imports,
        "blocked_reasons": global_blocked,
        "base_outputs": base_outputs,
        "generation_reports": generation_reports,
    }


def load_smoke_train_report(out_dir: Path) -> Dict[str, Any]:
    path = out_dir / "day2_smoke_train_report.json"
    if not path.exists():
        return {
            "status": "MISSING",
            "blocked_reasons": ["day2_smoke_train_report_missing"],
        }
    return read_json(path)


def build_adapter_paths_from_train_report(out_dir: Path, train_report: Dict[str, Any]) -> Dict[str, Path]:
    adapter_paths: Dict[str, Path] = {}
    adapter_reports = (
        train_report
        .get("training_result", {})
        .get("adapter_reports", {})
    )
    for name in SMOKE_VARIANTS_REQUIRED:
        p = adapter_reports.get(name, {}).get("adapter_path")
        if p:
            adapter_paths[name] = Path(p)
        else:
            adapter_paths[name] = out_dir / "adapters" / name
    return adapter_paths


def run(
    out_dir: Path,
    base_model_path: Optional[str] = None,
    execute_inference: bool = False,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    train_report = load_smoke_train_report(out_dir)
    adapter_paths = build_adapter_paths_from_train_report(out_dir, train_report)

    adapter_dir_reports = {}
    blocked: List[str] = []

    for name in SMOKE_VARIANTS_REQUIRED:
        report = validate_adapter_dir(adapter_paths[name], expected_rank=16)
        adapter_dir_reports[name] = report
        if report["status"] != "PASS":
            blocked.append(f"{name}:adapter_dir_validation_failed")
            blocked.extend([f"{name}:{r}" for r in report["blocked_reasons"]])

    if train_report.get("status") != "PASS":
        blocked.append(f"smoke_train_report_not_pass:{train_report.get('status')}")

    if execute_inference:
        if not base_model_path:
            generation_delta = {
                "status": "BLOCKED_MISSING_BASE_MODEL_ARG",
                "blocked_reasons": ["--base-model-path required with --execute-inference"],
                "generation_reports": {},
            }
        elif blocked:
            generation_delta = {
                "status": "BLOCKED_ADAPTER_DIR_OR_TRAIN_REPORT",
                "blocked_reasons": blocked,
                "generation_reports": {},
            }
        else:
            generation_delta = run_real_generation_delta(
                out_dir=out_dir,
                base_model_path=base_model_path,
                adapter_paths=adapter_paths,
            )
    else:
        generation_delta = {
            "status": "READY_FOR_RUNTIME",
            "blocked_reasons": ["execute_inference_not_requested"],
            "generation_reports": {},
        }

    final_blocked = list(blocked)
    if generation_delta.get("status") != "PASS":
        final_blocked.append(f"generation_delta_not_pass:{generation_delta.get('status')}")
        final_blocked.extend(generation_delta.get("blocked_reasons", []))

    if train_report.get("status") == "PASS" and generation_delta.get("status") == "PASS" and not final_blocked:
        status = "PASS"
    elif train_report.get("status") == "PASS" and not execute_inference and not blocked:
        status = "READY_FOR_RUNTIME"
    else:
        status = "FAIL"

    report = {
        "schema_version": 1,
        "created_by": "PHASE8_ADAPTER_DELTA_SMOKE_EVAL",
        "status": status,
        "smoke_train_report_status": train_report.get("status"),
        "execute_inference_requested": execute_inference,
        "base_model_path": base_model_path,
        "adapter_dir_reports": adapter_dir_reports,
        "generation_delta": generation_delta,
        "adapter_load_pass": all(r["status"] == "PASS" for r in adapter_dir_reports.values()),
        "output_delta_pass": generation_delta.get("status") == "PASS",
        "format_smoke_pass": (
            generation_delta.get("status") == "PASS"
            and all(
                r.get("format_problem_count", 999) == 0
                for r in generation_delta.get("generation_reports", {}).values()
            )
        ),
        "abstain_output_count": sum(
            int(r.get("abstain_count", 0) or 0)
            for r in generation_delta.get("generation_reports", {}).values()
        ),
        "safe_for_full_training_matrix": status == "PASS",
        "training_authorized": True,
        "package_authorized": False,
        "submission_authorized": False,
        "leaderboard_claim": False,
        "no_0_93_evidence": True,
        "no_0_95_evidence": True,
        "next_phase": "PHASE9_FIRST_WAVE_SFT_ADAPTER_MATRIX" if status == "PASS" else "FIX_PHASE8_SMOKE_BEFORE_FULL_TRAINING",
        "blocked_reasons": final_blocked,
    }

    write_json(out_dir / "day2_adapter_delta_smoke_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="artifacts/sprint11")
    parser.add_argument("--base-model-path", default=None)
    parser.add_argument("--execute-inference", action="store_true")
    args = parser.parse_args()

    report = run(
        out_dir=Path(args.out_dir),
        base_model_path=args.base_model_path,
        execute_inference=args.execute_inference,
    )

    summary = {
        "status": report["status"],
        "smoke_train_report_status": report["smoke_train_report_status"],
        "adapter_load_pass": report["adapter_load_pass"],
        "output_delta_pass": report["output_delta_pass"],
        "format_smoke_pass": report["format_smoke_pass"],
        "safe_for_full_training_matrix": report["safe_for_full_training_matrix"],
        "package_authorized": report["package_authorized"],
        "submission_authorized": report["submission_authorized"],
        "leaderboard_claim": report["leaderboard_claim"],
        "next_phase": report["next_phase"],
        "blocked_reasons": report["blocked_reasons"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
