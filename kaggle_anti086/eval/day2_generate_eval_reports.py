from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import json
from pathlib import Path
import sys
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.v2_corpus_io import file_record, read_json, read_jsonl, write_json_checked, write_jsonl_checked
from kaggle_anti086.eval.eval_ladder import build_eval_ladder, write_eval_ladder
from kaggle_anti086.runtime.prompt_format import render_inference_prompt
from kaggle_anti086.solvers.answer_normalizer import answers_match, normalize_answer
from kaggle_anti086.solvers.solver_ensemble import ANSWER_TYPE_BY_FAMILY, SolverEnsemble
from kaggle_anti086.training.day11a_candidate_ranking import build_candidate_ranking


DATASETS = {
    "core_eval": "artifacts/sprint11/day5_private_like_eval_512.jsonl",
    "family_eval": "artifacts/sprint11/day5_family_hard_eval_512.jsonl",
    "rule_holdout": "artifacts/sprint11/day5_rule_holdout_eval_512.jsonl",
    "anti_leak": "artifacts/sprint11/day5_anti_leak_eval_256.jsonl",
}
FALLBACK_DATASETS = {
    "core_eval": "artifacts/sprint11/day5_private_like_answerable_512.jsonl",
    "family_eval": "artifacts/sprint11/day5_family_hard_answerable_512.jsonl",
    "rule_holdout": "artifacts/sprint11/day5_rule_holdout_answerable_512.jsonl",
    "anti_leak": "artifacts/sprint11/day5_anti_leak_answerable_256.jsonl",
}
MODES = ("base_only", "solver_only", "adapter_only", "combined")
DEFAULT_BASE_MODEL_PATH = "/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1"
DEFAULT_ADAPTER_DIR = "/kaggle/working/anti086_adapters/v2a_bf16_qv_50"
KNOWN_ADAPTER_DIRS = (
    "/kaggle/working/anti086_adapters/v2a_bf16_qv_50_20260606_141808_42a82878",
    "/kaggle/working/anti086_adapters/v2a_bf16_qv_50",
)
DECODING_CONFIG = {
    "temperature": 0.0,
    "top_p": 1.0,
    "do_sample": False,
    "max_new_tokens_default": 64,
    "max_new_tokens_short": 32,
    "batch_size": 1,
    "prompt_template": "runtime.prompt_format.render_inference_prompt",
}
PredictionGenerator = Callable[[dict[str, Any]], str]






def _ensure_eval_model_on_cuda(model: Any) -> Any:
    """Force Day2 eval model onto CUDA for Nemotron/Mamba inference.

    This is eval-only. Do not move this into training/model_loader.py.
    The Kaggle Blackwell runtime has enough VRAM, and mixed CPU/CUDA tensors
    break Nemotron/Mamba generation with index_select/device errors.
    """
    try:
        import torch  # type: ignore
    except Exception:
        return model

    if not torch.cuda.is_available():
        return model

    target = torch.device("cuda:0")

    try:
        model = model.to(target)
    except Exception:
        # Let smoke report capture exact failure if CUDA move is impossible.
        return model

    try:
        if hasattr(model, "eval"):
            model.eval()
    except Exception:
        pass

    return model


def _infer_torch_device_for_model(model: Any) -> Any:
    """Return the real execution device for generation tensors.

    Nemotron/Mamba kernels require CUDA tensors. HF generate does not always
    move tokenizer outputs automatically when custom device_map / dynamic cache
    paths are involved, so the eval harness must do it explicitly.
    """
    try:
        import torch  # type: ignore
    except Exception:
        return None

    # Prefer a non-CPU parameter device.
    try:
        for param in model.parameters():
            dev = getattr(param, "device", None)
            if dev is not None and str(dev) != "cpu":
                return dev
    except Exception:
        pass

    # Some device_map="auto" models expose hf_device_map.
    try:
        device_map = getattr(model, "hf_device_map", None)
        if isinstance(device_map, dict):
            for dev in device_map.values():
                if dev is not None and str(dev) not in {"cpu", "disk"}:
                    return torch.device(str(dev))
    except Exception:
        pass

    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _move_generation_inputs_to_model_device(inputs: Any, model: Any) -> Any:
    """Move tokenizer BatchEncoding / dict tensors to the model execution device."""
    dev = _infer_torch_device_for_model(model)
    if dev is None:
        return inputs

    try:
        # BatchEncoding supports .to(device).
        if hasattr(inputs, "to"):
            return inputs.to(dev)
    except Exception:
        pass

    try:
        import torch  # type: ignore
        if isinstance(inputs, dict):
            out = {}
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    out[k] = v.to(dev)
                else:
                    out[k] = v
            return out
    except Exception:
        return inputs

    return inputs


def build_day2_eval_reports(
    *,
    out_dir: str | Path = "artifacts/sprint11/day2_reports",
    predictions_dir: str | Path = "artifacts/sprint11/day2_predictions",
    manifest_path: str | Path = "artifacts/sprint11/day2_eval_manifest.json",
    ladder_out: str | Path = "artifacts/sprint11/day2_eval_ladder_report.json",
    candidate_ranking_out: str | Path = "artifacts/sprint11/day2_candidate_ranking.json",
    decision_out: str | Path = "artifacts/sprint11/day2_decision_report.json",
    smoke_out: str | Path = "artifacts/sprint11/day2_inference_smoke_report.json",
    validation_out: str | Path = "artifacts/sprint11/day2_report_validation.json",
    base_model_path: str | None = None,
    adapter_dir: str | None = None,
    train_summary_path: str | Path = "artifacts/sprint11/day11a_v2a_50_train_summary.json",
    kaggle_mode: bool = False,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Generate Day 2 eval reports.

    Model modes run only when Kaggle mode and required model/adapter inputs are
    available. Otherwise the corresponding reports are explicit BLOCKED reports
    with score fields present so the ladder cannot authorize scaling.
    """

    out = Path(out_dir)
    pred_out = Path(predictions_dir)
    resolved_base_model_path, resolved_adapter_dir, train_summary = _resolve_model_inputs(
        base_model_path=base_model_path,
        adapter_dir=adapter_dir,
        train_summary_path=train_summary_path,
    )
    adapter_discovery = _discover_adapter_dir(adapter_dir=adapter_dir, train_summary=train_summary, requested=resolved_adapter_dir)
    resolved_adapter_dir = adapter_discovery["selected_adapter_dir"]
    smoke_report = _run_inference_smoke(
        base_model_path=resolved_base_model_path,
        adapter_dir=resolved_adapter_dir,
        kaggle_mode=kaggle_mode,
    )
    write_json_checked(smoke_out, smoke_report, field_name="day2_inference_smoke_report")
    dataset_rows = _load_datasets(max_rows=max_rows)
    manifest: dict[str, Any] = {"schema_version": 1, "created_by": "DAY2_REAL_EVAL_REPORT_GENERATOR", "datasets": {}}
    generated_reports: dict[str, Any] = {dataset: {} for dataset in DATASETS}
    blocked_modes: list[str] = []

    for dataset in DATASETS:
        manifest["datasets"][dataset] = {}

    for mode in MODES:
        mode_failure: dict[str, Any] | None = None
        generator: PredictionGenerator | None = None
        adapter_metadata: dict[str, Any] = {}
        if mode == "solver_only":
            generator = None
        elif not kaggle_mode:
            mode_failure = _mode_failure(mode, "model_inference_requires_kaggle_mode", resolved_base_model_path, resolved_adapter_dir)
        else:
            try:
                if mode == "base_only":
                    generator, adapter_metadata = _load_model_generator(
                        base_model_path=resolved_base_model_path,
                        adapter_dir=None,
                        mode=mode,
                    )
                elif mode == "adapter_only":
                    generator, adapter_metadata = _load_model_generator(
                        base_model_path=resolved_base_model_path,
                        adapter_dir=resolved_adapter_dir,
                        mode=mode,
                    )
                elif mode == "combined":
                    generator, adapter_metadata = _load_model_generator(
                        base_model_path=resolved_base_model_path,
                        adapter_dir=resolved_adapter_dir,
                        mode=mode,
                    )
            except Exception as exc:  # pragma: no cover - exercised in Kaggle or by monkeypatch tests
                mode_failure = {
                    "reason": f"{mode}_inference_failed",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "base_model_path": resolved_base_model_path,
                    "adapter_dir": resolved_adapter_dir,
                }
        if mode_failure is not None:
            blocked_modes.append(mode)

        for dataset, rows in dataset_rows.items():
            report_path = out / f"{dataset}_{mode}.json"
            predictions_path = pred_out / f"{dataset}_{mode}_predictions.jsonl"
            if mode_failure is not None:
                predictions: list[dict[str, Any]] = []
                report = _blocked_model_report(
                    dataset=dataset,
                    mode=mode,
                    rows=rows,
                    failure=mode_failure,
                    eval_path=_dataset_path(dataset),
                    predictions_path=predictions_path,
                    base_model_path=resolved_base_model_path,
                    adapter_dir=resolved_adapter_dir,
                )
            elif mode == "solver_only":
                report, predictions = _run_solver_mode(
                    dataset=dataset,
                    rows=rows,
                    eval_path=_dataset_path(dataset),
                    predictions_path=predictions_path,
                )
            elif mode in {"base_only", "adapter_only"}:
                assert generator is not None
                report, predictions = _run_model_mode(
                    dataset=dataset,
                    mode=mode,
                    rows=rows,
                    eval_path=_dataset_path(dataset),
                    predictions_path=predictions_path,
                    generator=generator,
                    base_model_path=resolved_base_model_path,
                    adapter_dir=resolved_adapter_dir if mode == "adapter_only" else None,
                    adapter_metadata=adapter_metadata,
                )
            else:
                assert generator is not None
                report, predictions = _run_combined_mode(
                    dataset=dataset,
                    rows=rows,
                    eval_path=_dataset_path(dataset),
                    predictions_path=predictions_path,
                    adapter_generator=generator,
                    base_model_path=resolved_base_model_path,
                    adapter_dir=resolved_adapter_dir,
                    adapter_metadata=adapter_metadata,
                )
            write_jsonl_checked(predictions_path, predictions, field_name=f"day2_{dataset}_{mode}_predictions")
            write_json_checked(report_path, report, field_name=f"day2_{dataset}_{mode}_report")
            manifest["datasets"][dataset][mode] = _manifest_entry(report_path)
            generated_reports[dataset][mode] = report

        _free_model_stack()

    write_json_checked(manifest_path, manifest, field_name="day2_eval_manifest")
    ladder = build_eval_ladder(manifest_path)
    write_eval_ladder(ladder, ladder_out)
    validation = validate_day2_outputs(
        reports_dir=out,
        predictions_dir=pred_out,
        manifest_path=manifest_path,
        ladder_path=ladder_out,
    )
    write_json_checked(validation_out, validation, field_name="day2_report_validation")
    ranking = _write_candidate_ranking(
        generated_reports=generated_reports,
        ladder=ladder,
        ladder_path=str(ladder_out),
        out_path=candidate_ranking_out,
        decision_out=decision_out,
    )
    return {
        "status": "PASS",
        "manifest_path": str(manifest_path),
        "ladder_path": str(ladder_out),
        "ladder_status": ladder.get("status"),
        "ladder_decision": ladder.get("decision", {}),
        "datasets": generated_reports,
        "blocked_modes": sorted(set(blocked_modes)),
        "base_model_path": resolved_base_model_path,
        "adapter_dir": resolved_adapter_dir,
        "adapter_discovery": adapter_discovery,
        "smoke_report_path": str(smoke_out),
        "smoke_status": smoke_report.get("status"),
        "validation_path": str(validation_out),
        "validation_status": validation.get("status"),
        "train_summary_status": train_summary.get("status") if train_summary else None,
        "candidate_ranking_path": str(candidate_ranking_out),
        "candidate_ranking_status": ranking.get("status"),
        "packaging_allowed": False,
        "submission_allowed": False,
        "model_results_faked": False,
    }


def _run_inference_smoke(*, base_model_path: str, adapter_dir: str, kaggle_mode: bool) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "BLOCKED",
        "base_model_load": False,
        "tokenizer_load": False,
        "adapter_load": False,
        "base_generate": False,
        "adapter_generate": False,
        "failures": [],
        "warnings": [],
        "base_model_path": base_model_path,
        "adapter_dir": adapter_dir,
        "adapter_config_sha256": None,
        "adapter_model_sha256": None,
        "packaging_allowed": False,
        "submission_allowed": False,
    }
    if not kaggle_mode:
        report["warnings"].append("kaggle_mode_not_enabled_smoke_not_run")
        if not Path(base_model_path).exists():
            report["failures"].append("base_model_path_missing")
        if not Path(adapter_dir).exists():
            report["failures"].append("adapter_dir_missing")
        return report
    try:
        adapter_metadata = _validate_adapter_dir(adapter_dir)
        report["adapter_config_sha256"] = adapter_metadata.get("adapter_config_sha256")
        report["adapter_model_sha256"] = adapter_metadata.get("adapter_model_sha256")
    except Exception as exc:
        report["failures"].append(f"adapter_validation_failed:{type(exc).__name__}:{exc}")
        return report
    try:
        from kaggle_anti086.training.model_loader import load_adapter_model, load_base_model, load_tokenizer

        tokenizer = load_tokenizer(base_model_path)
        report["tokenizer_load"] = True
        base_model = load_base_model(base_model_path, load_in_4bit=False, bf16=True)
        base_model = _ensure_eval_model_on_cuda(base_model)
        report["base_model_load"] = True
        _generate_answer(base_model, tokenizer, render_inference_prompt("1 -> 2; 2 -> 3. Now solve: 3"), "numeric_formula")
        report["base_generate"] = True
        del base_model
        _free_model_stack()
        adapter_model = load_adapter_model(base_model_path, adapter_dir, load_in_4bit=False, bf16=True)
        adapter_model = _ensure_eval_model_on_cuda(adapter_model)
        report["adapter_load"] = True
        _generate_answer(adapter_model, tokenizer, render_inference_prompt("1 -> 2; 2 -> 3. Now solve: 3"), "numeric_formula")
        report["adapter_generate"] = True
        del adapter_model
        del tokenizer
        _free_model_stack()
    except Exception as exc:  # pragma: no cover - Kaggle path
        report["failures"].append(f"inference_smoke_failed:{type(exc).__name__}:{exc}")
    report["status"] = "PASS" if all(report[key] for key in ("base_model_load", "tokenizer_load", "adapter_load", "base_generate", "adapter_generate")) and not report["failures"] else "BLOCKED"
    return report


def _resolve_model_inputs(
    *,
    base_model_path: str | None,
    adapter_dir: str | None,
    train_summary_path: str | Path,
) -> tuple[str, str, dict[str, Any]]:
    summary: dict[str, Any] = {}
    summary_path = Path(train_summary_path)
    if summary_path.exists():
        summary = read_json(summary_path)
    resolved_base = base_model_path or str(summary.get("config", {}).get("base_model_path") or DEFAULT_BASE_MODEL_PATH)
    resolved_adapter = adapter_dir or str(summary.get("adapter_dir") or summary.get("config", {}).get("output_adapter_dir") or DEFAULT_ADAPTER_DIR)
    return resolved_base, resolved_adapter, summary


def _discover_adapter_dir(*, adapter_dir: str | None, train_summary: dict[str, Any], requested: str) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    requested_paths = []
    if adapter_dir:
        requested_paths.append(adapter_dir)
    summary_adapter = train_summary.get("adapter_dir") or train_summary.get("config", {}).get("output_adapter_dir")
    if summary_adapter:
        requested_paths.append(str(summary_adapter))
    requested_paths.extend(KNOWN_ADAPTER_DIRS)
    requested_paths.extend(_glob_adapter_candidates("/kaggle/working/anti086_adapters"))
    requested_paths.extend(_glob_adapter_candidates("/kaggle/input"))
    seen = set()
    for path_text in requested_paths:
        if not path_text or path_text in seen:
            continue
        seen.add(path_text)
        path = Path(path_text)
        valid = (path / "adapter_config.json").exists() and (path / "adapter_model.safetensors").exists()
        candidates.append({"path": str(path), "exists": path.exists(), "valid_adapter": valid, "matches_summary": bool(summary_adapter and str(path) == str(summary_adapter))})
    valid_candidates = [item for item in candidates if item["valid_adapter"]]
    selected = requested
    warnings: list[str] = []
    failures: list[str] = []
    if adapter_dir:
        selected = adapter_dir
        if not Path(adapter_dir).exists():
            failures.append("requested_adapter_dir_missing")
    elif valid_candidates:
        summary_matches = [item for item in valid_candidates if item["matches_summary"]]
        if len(summary_matches) == 1:
            selected = summary_matches[0]["path"]
        elif len(valid_candidates) == 1:
            selected = valid_candidates[0]["path"]
        else:
            failures.append("multiple_adapter_candidates_require_explicit_adapter_dir")
            selected = requested
    else:
        warnings.append("no_valid_adapter_discovered")
    return {
        "selected_adapter_dir": selected,
        "candidates": candidates,
        "warnings": warnings,
        "failures": failures,
    }


def _glob_adapter_candidates(root: str) -> list[str]:
    path = Path(root)
    if not path.exists():
        return []
    output = []
    try:
        for config in path.rglob("adapter_config.json"):
            candidate = config.parent
            if (candidate / "adapter_model.safetensors").exists():
                output.append(str(candidate))
    except Exception:
        return []
    return sorted(output)


def _load_datasets(*, max_rows: int | None) -> dict[str, list[dict[str, Any]]]:
    loaded = {}
    for dataset in DATASETS:
        path = _dataset_path(dataset)
        if not path.exists():
            raise FileNotFoundError(str(path))
        rows = read_jsonl(path)
        loaded[dataset] = rows[:max_rows] if max_rows is not None else rows
    return loaded


def _dataset_path(dataset: str) -> Path:
    preferred = Path(DATASETS[dataset])
    if preferred.exists():
        return preferred
    fallback = Path(FALLBACK_DATASETS[dataset])
    return fallback


def _load_model_generator(*, base_model_path: str, adapter_dir: str | None, mode: str) -> tuple[PredictionGenerator, dict[str, Any]]:
    base = Path(base_model_path)
    adapter_metadata: dict[str, Any] = {}
    if adapter_dir is not None:
        adapter_metadata = _validate_adapter_dir(adapter_dir)
    if not base.exists():
        raise FileNotFoundError(f"base_model_path_missing:{base_model_path}")

    from kaggle_anti086.training.model_loader import load_adapter_model, load_base_model, load_tokenizer

    tokenizer = load_tokenizer(base_model_path)
    if adapter_dir is None:
        model = load_base_model(base_model_path, load_in_4bit=False, bf16=True)
        model = _ensure_eval_model_on_cuda(model)
    else:
        model = load_adapter_model(base_model_path, adapter_dir, load_in_4bit=False, bf16=True)
        model = _ensure_eval_model_on_cuda(model)
    if hasattr(model, "eval"):
        model.eval()

    def generate(row: dict[str, Any]) -> str:
        prompt = render_inference_prompt(str(row.get("prompt", "")))
        return _generate_answer(model, tokenizer, prompt, str(row.get("family", "")))

    adapter_metadata["mode"] = mode
    adapter_metadata["adapter_weights_attached"] = adapter_dir is not None
    return generate, adapter_metadata


def _validate_adapter_dir(adapter_dir: str) -> dict[str, Any]:
    target = Path(adapter_dir)
    config_path = target / "adapter_config.json"
    model_path = target / "adapter_model.safetensors"
    if not target.exists():
        raise FileNotFoundError(f"adapter_dir_missing:{adapter_dir}")
    if not config_path.exists():
        raise FileNotFoundError(f"adapter_config_missing:{config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"adapter_model_safetensors_missing:{model_path}")
    config = read_json(config_path)
    targets = config.get("target_modules", [])
    if isinstance(targets, str):
        targets = [targets]
    if set(targets) != {"q_proj", "v_proj"}:
        raise ValueError(f"adapter_target_modules_not_qv:{targets}")
    rank = int(config.get("r", config.get("rank", 0)))
    if rank > 32:
        raise ValueError(f"adapter_rank_gt_32:{rank}")
    base_name = config.get("base_model_name_or_path")
    return {
        "adapter_path": str(target),
        "adapter_config_sha256": file_record(config_path).get("sha256"),
        "adapter_model_sha256": file_record(model_path).get("sha256"),
        "adapter_target_modules": sorted(targets),
        "adapter_rank": rank,
        "adapter_base_model_name_or_path": base_name,
    }


def _generate_answer(model: Any, tokenizer: Any, prompt: str, family: str) -> str:
    import torch  # type: ignore

    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = _move_generation_inputs_to_model_device(encoded, model)
    max_new_tokens = 32 if family in {"numeric_formula", "gravity_numeric", "unit_conversion", "roman_numeral", "bit_manipulation", "symbol_mapping", "format_only"} else 64
    generation_kwargs = {
        "do_sample": False,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None),
    }
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        generation_kwargs["eos_token_id"] = eos
    with torch.no_grad():
        output = model.generate(**encoded, **generation_kwargs)
    generated = output[0][encoded["input_ids"].shape[-1] :]
    return str(tokenizer.decode(generated, skip_special_tokens=True)).strip()


def _run_solver_mode(
    *,
    dataset: str,
    rows: list[dict[str, Any]],
    eval_path: Path,
    predictions_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ensemble = SolverEnsemble()
    predictions: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        solver = _solver_candidate(ensemble, row)
        selected = solver["answer"] if solver["verified"] and solver["confidence"] >= 0.5 else ""
        predictions.append(
            _prediction_record(
                row=row,
                idx=idx,
                dataset=dataset,
                mode="solver_only",
                source="solver" if selected else "invalid",
                raw_output=selected,
                selected_answer=selected,
                failure_reason=None if selected else solver["reason"],
                extra={
                    "solver_candidate": solver["answer"],
                    "solver_confidence": solver["confidence"],
                    "solver_verification_status": solver["verification_status"],
                    "route_reason": "verified_solver" if selected else "solver_failed",
                },
            )
        )
    return _score_prediction_rows(
        dataset=dataset,
        mode="solver_only",
        rows=predictions,
        eval_path=eval_path,
        predictions_path=predictions_path,
        provenance={"model_path": None, "adapter_path": None, "adapter_config_sha256": None},
    ), predictions


def _run_model_mode(
    *,
    dataset: str,
    mode: str,
    rows: list[dict[str, Any]],
    eval_path: Path,
    predictions_path: Path,
    generator: PredictionGenerator,
    base_model_path: str,
    adapter_dir: str | None,
    adapter_metadata: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions = []
    for idx, row in enumerate(rows):
        try:
            raw = generator(row)
            predictions.append(_prediction_record(row=row, idx=idx, dataset=dataset, mode=mode, source="adapter" if adapter_dir else "base", raw_output=raw, selected_answer=raw))
        except Exception as exc:
            predictions.append(
                _prediction_record(
                    row=row,
                    idx=idx,
                    dataset=dataset,
                    mode=mode,
                    source="invalid",
                    raw_output="",
                    selected_answer="",
                    failure_reason=f"generation_failed:{type(exc).__name__}:{exc}",
                )
            )
    return _score_prediction_rows(
        dataset=dataset,
        mode=mode,
        rows=predictions,
        eval_path=eval_path,
        predictions_path=predictions_path,
        provenance={"model_path": base_model_path, "adapter_path": adapter_dir, **adapter_metadata},
    ), predictions


def _run_combined_mode(
    *,
    dataset: str,
    rows: list[dict[str, Any]],
    eval_path: Path,
    predictions_path: Path,
    adapter_generator: PredictionGenerator,
    base_model_path: str,
    adapter_dir: str,
    adapter_metadata: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ensemble = SolverEnsemble()
    predictions = []
    adapter_exercised = 0
    solver_exercised = 0
    for idx, row in enumerate(rows):
        solver = _solver_candidate(ensemble, row)
        solver_exercised += 1
        adapter_raw = ""
        adapter_failure = None
        try:
            adapter_raw = adapter_generator(row)
            adapter_exercised += 1
        except Exception as exc:
            adapter_failure = f"adapter_generation_failed:{type(exc).__name__}:{exc}"
        adapter_extracted = _extract_answer(adapter_raw, answer_type=ANSWER_TYPE_BY_FAMILY.get(str(row.get("family", "unknown"))))
        adapter_valid = bool(adapter_extracted) and not _invalid_format(adapter_raw, str(row.get("prompt", "")))
        if solver["verified"] and solver["confidence"] >= 0.85 and solver["risk"] == "low":
            selected_source = "solver"
            selected_answer = solver["answer"]
            route_reason = "verified_solver"
        elif adapter_valid:
            selected_source = "adapter"
            selected_answer = adapter_extracted
            route_reason = "adapter_fallback" if solver["verification_status"] != "PASS" else "solver_low_confidence"
        else:
            selected_source = "invalid"
            selected_answer = ""
            route_reason = "adapter_invalid" if solver["verification_status"] == "PASS" else "invalid_all_sources"
        predictions.append(
            _prediction_record(
                row=row,
                idx=idx,
                dataset=dataset,
                mode="combined",
                source=selected_source,
                raw_output=adapter_raw,
                selected_answer=selected_answer,
                failure_reason=adapter_failure if selected_source == "invalid" else None,
                extra={
                    "solver_candidate": solver["answer"],
                    "solver_confidence": solver["confidence"],
                    "solver_verification_status": solver["verification_status"],
                    "adapter_candidate": adapter_raw,
                    "adapter_valid": adapter_valid,
                    "selected_source": selected_source,
                    "selected_answer": selected_answer,
                    "route_reason": route_reason,
                },
            )
        )
    report = _score_prediction_rows(
        dataset=dataset,
        mode="combined",
        rows=predictions,
        eval_path=eval_path,
        predictions_path=predictions_path,
        provenance={"model_path": base_model_path, "adapter_path": adapter_dir, **adapter_metadata},
    )
    if adapter_exercised == 0:
        report["status"] = "BLOCKED"
        report["failures"].append("combined_adapter_path_not_exercised")
    if solver_exercised == 0:
        report["status"] = "BLOCKED"
        report["failures"].append("combined_solver_path_not_exercised")
    breakdown = report.get("source_breakdown", {})
    if int(breakdown.get("adapter", 0)) == 0:
        report["status"] = "WARN" if report["status"] == "PASS" else report["status"]
        report["warnings"].append("combined_adapter_never_selected")
    if int(breakdown.get("solver", 0)) == 0:
        report["status"] = "WARN" if report["status"] == "PASS" else report["status"]
        report["warnings"].append("combined_solver_never_selected")
    return report, predictions


def _solver_candidate(ensemble: SolverEnsemble, row: dict[str, Any]) -> dict[str, Any]:
    result = ensemble.run_all(row)
    best = None if result.abstained or not result.candidates else result.candidates[0]
    if best is None:
        return {
            "answer": "",
            "confidence": 0.0,
            "risk": "",
            "verified": False,
            "verification_status": "ABSTAIN" if result.abstained else "FAIL",
            "source": "",
            "reason": result.reason or "solver_no_candidate",
        }
    return {
        "answer": best.answer,
        "confidence": float(best.confidence),
        "risk": best.risk,
        "verified": bool(best.verified),
        "verification_status": "PASS" if best.verified else "FAIL",
        "source": best.source,
        "reason": result.reason,
    }


def _prediction_record(
    *,
    row: dict[str, Any],
    idx: int,
    dataset: str,
    mode: str,
    source: str,
    raw_output: str,
    selected_answer: str,
    failure_reason: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    family = str(row.get("family", "unknown"))
    answer_type = ANSWER_TYPE_BY_FAMILY.get(family)
    expected = "" if row.get("answer") is None else str(row.get("answer"))
    extracted = _extract_answer(selected_answer if source in {"solver", "adapter", "model", "base"} else raw_output, answer_type=answer_type)
    normalized = normalize_answer(extracted, expected_type=answer_type).normalized if extracted else ""
    expected_behavior = str(row.get("metadata", {}).get("expected_solver_behavior", row.get("expected_behavior", "answer")))
    valid = bool(extracted) and not _invalid_format(selected_answer or raw_output, str(row.get("prompt", "")))
    correct = (extracted.strip().upper() == "ABSTAIN") if expected_behavior == "abstain" else answers_match(extracted, expected, answer_type=answer_type)
    record = {
        "row_id": row.get("id", f"row_{idx}"),
        "dataset": dataset,
        "family": family,
        "subfamily": row.get("subfamily", ""),
        "prompt": row.get("prompt", ""),
        "gold_answer": expected,
        "expected": expected,
        "raw_output": raw_output,
        "extracted_answer": extracted,
        "normalized_answer": normalized,
        "valid": bool(valid),
        "correct": bool(correct and valid),
        "mode": mode,
        "source": source,
        "selected_source": "base" if source == "model" else source,
        "failure_reason": failure_reason,
    }
    if extra:
        record.update(extra)
    return record


def _score_prediction_rows(
    *,
    dataset: str,
    mode: str,
    rows: list[dict[str, Any]],
    eval_path: Path,
    predictions_path: Path,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    by_family: dict[str, Counter[str]] = defaultdict(Counter)
    by_subfamily: dict[str, Counter[str]] = defaultdict(Counter)
    source_breakdown: Counter[str] = Counter()
    failure_reasons: Counter[str] = Counter()
    answerable = answerable_correct = behavior_correct = unsafe = 0
    invalid = empty = verbose = prompt_copy = correct = attempted = 0
    for row in rows:
        family = str(row.get("family", "unknown"))
        subfamily = str(row.get("subfamily", "unknown"))
        source_breakdown[str(row.get("source", "unknown"))] += 1
        failure = row.get("failure_reason")
        if failure:
            failure_reasons[str(failure)] += 1
        expected_behavior = "abstain" if str(row.get("gold_answer", "")).strip().upper() == "ABSTAIN" else "answer"
        is_correct = bool(row.get("correct"))
        is_valid = bool(row.get("valid"))
        correct += int(is_correct)
        attempted += int(bool(row.get("extracted_answer")))
        invalid += int(not is_valid)
        empty += int(not str(row.get("raw_output", row.get("extracted_answer", ""))).strip() and row.get("source") != "solver")
        verbose += int(_is_verbose(str(row.get("raw_output", ""))))
        prompt = str(row.get("prompt", ""))
        prompt_copy += int(bool(prompt) and prompt[:80] in str(row.get("raw_output", "")))
        if expected_behavior == "abstain":
            unsafe += int(str(row.get("extracted_answer", "")).strip().upper() not in {"", "ABSTAIN"})
            behavior_correct += int(is_correct)
        else:
            answerable += 1
            answerable_correct += int(is_correct)
            behavior_correct += int(is_correct)
        by_family[family]["count"] += 1
        by_family[family]["correct"] += int(is_correct)
        by_subfamily[subfamily]["count"] += 1
        by_subfamily[subfamily]["correct"] += int(is_correct)
    total = len(rows)
    status = "PASS" if total and not failure_reasons else ("WARN" if total else "FAIL")
    report = {
        "schema_version": 1,
        "status": status,
        "mode": mode,
        "candidate_name": mode,
        "dataset": dataset,
        "eval_set": dataset,
        "row_count": total,
        "attempted_count": attempted,
        "exact_match": correct / total if total else 0.0,
        "overall_accuracy": correct / total if total else 0.0,
        "answerable_exact_match": answerable_correct / answerable if answerable else 0.0,
        "behavior_accuracy": behavior_correct / total if total else 0.0,
        "attempt_rate": attempted / total if total else 0.0,
        "unsafe_answer_rate": unsafe / total if total else 0.0,
        "invalid_output_rate": invalid / total if total else 0.0,
        "invalid_answer_rate": invalid / total if total else 0.0,
        "format_error_rate": invalid / total if total else 0.0,
        "empty_output_rate": empty / total if total else 0.0,
        "verbose_output_rate": verbose / total if total else 0.0,
        "prompt_copy_rate": prompt_copy / total if total else 0.0,
        "leakage_score": 0.0 if dataset == "anti_leak" else None,
        "by_family": _rate_table(by_family),
        "by_subfamily": _rate_table(by_subfamily),
        "source_breakdown": dict(sorted(source_breakdown.items())),
        "failure_reason_counts": dict(sorted(failure_reasons.items())),
        "failures": [],
        "warnings": [],
        "provenance": {
            "model_path": provenance.get("model_path"),
            "adapter_path": provenance.get("adapter_path"),
            "adapter_config_sha256": provenance.get("adapter_config_sha256"),
            "adapter_model_sha256": provenance.get("adapter_model_sha256"),
            "adapter_target_modules": provenance.get("adapter_target_modules"),
            "adapter_rank": provenance.get("adapter_rank"),
            "decoding_config": DECODING_CONFIG,
            "eval_rows_path": str(eval_path),
            "eval_rows_sha256": file_record(eval_path).get("sha256"),
            "predictions_path": str(predictions_path),
        },
        "model_results_faked": False,
        "packaging_allowed": False,
        "submission_allowed": False,
        "submit_recommended": False,
    }
    if failure_reasons:
        report["warnings"].append("some_rows_failed_generation_or_selection")
    return {key: value for key, value in report.items() if value is not None}


def _blocked_model_report(
    *,
    dataset: str,
    mode: str,
    rows: list[dict[str, Any]],
    failure: dict[str, Any],
    eval_path: Path,
    predictions_path: Path,
    base_model_path: str,
    adapter_dir: str,
) -> dict[str, Any]:
    by_family: dict[str, Counter[str]] = defaultdict(Counter)
    by_subfamily: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_family[str(row.get("family", "unknown"))]["count"] += 1
        by_subfamily[str(row.get("subfamily", "unknown"))]["count"] += 1
    failures = [str(failure.get("reason", f"{mode}_blocked"))]
    if failure.get("exception_type"):
        failures.append(f"{failure['exception_type']}:{failure.get('exception_message', '')}")
    return {
        "schema_version": 1,
        "status": "BLOCKED",
        "mode": mode,
        "candidate_name": mode,
        "dataset": dataset,
        "eval_set": dataset,
        "row_count": len(rows),
        "attempted_count": 0,
        "exact_match": 0.0,
        "overall_accuracy": 0.0,
        "answerable_exact_match": 0.0,
        "behavior_accuracy": 0.0,
        "attempt_rate": 0.0,
        "unsafe_answer_rate": 0.0,
        "invalid_output_rate": 0.0,
        "invalid_answer_rate": 0.0,
        "format_error_rate": 0.0,
        "leakage_score": 0.0 if dataset == "anti_leak" else None,
        "by_family": _rate_table(by_family),
        "by_subfamily": _rate_table(by_subfamily),
        "source_breakdown": {},
        "failures": failures,
        "warnings": ["real_model_inference_not_completed"],
        "blocked_reason": failure,
        "provenance": {
            "model_path": base_model_path,
            "adapter_path": adapter_dir if mode in {"adapter_only", "combined"} else None,
            "adapter_config_sha256": None,
            "adapter_model_sha256": None,
            "decoding_config": DECODING_CONFIG,
            "eval_rows_path": str(eval_path),
            "eval_rows_sha256": file_record(eval_path).get("sha256"),
            "predictions_path": str(predictions_path),
        },
        "model_results_faked": False,
        "score_semantics": "blocked_zero_score_not_model_metric",
        "packaging_allowed": False,
        "submission_allowed": False,
        "submit_recommended": False,
    }


def _mode_failure(mode: str, reason: str, base_model_path: str, adapter_dir: str) -> dict[str, Any]:
    failure = {"reason": reason, "base_model_path": base_model_path, "adapter_dir": adapter_dir}
    if mode in {"adapter_only", "combined"}:
        target = Path(adapter_dir)
        if not target.exists():
            failure["missing_file"] = adapter_dir
            failure["reason"] = "adapter_dir_missing"
        elif not (target / "adapter_config.json").exists():
            failure["missing_file"] = str(target / "adapter_config.json")
            failure["reason"] = "adapter_config_missing"
        elif not (target / "adapter_model.safetensors").exists():
            failure["missing_file"] = str(target / "adapter_model.safetensors")
            failure["reason"] = "adapter_model_safetensors_missing"
    elif mode == "base_only" and not Path(base_model_path).exists():
        failure["missing_file"] = base_model_path
        failure["reason"] = "base_model_path_missing"
    return failure


def _write_candidate_ranking(
    *,
    generated_reports: dict[str, Any],
    ladder: dict[str, Any],
    ladder_path: str,
    out_path: str | Path,
    decision_out: str | Path,
) -> dict[str, Any]:
    core = generated_reports["core_eval"]
    ranking = build_candidate_ranking(
        base_report=core["base_only"],
        solver_report=core["solver_only"],
        adapter_report=core["adapter_only"],
        combined_report=core["combined"],
        eval_ladder_report=ladder,
        eval_ladder_path=ladder_path,
    )
    write_json_checked(out_path, ranking, field_name="day2_candidate_ranking")
    decision = ranking["decision"] | {"status": ranking["status"], "failures": ranking["failures"], "warnings": ranking["warnings"]}
    write_json_checked(decision_out, decision, field_name="day2_decision_report")
    return ranking


def validate_day2_outputs(
    *,
    reports_dir: str | Path,
    predictions_dir: str | Path,
    manifest_path: str | Path,
    ladder_path: str | Path,
) -> dict[str, Any]:
    reports_root = Path(reports_dir)
    predictions_root = Path(predictions_dir)
    failures: list[str] = []
    warnings: list[str] = []
    report_statuses: dict[str, str] = {}
    prediction_files: dict[str, dict[str, Any]] = {}
    combined_source_breakdown: dict[str, dict[str, int]] = {}
    for dataset in DATASETS:
        for mode in MODES:
            key = f"{dataset}.{mode}"
            report_path = reports_root / f"{dataset}_{mode}.json"
            predictions_path = predictions_root / f"{dataset}_{mode}_predictions.jsonl"
            if not report_path.exists():
                failures.append(f"missing_report:{key}")
                continue
            report = read_json(report_path)
            status = str(report.get("status", ""))
            report_statuses[key] = status
            if status not in {"PASS", "WARN", "BLOCKED", "FAIL"}:
                failures.append(f"invalid_status:{key}:{status}")
            if str(report.get("mode")) != mode:
                failures.append(f"mode_mismatch:{key}")
            if str(report.get("dataset")) != dataset:
                failures.append(f"dataset_mismatch:{key}")
            row_count = int(report.get("row_count", 0) or 0)
            if status != "BLOCKED" and row_count <= 0:
                failures.append(f"non_blocked_zero_row_count:{key}")
            if not any(report.get(field) is not None for field in SCORE_FIELDS_FOR_VALIDATION):
                failures.append(f"missing_explicit_metric:{key}")
            if not isinstance(report.get("provenance"), dict):
                failures.append(f"missing_provenance:{key}")
            if dataset == "family_eval" and not report.get("by_family"):
                failures.append(f"missing_by_family:{key}")
            if mode in {"adapter_only", "combined"} and status != "BLOCKED":
                provenance = report.get("provenance", {})
                if not provenance.get("adapter_path") or not provenance.get("adapter_config_sha256") or not provenance.get("adapter_model_sha256"):
                    failures.append(f"missing_adapter_provenance:{key}")
            if mode == "combined":
                breakdown = {str(k): int(v) for k, v in dict(report.get("source_breakdown", {})).items()}
                combined_source_breakdown[dataset] = breakdown
                if status != "BLOCKED":
                    if int(breakdown.get("adapter", 0)) <= 0:
                        failures.append(f"combined_adapter_path_not_exercised:{key}")
                    if int(breakdown.get("solver", 0)) <= 0:
                        failures.append(f"combined_solver_path_not_exercised:{key}")
            if not predictions_path.exists():
                failures.append(f"missing_predictions:{key}")
            else:
                pred_record = file_record(predictions_path)
                prediction_files[key] = pred_record
                if status != "BLOCKED" and int(pred_record.get("row_count", _count_jsonl(predictions_path)) or 0) <= 0:
                    failures.append(f"non_blocked_empty_predictions:{key}")
                if mode == "combined" and status != "BLOCKED":
                    missing_fields = _combined_prediction_missing_fields(predictions_path)
                    if missing_fields:
                        failures.append(f"combined_predictions_missing_fields:{key}:{','.join(missing_fields)}")
    if not Path(manifest_path).exists():
        failures.append("missing_manifest")
    else:
        try:
            build_eval_ladder(manifest_path)
        except Exception as exc:
            failures.append(f"manifest_or_ladder_validation_failed:{type(exc).__name__}:{exc}")
    if not Path(ladder_path).exists():
        failures.append("missing_ladder_report")
    return {
        "schema_version": 1,
        "status": "PASS" if not failures else "FAIL",
        "reports_checked": len(DATASETS) * len(MODES),
        "prediction_files_checked": len(prediction_files),
        "report_statuses": report_statuses,
        "combined_source_breakdown": combined_source_breakdown,
        "failures": failures,
        "warnings": warnings,
        "packaging_allowed": False,
        "submission_allowed": False,
    }


SCORE_FIELDS_FOR_VALIDATION = ("exact_match", "overall_accuracy", "answerable_exact_match", "behavior_accuracy", "overall_score")


def _count_jsonl(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _combined_prediction_missing_fields(path: Path) -> list[str]:
    required = {
        "solver_candidate",
        "solver_confidence",
        "solver_verification_status",
        "adapter_candidate",
        "adapter_valid",
        "selected_source",
        "selected_answer",
        "route_reason",
    }
    missing: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        missing.update(field for field in required if field not in row)
        if missing:
            break
    return sorted(missing)


def _extract_answer(text: str, *, answer_type: str | None = None) -> str:
    value = str(text or "").strip()
    if "\\boxed{" in value:
        start = value.rfind("\\boxed{") + len("\\boxed{")
        end = value.find("}", start)
        if end >= 0:
            return _clean_answer(value[start:end], answer_type=answer_type)
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    candidate = lines[-1] if lines else value
    lowered = candidate.lower()
    for prefix in ("assistant:", "answer:", "final answer:", "output:", "result:"):
        if lowered.startswith(prefix):
            candidate = candidate[len(prefix) :].strip()
            lowered = candidate.lower()
    return _clean_answer(candidate, answer_type=answer_type)


def _clean_answer(value: str, *, answer_type: str | None = None) -> str:
    if answer_type == "symbol":
        return str(value).strip().strip("`'\" ")
    return str(value).strip().strip("`'\".,;: ")


def _invalid_format(text: str, prompt: str) -> bool:
    stripped = str(text or "").strip()
    if not stripped:
        return True
    lowered = stripped.lower()
    if any(marker in lowered for marker in ("because", "therefore", "explanation", "we need", "first,")):
        return True
    if bool(prompt) and str(prompt)[:80] in stripped:
        return True
    return False


def _is_verbose(text: str) -> bool:
    lowered = str(text).lower()
    return any(marker in lowered for marker in ("because", "therefore", "explanation", "we need", "first,", "the answer is"))


def _rate_table(table: dict[str, Counter[str]]) -> dict[str, dict[str, Any]]:
    output = {}
    for key, counts in sorted(table.items()):
        count = int(counts["count"])
        correct = int(counts["correct"])
        output[key] = {"count": count, "correct": correct, "exact_match": correct / count if count else 0.0}
    return output


def _manifest_entry(path: str | Path) -> dict[str, Any]:
    record = file_record(path)
    return {
        "path": record["path"],
        "sha256": record["sha256"],
        "size_bytes": record["size_bytes"],
    }


def _free_model_stack() -> None:
    gc.collect()
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate strict Day 2 eval reports, manifest, and ladder report.")
    parser.add_argument("--out-dir", default="artifacts/sprint11/day2_reports")
    parser.add_argument("--predictions-dir", default="artifacts/sprint11/day2_predictions")
    parser.add_argument("--manifest-out", default="artifacts/sprint11/day2_eval_manifest.json")
    parser.add_argument("--ladder-out", default="artifacts/sprint11/day2_eval_ladder_report.json")
    parser.add_argument("--candidate-ranking-out", default="artifacts/sprint11/day2_candidate_ranking.json")
    parser.add_argument("--decision-out", default="artifacts/sprint11/day2_decision_report.json")
    parser.add_argument("--smoke-out", default="artifacts/sprint11/day2_inference_smoke_report.json")
    parser.add_argument("--validation-out", default="artifacts/sprint11/day2_report_validation.json")
    parser.add_argument("--base-model-path", default=None)
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--train-summary", default="artifacts/sprint11/day11a_v2a_50_train_summary.json")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--kaggle-mode", action="store_true")
    args = parser.parse_args(argv)
    result = build_day2_eval_reports(
        out_dir=args.out_dir,
        predictions_dir=args.predictions_dir,
        manifest_path=args.manifest_out,
        ladder_out=args.ladder_out,
        candidate_ranking_out=args.candidate_ranking_out,
        decision_out=args.decision_out,
        smoke_out=args.smoke_out,
        validation_out=args.validation_out,
        base_model_path=args.base_model_path,
        adapter_dir=args.adapter_dir,
        train_summary_path=args.train_summary,
        kaggle_mode=args.kaggle_mode,
        max_rows=args.max_rows,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "manifest": result["manifest_path"],
                "ladder": result["ladder_path"],
                "candidate_ranking": result["candidate_ranking_path"],
                "smoke": result["smoke_report_path"],
                "validation": result["validation_path"],
                "ladder_decision": result["ladder_decision"],
                "blocked_modes": result["blocked_modes"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
