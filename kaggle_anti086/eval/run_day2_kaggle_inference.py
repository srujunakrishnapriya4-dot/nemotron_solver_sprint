from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import zipfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.v2_corpus_io import file_record, read_json, write_json_checked
from kaggle_anti086.eval.day2_generate_eval_reports import (
    DATASETS,
    MODES,
    build_day2_eval_reports,
    validate_day2_outputs,
    _run_inference_smoke,
    _validate_adapter_dir,
)


PREFERRED_ADAPTER_MARKERS = ("v2a_bf16_qv_50", "20260606_141808_42a82878")
DEFAULT_RESTORE_ROOT = Path("/kaggle/working/anti086_adapters_restored")
ARTIFACTS_TO_EXPORT = (
    "day2_reports",
    "day2_predictions",
    "day2_eval_manifest.json",
    "day2_eval_ladder_report.json",
    "day2_candidate_ranking.json",
    "day2_decision_report.json",
    "day2_inference_smoke_report.json",
    "day2_report_validation.json",
)


def run_day2_kaggle_inference(
    *,
    base_model_path: str | Path,
    adapter_dir: str | Path | None,
    adapter_zip: str | Path | None,
    auto_discover_adapter: bool,
    out_dir: str | Path,
    predictions_dir: str | Path,
    manifest_out: str | Path,
    ladder_out: str | Path,
    candidate_ranking_out: str | Path,
    decision_out: str | Path,
    smoke_out: str | Path,
    validation_out: str | Path,
    export_zip: str | Path | None,
    kaggle_mode: bool,
    dry_run_verify_only: bool = False,
    max_rows: int | None = None,
) -> dict[str, Any]:
    base = Path(base_model_path)
    selected_adapter: Path | None = None
    restore_report: dict[str, Any] = {"adapter_zip": str(adapter_zip) if adapter_zip else None, "restored_root": None, "restored": False}
    if adapter_zip:
        restore_report = restore_adapter_zip(adapter_zip, DEFAULT_RESTORE_ROOT)
    smoke: dict[str, Any]
    if not base.exists():
        smoke = _blocked_smoke(base, Path(adapter_dir) if adapter_dir else None, ["base_model_path_missing"])
        write_json_checked(smoke_out, smoke, field_name="day2_inference_smoke_report")
        return _blocked_result(
            reason="base_model_path_missing",
            base_model_path=base,
            adapter_dir=selected_adapter,
            smoke=smoke,
            restore_report=restore_report,
            export_zip=export_zip,
        )
    adapter_resolution = resolve_adapter_dir(adapter_dir=adapter_dir, auto_discover=auto_discover_adapter or adapter_dir is None, extra_roots=[DEFAULT_RESTORE_ROOT])
    if adapter_resolution["status"] != "PASS":
        smoke = _blocked_smoke(base, Path(adapter_resolution.get("selected_adapter_dir") or adapter_dir or ""), adapter_resolution["failures"])
        write_json_checked(smoke_out, smoke, field_name="day2_inference_smoke_report")
        return _blocked_result(
            reason="adapter_dir_missing" if "adapter_dir_missing" in adapter_resolution["failures"] else "adapter_resolution_failed",
            base_model_path=base,
            adapter_dir=Path(adapter_resolution.get("selected_adapter_dir") or adapter_dir or ""),
            smoke=smoke,
            restore_report=restore_report,
            adapter_resolution=adapter_resolution,
            export_zip=export_zip,
        )
    selected_adapter = Path(adapter_resolution["selected_adapter_dir"])
    smoke = _run_inference_smoke(base_model_path=str(base), adapter_dir=str(selected_adapter), kaggle_mode=kaggle_mode)
    smoke["adapter_resolution"] = adapter_resolution
    smoke["restore_report"] = restore_report
    write_json_checked(smoke_out, smoke, field_name="day2_inference_smoke_report")
    if smoke.get("status") != "PASS":
        return _blocked_result(
            reason="smoke_preflight_failed",
            base_model_path=base,
            adapter_dir=selected_adapter,
            smoke=smoke,
            restore_report=restore_report,
            adapter_resolution=adapter_resolution,
            export_zip=export_zip,
        )
    if dry_run_verify_only:
        return {
            "status": "PASS_VERIFY_ONLY",
            "base_model_path": str(base),
            "adapter_dir": str(selected_adapter),
            "smoke_status": smoke.get("status"),
            "reports_generated": 0,
            "prediction_files_generated": 0,
            "export_zip": None,
            "packaging_allowed": False,
            "submission_allowed": False,
        }
    result = build_day2_eval_reports(
        out_dir=out_dir,
        predictions_dir=predictions_dir,
        manifest_path=manifest_out,
        ladder_out=ladder_out,
        candidate_ranking_out=candidate_ranking_out,
        decision_out=decision_out,
        smoke_out=smoke_out,
        validation_out=validation_out,
        base_model_path=str(base),
        adapter_dir=str(selected_adapter),
        kaggle_mode=kaggle_mode,
        max_rows=max_rows,
    )
    validation = validate_day2_outputs(reports_dir=out_dir, predictions_dir=predictions_dir, manifest_path=manifest_out, ladder_path=ladder_out)
    write_json_checked(validation_out, validation, field_name="day2_report_validation")
    export_record = None
    if export_zip:
        export_record = create_export_bundle(export_zip, artifacts_root=Path("artifacts/sprint11"))
    return result | {
        "status": "PASS" if validation["status"] == "PASS" else "FAIL",
        "adapter_resolution": adapter_resolution,
        "restore_report": restore_report,
        "smoke_status": smoke.get("status"),
        "validation_status": validation["status"],
        "export_zip": None if export_record is None else export_record["path"],
        "packaging_allowed": False,
        "submission_allowed": False,
    }


def restore_adapter_zip(adapter_zip: str | Path, restore_root: str | Path = DEFAULT_RESTORE_ROOT) -> dict[str, Any]:
    source = Path(adapter_zip)
    if not source.exists():
        return {"status": "BLOCKED", "adapter_zip": str(source), "restored": False, "failures": ["adapter_zip_missing"]}
    target = Path(restore_root)
    target.mkdir(parents=True, exist_ok=True)
    extracted = []
    with zipfile.ZipFile(source, "r") as archive:
        for member in archive.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"unsafe_zip_member:{member.filename}")
            out_path = target / member_path
            if member.is_dir():
                out_path.mkdir(parents=True, exist_ok=True)
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member, "r") as src, out_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(str(out_path))
    return {"status": "PASS", "adapter_zip": str(source), "restored_root": str(target), "restored": True, "extracted_files": extracted, "failures": []}


def resolve_adapter_dir(*, adapter_dir: str | Path | None, auto_discover: bool, extra_roots: list[str | Path] | None = None) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    roots = ["/kaggle/working/anti086_adapters", "/kaggle/input"]
    roots.extend(str(root) for root in (extra_roots or []))
    if adapter_dir:
        explicit = Path(adapter_dir)
        candidates.append(_adapter_candidate_record(explicit, explicit=True))
        if _is_valid_adapter_dir(explicit):
            return _validated_adapter_result(explicit, candidates)
    if not auto_discover:
        return {"status": "BLOCKED", "selected_adapter_dir": str(adapter_dir) if adapter_dir else None, "candidates": candidates, "failures": ["adapter_dir_missing"]}
    for root in roots:
        candidates.extend(_discover_adapter_candidates(Path(root)))
    valid = [item for item in candidates if item["valid_adapter"]]
    if not valid:
        return {"status": "BLOCKED", "selected_adapter_dir": str(adapter_dir) if adapter_dir else None, "candidates": candidates, "failures": ["adapter_dir_missing"]}
    preferred = _preferred_candidates(valid)
    if len(preferred) == 1:
        return _validated_adapter_result(Path(preferred[0]["path"]), candidates)
    if len(valid) == 1:
        return _validated_adapter_result(Path(valid[0]["path"]), candidates)
    return {"status": "BLOCKED", "selected_adapter_dir": None, "candidates": candidates, "failures": ["multiple_adapter_candidates_require_explicit_adapter_dir"]}


def _validated_adapter_result(path: Path, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        metadata = _validate_adapter_dir(str(path))
        config = read_json(path / "adapter_config.json")
        failures = _adapter_config_failures(config)
        if failures:
            return {"status": "BLOCKED", "selected_adapter_dir": str(path), "candidates": candidates, "adapter_metadata": metadata, "failures": failures}
        return {"status": "PASS", "selected_adapter_dir": str(path), "candidates": candidates, "adapter_metadata": metadata, "failures": []}
    except Exception as exc:
        return {"status": "BLOCKED", "selected_adapter_dir": str(path), "candidates": candidates, "failures": [f"adapter_validation_failed:{type(exc).__name__}:{exc}"]}


def _adapter_config_failures(config: dict[str, Any]) -> list[str]:
    failures = []
    if str(config.get("peft_type", "LORA")).upper() != "LORA":
        failures.append("adapter_peft_type_not_lora")
    if str(config.get("task_type", "CAUSAL_LM")).upper() != "CAUSAL_LM":
        failures.append("adapter_task_type_not_causal_lm")
    rank = int(config.get("r", config.get("rank", 0)))
    if rank > 32:
        failures.append("adapter_rank_gt_32")
    targets = config.get("target_modules", [])
    if isinstance(targets, str):
        targets = [targets]
    target_set = set(str(item) for item in targets)
    if not {"q_proj", "v_proj"}.issubset(target_set):
        failures.append("adapter_missing_qv_targets")
    unexpected = sorted(target_set - {"q_proj", "v_proj"})
    if unexpected:
        failures.append("adapter_unexpected_target_modules:" + ",".join(unexpected))
    return failures


def _preferred_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored = []
    for item in candidates:
        path = str(item["path"])
        score = sum(1 for marker in PREFERRED_ADAPTER_MARKERS if marker in path)
        if score:
            scored.append((score, item))
    if not scored:
        return []
    best = max(score for score, _ in scored)
    return [item for score, item in scored if score == best]


def _discover_adapter_candidates(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    output = []
    try:
        for config in root.rglob("adapter_config.json"):
            output.append(_adapter_candidate_record(config.parent, explicit=False))
    except Exception:
        return []
    return output


def _adapter_candidate_record(path: Path, *, explicit: bool) -> dict[str, Any]:
    return {"path": str(path), "exists": path.exists(), "explicit": explicit, "valid_adapter": _is_valid_adapter_dir(path)}


def _is_valid_adapter_dir(path: Path) -> bool:
    return path.exists() and (path / "adapter_config.json").exists() and (path / "adapter_model.safetensors").exists()


def _blocked_smoke(base_model_path: Path, adapter_dir: Path | None, failures: list[str]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if adapter_dir and _is_valid_adapter_dir(adapter_dir):
        try:
            metadata = _validate_adapter_dir(str(adapter_dir))
        except Exception:
            metadata = {}
    return {
        "schema_version": 1,
        "status": "BLOCKED",
        "base_model_path": str(base_model_path),
        "adapter_dir": None if adapter_dir is None else str(adapter_dir),
        "tokenizer_load": False,
        "base_model_load": False,
        "base_generate": False,
        "adapter_config_exists": bool(adapter_dir and (adapter_dir / "adapter_config.json").exists()),
        "adapter_model_exists": bool(adapter_dir and (adapter_dir / "adapter_model.safetensors").exists()),
        "adapter_load": False,
        "adapter_generate": False,
        "adapter_config_sha256": metadata.get("adapter_config_sha256"),
        "adapter_model_sha256": metadata.get("adapter_model_sha256"),
        "decoding_config": {},
        "failures": failures,
        "warnings": [],
        "packaging_allowed": False,
        "submission_allowed": False,
    }


def _blocked_result(
    *,
    reason: str,
    base_model_path: Path,
    adapter_dir: Path | None,
    smoke: dict[str, Any],
    restore_report: dict[str, Any],
    export_zip: str | Path | None,
    adapter_resolution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": "BLOCKED",
        "reason": reason,
        "base_model_path": str(base_model_path),
        "adapter_dir": None if adapter_dir is None else str(adapter_dir),
        "smoke_status": smoke.get("status"),
        "smoke_report": smoke,
        "restore_report": restore_report,
        "adapter_resolution": adapter_resolution,
        "reports_generated": 0,
        "prediction_files_generated": 0,
        "export_zip": None,
        "packaging_allowed": False,
        "submission_allowed": False,
    }


def create_export_bundle(export_zip: str | Path, *, artifacts_root: str | Path = "artifacts/sprint11") -> dict[str, Any]:
    target = Path(export_zip)
    root = Path(artifacts_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    included = []
    excluded = []
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in ARTIFACTS_TO_EXPORT:
            path = root / item
            if not path.exists():
                continue
            if path.is_dir():
                for file in sorted(p for p in path.rglob("*") if p.is_file()):
                    if _forbidden_export_file(file):
                        excluded.append(str(file))
                        continue
                    archive.write(file, file.relative_to(root).as_posix())
                    included.append(str(file))
            elif path.is_file():
                if _forbidden_export_file(path):
                    excluded.append(str(path))
                    continue
                archive.write(path, path.relative_to(root).as_posix())
                included.append(str(path))
    return {"path": str(target), "included_count": len(included), "excluded_files": excluded, **file_record(target)}


def _forbidden_export_file(path: Path) -> bool:
    name = path.name
    return path.suffix in {".safetensors", ".pt", ".bin"} or name in {"submission.zip", "optimizer.pt", "trainer_state.json"}


def decision_table(*, reports_dir: str | Path, ladder_path: str | Path, ranking_path: str | Path) -> str:
    reports_root = Path(reports_dir)
    lines = []
    for mode in MODES:
        lines.append(f"{mode}:")
        for dataset in DATASETS:
            report = read_json(reports_root / f"{dataset}_{mode}.json")
            lines.append(f"  {dataset}: {report.get('status')} exact_match={report.get('exact_match')}")
        lines.append("")
    aggregate = {"solver": 0, "adapter": 0, "base_fallback": 0, "invalid": 0}
    for dataset in DATASETS:
        report = read_json(reports_root / f"{dataset}_combined.json")
        breakdown = report.get("source_breakdown", {})
        for key in aggregate:
            aggregate[key] += int(breakdown.get(key, 0))
    lines.append("combined_source_breakdown:")
    for key in ("solver", "adapter", "base_fallback", "invalid"):
        lines.append(f"  {key}: {aggregate[key]}")
    ladder = read_json(ladder_path)
    decision = ladder.get("decision", {})
    lines.append("")
    lines.append("ladder:")
    lines.append(f"  status: {ladder.get('status')}")
    lines.append(f"  decision: {decision.get('decision')}")
    lines.append(f"  train_v2a_150: {decision.get('train_v2a_150')}")
    lines.append(f"  reason_codes: {decision.get('reason_codes', [])}")
    blocked = []
    for mode in MODES:
        if mode == "solver_only":
            continue
        for dataset in DATASETS:
            report = read_json(reports_root / f"{dataset}_{mode}.json")
            if report.get("status") == "BLOCKED":
                blocked.append(f"{dataset}.{mode}")
    lines.append("")
    lines.append("blocked_modes:")
    lines.extend(f"  {item}" for item in blocked)
    ranking = read_json(ranking_path)
    lines.append("")
    lines.append(f"next_action: {next_action(ladder=ladder, ranking=ranking, blocked_modes=blocked)}")
    return "\n".join(lines)


def next_action(*, ladder: dict[str, Any], ranking: dict[str, Any], blocked_modes: list[str]) -> str:
    if any(".base_only" in item for item in blocked_modes):
        return "run_kaggle_inference"
    if any(".adapter_only" in item for item in blocked_modes):
        return "fix_adapter_inference"
    if any(".combined" in item for item in blocked_modes):
        return "fix_adapter_inference"
    if bool(ladder.get("decision", {}).get("train_v2a_150", False)):
        return "train_v2a_150"
    reason_codes = set(ladder.get("decision", {}).get("reason_codes", []))
    if "combined_not_better_than_solver" in reason_codes:
        return "fix_router_solver_first"
    if "adapter_not_better_than_base" in reason_codes:
        return "stop_adapter_scaling"
    return "stop_adapter_scaling"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Kaggle-safe Day 2 real inference wrapper.")
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--adapter-dir")
    parser.add_argument("--adapter-zip")
    parser.add_argument("--auto-discover-adapter", action="store_true")
    parser.add_argument("--out-dir", default="artifacts/sprint11/day2_reports")
    parser.add_argument("--predictions-dir", default="artifacts/sprint11/day2_predictions")
    parser.add_argument("--manifest-out", default="artifacts/sprint11/day2_eval_manifest.json")
    parser.add_argument("--ladder-out", default="artifacts/sprint11/day2_eval_ladder_report.json")
    parser.add_argument("--candidate-ranking-out", default="artifacts/sprint11/day2_candidate_ranking.json")
    parser.add_argument("--decision-out", default="artifacts/sprint11/day2_decision_report.json")
    parser.add_argument("--smoke-out", default="artifacts/sprint11/day2_inference_smoke_report.json")
    parser.add_argument("--validation-out", default="artifacts/sprint11/day2_report_validation.json")
    parser.add_argument("--export-zip", default="/kaggle/working/day2_real_inference_artifacts.zip")
    parser.add_argument("--dry-run-verify-only", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--kaggle-mode", action="store_true")
    args = parser.parse_args(argv)
    result = run_day2_kaggle_inference(
        base_model_path=args.base_model_path,
        adapter_dir=args.adapter_dir,
        adapter_zip=args.adapter_zip,
        auto_discover_adapter=args.auto_discover_adapter,
        out_dir=args.out_dir,
        predictions_dir=args.predictions_dir,
        manifest_out=args.manifest_out,
        ladder_out=args.ladder_out,
        candidate_ranking_out=args.candidate_ranking_out,
        decision_out=args.decision_out,
        smoke_out=args.smoke_out,
        validation_out=args.validation_out,
        export_zip=args.export_zip,
        kaggle_mode=args.kaggle_mode,
        dry_run_verify_only=args.dry_run_verify_only,
        max_rows=args.max_rows,
    )
    print(json.dumps({"status": result["status"], "reason": result.get("reason"), "smoke_status": result.get("smoke_status"), "export_zip": result.get("export_zip")}, sort_keys=True))
    if result.get("ladder_path") and Path(args.ladder_out).exists() and Path(args.candidate_ranking_out).exists() and Path(args.out_dir).exists():
        print(decision_table(reports_dir=args.out_dir, ladder_path=args.ladder_out, ranking_path=args.candidate_ranking_out))
    return 0 if result["status"] in {"PASS", "PASS_VERIFY_ONLY", "BLOCKED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
