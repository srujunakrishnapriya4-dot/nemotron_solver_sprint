from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.v2_corpus_io import read_json, write_json_checked
from kaggle_anti086.solvers.answer_normalizer import normalize_answer


FAILURE_MODE_KEYS = (
    "01_repo_code_missing",
    "02_bad_kaggle_dataset_packaging",
    "03_adapter_path_mismatch",
    "04_mamba_ssm_cutlass_import_failure",
    "05_cuda_cpu_device_mismatch",
    "06_model_input_tensor_device_mismatch",
    "07_blackwell_ptxas_permission_failure",
    "08_missing_day5_eval_jsonl_files",
    "09_empty_symbol_normalizer_crash",
    "10_full_wrapper_slow_incomplete_combined",
    "11_eval_manifest_schema_mismatch",
    "12_eval_ladder_cli_no_output",
    "13_base_model_accuracy_extremely_low",
    "14_v2a_50_adapter_worse_or_equal_to_base",
    "15_combined_mode_not_measured",
    "16_solver_raw_exact_match_below_0_93",
    "17_old_v1_custom_numeral_unsupported",
    "18_old_v1_equation_operator_unsupported",
    "19_old_v1_sequence_pattern_unsupported",
    "20_old_v1_permutation_sorting_unsupported",
    "21_failure_report_v1_abstain_misclassification",
)
OLD_V1_FAMILIES = {
    "17_old_v1_custom_numeral_unsupported": "custom_numeral",
    "18_old_v1_equation_operator_unsupported": "equation_operator",
    "19_old_v1_sequence_pattern_unsupported": "sequence_pattern",
    "20_old_v1_permutation_sorting_unsupported": "permutation_sorting",
}
REQUIRED_REPO_FILES = (
    "kaggle_anti086",
    "tests",
    "kaggle_anti086/eval/eval_ladder.py",
    "kaggle_anti086/eval/run_day2_kaggle_inference.py",
)
DAY5_DATASETS = (
    "artifacts/sprint11/day5_private_like_eval_512.jsonl",
    "artifacts/sprint11/day5_family_hard_eval_512.jsonl",
    "artifacts/sprint11/day5_rule_holdout_eval_512.jsonl",
    "artifacts/sprint11/day5_anti_leak_eval_256.jsonl",
)
DAY5_FALLBACKS = (
    "artifacts/sprint11/day5_private_like_answerable_512.jsonl",
    "artifacts/sprint11/day5_family_hard_answerable_512.jsonl",
    "artifacts/sprint11/day5_rule_holdout_answerable_512.jsonl",
    "artifacts/sprint11/day5_anti_leak_answerable_256.jsonl",
)


def build_hardening_audit(
    *,
    repo_root: str | Path = ".",
    failure_report_v2_path: str | Path = "artifacts/sprint11/day2_failure_report_v2.json",
    eval_ladder_report_path: str | Path = "artifacts/sprint11/day2_eval_ladder_report.json",
    manifest_path: str | Path = "artifacts/sprint11/day2_eval_manifest.json",
    reports_dir: str | Path = "artifacts/sprint11/day2_reports",
) -> dict[str, Any]:
    root = Path(repo_root)
    failure_v2 = _read_optional(root / failure_report_v2_path)
    ladder = _read_optional(root / eval_ladder_report_path)
    manifest = _read_optional(root / manifest_path)
    modes: dict[str, Any] = {}
    modes["01_repo_code_missing"] = _check_repo_code(root)
    modes["02_bad_kaggle_dataset_packaging"] = _check_bad_packaging(root)
    modes["03_adapter_path_mismatch"] = _check_adapter_path_resolution(root)
    modes["04_mamba_ssm_cutlass_import_failure"] = _record(
        "NOT_APPLICABLE",
        {"local_cpu_context": True, "wrapper_import_is_lazy": _file_contains(root / "kaggle_anti086/eval/day2_generate_eval_reports.py", "_load_model_generator")},
        "Kaggle wrapper must report exact import errors instead of requiring local mamba/cutlass imports.",
        "guarded",
    )
    modes["05_cuda_cpu_device_mismatch"] = _check_device_helpers(root, "cuda_cpu_device_mismatch")
    modes["06_model_input_tensor_device_mismatch"] = _check_device_helpers(root, "model_input_tensor_device_mismatch")
    modes["07_blackwell_ptxas_permission_failure"] = _check_ptxas_guard(root)
    modes["08_missing_day5_eval_jsonl_files"] = _check_day5_datasets(root)
    modes["09_empty_symbol_normalizer_crash"] = _check_empty_symbol_normalizer(root)
    modes["10_full_wrapper_slow_incomplete_combined"] = _check_wrapper_combined_guard(root, reports_dir)
    modes["11_eval_manifest_schema_mismatch"] = _check_manifest_schema(manifest)
    modes["12_eval_ladder_cli_no_output"] = _check_eval_ladder_cli(root)
    modes["13_base_model_accuracy_extremely_low"] = _performance_record_from_ladder(ladder, "base_exact_match", "Base model measured weak; this is a performance limitation, not an infrastructure fix.")
    modes["14_v2a_50_adapter_worse_or_equal_to_base"] = _adapter_weak_record(ladder)
    modes["15_combined_mode_not_measured"] = _check_combined_measured(root, reports_dir)
    modes["16_solver_raw_exact_match_below_0_93"] = _solver_score_record(ladder)
    for key, family in OLD_V1_FAMILIES.items():
        modes[key] = _old_v1_family_record(family, failure_v2)
    modes["21_failure_report_v1_abstain_misclassification"] = _check_failure_report_v2_guard(root, failure_v2)
    summary = _summarize(modes)
    status = "FAIL" if summary["unresolved_count"] else ("WARN" if summary["blocked_count"] else "PASS")
    return {
        "schema_version": 1,
        "created_by": "DAY2_21_FAILURE_HARDENING_AUDIT",
        "status": status,
        "failure_modes": {key: modes[key] for key in FAILURE_MODE_KEYS},
        "summary": summary,
        "decision": {
            "v2a_150_authorized": False,
            "package_authorized": False,
            "submission_authorized": False,
            "next_action": _next_action(summary),
        },
        "model_results_faked": False,
        "leaderboard_claim": False,
        "no_0_93_evidence": True,
        "no_0_95_evidence": True,
    }


def _check_repo_code(root: Path) -> dict[str, Any]:
    missing = [path for path in REQUIRED_REPO_FILES if not (root / path).exists()]
    return _record(
        "FAIL" if missing else "PASS",
        {"missing": missing, "required": list(REQUIRED_REPO_FILES)},
        "Kaggle notebook must contain repo code, tests, ladder, and wrapper.",
        "unresolved" if missing else "fixed",
    )


def _check_bad_packaging(root: Path) -> dict[str, Any]:
    matches = [str(path.relative_to(root)) for path in root.rglob("win_system_kaggle_cells") if ".git" not in path.parts]
    tracked = _tracked_paths(root, "win_system_kaggle_cells")
    status = "FAIL" if tracked else "PASS"
    return _record(status, {"found_untracked_or_local": matches, "tracked": tracked}, "No duplicate bad Kaggle dataset folders may be tracked.", "unresolved" if tracked else "guarded")


def _check_adapter_path_resolution(root: Path) -> dict[str, Any]:
    wrapper = root / "kaggle_anti086/eval/run_day2_kaggle_inference.py"
    text = _read_text(wrapper)
    code_guarded = all(token in text for token in ("resolve_adapter_dir", "adapter_zip", "auto_discover_adapter"))
    local_adapter_exists = any((root / path).exists() for path in ("anti086_adapters", "artifacts/sprint11/day11a_v2a_50_train_summary.json"))
    return _record(
        "PASS" if code_guarded and local_adapter_exists else ("BLOCKED" if code_guarded else "FAIL"),
        {"resolution_code_exists": code_guarded, "local_adapter_artifact_present": local_adapter_exists},
        "Adapter must be explicit or discovered with validation, never silently random.",
        "blocked" if code_guarded and not local_adapter_exists else ("guarded" if code_guarded else "unresolved"),
    )


def _check_device_helpers(root: Path, name: str) -> dict[str, Any]:
    target = root / "kaggle_anti086/eval/day2_generate_eval_reports.py"
    text = _read_text(target)
    helpers = {
        "_ensure_eval_model_on_cuda": "_ensure_eval_model_on_cuda" in text,
        "_move_generation_inputs_to_model_device": "_move_generation_inputs_to_model_device" in text,
        "generate_uses_moved_inputs": "_move_generation_inputs_to_model_device(encoded, model)" in text,
    }
    ok = all(helpers.values())
    return _record("PASS" if ok else "FAIL", {"check": name, "helpers": helpers}, "Model and generation inputs must be on the same CUDA device.", "guarded" if ok else "unresolved")


def _check_ptxas_guard(root: Path) -> dict[str, Any]:
    runtime = root / "kaggle_anti086/kaggle_runtime_patches.py"
    benchmark = root / "kaggle_anti086/eval/run_stack_benchmark.py"
    has_guard = _file_contains(runtime, "TRITON_PTXAS_PATH") or _file_contains(benchmark, "TRITON_PTXAS_PATH")
    return _record("PASS" if has_guard else "BLOCKED", {"ptxas_env_guard_present": has_guard}, "Blackwell ptxas must be patched/recorded by runtime preflight, not chmodded in repo.", "guarded" if has_guard else "blocked")


def _check_day5_datasets(root: Path) -> dict[str, Any]:
    present = [path for path in DAY5_DATASETS if (root / path).exists()]
    fallback_present = [path for path in DAY5_FALLBACKS if (root / path).exists()]
    generator_text = _read_text(root / "kaggle_anti086/eval/day2_generate_eval_reports.py")
    fallback_code = "FALLBACK_DATASETS" in generator_text and "_load_datasets" in generator_text
    ok = fallback_code and (len(present) == len(DAY5_DATASETS) or len(fallback_present) == len(DAY5_FALLBACKS))
    return _record("PASS" if ok else "FAIL", {"present": present, "fallback_present": fallback_present, "fallback_code": fallback_code}, "Missing eval rows must raise loudly or use explicit fallback, never empty eval.", "guarded" if ok else "unresolved")


def _check_empty_symbol_normalizer(root: Path) -> dict[str, Any]:
    try:
        normalized = normalize_answer("", expected_type="symbol")
        safe = normalized.normalized == ""
    except Exception as exc:  # pragma: no cover - defensive branch
        return _record("FAIL", {"exception": f"{type(exc).__name__}:{exc}"}, "Empty symbol output must normalize safely.", "unresolved")
    tests_cover = _file_contains(root / "tests/test_day2_symbol_mapping_answerable_fix.py", "abstain_correct") or _file_contains(root / "tests/test_day2_failure_report_v2.py", "abstain_correct")
    return _record("PASS" if safe and tests_cover else "WARN", {"normalized": normalized.normalized, "tests_cover": tests_cover}, "Empty symbol output must not crash or become unsafe answer.", "guarded" if safe else "unresolved")


def _check_wrapper_combined_guard(root: Path, reports_dir: str | Path) -> dict[str, Any]:
    text = _read_text(root / "kaggle_anti086/eval/day2_generate_eval_reports.py")
    has_guard = "combined_adapter_path_not_exercised" in text and "combined_solver_path_not_exercised" in text
    invalid_pass = _combined_pass_without_sources(root / reports_dir)
    status = "FAIL" if invalid_pass else ("PASS" if has_guard else "BLOCKED")
    return _record(status, {"combined_guard_code_present": has_guard, "invalid_pass_reports": invalid_pass}, "Incomplete combined cannot produce PASS.", "guarded" if has_guard and not invalid_pass else "unresolved")


def _check_manifest_schema(manifest: dict[str, Any] | None) -> dict[str, Any]:
    if manifest is None:
        return _record("BLOCKED", {"manifest_present": False}, "Eval manifest must be schema_version 1.", "blocked")
    schema_ok = manifest.get("schema_version") == 1 and isinstance(manifest.get("datasets"), dict)
    return _record("PASS" if schema_ok else "FAIL", {"schema_version": manifest.get("schema_version"), "has_datasets": isinstance(manifest.get("datasets"), dict)}, "Eval ladder manifest schema must be explicit and compatible.", "guarded" if schema_ok else "unresolved")


def _check_eval_ladder_cli(root: Path) -> dict[str, Any]:
    text = _read_text(root / "kaggle_anti086/eval/eval_ladder.py")
    has_cli = "argparse" in text and "write_eval_ladder" in text and "__main__" in text
    return _record("PASS" if has_cli else "FAIL", {"cli_present": has_cli}, "eval_ladder CLI must write output or exit nonzero.", "guarded" if has_cli else "unresolved")


def _performance_record_from_ladder(ladder: dict[str, Any] | None, field: str, requirement: str) -> dict[str, Any]:
    if not ladder:
        return _record("BLOCKED", {"ladder_present": False}, requirement, "blocked")
    value = ladder.get("decision", {}).get(field)
    return _record("WARN", {"measured_value": value}, requirement, "blocked")


def _adapter_weak_record(ladder: dict[str, Any] | None) -> dict[str, Any]:
    if not ladder:
        return _record("BLOCKED", {"ladder_present": False}, "Adapter weakness must be measured, not inferred.", "blocked")
    decision = ladder.get("decision", {})
    adapter = decision.get("adapter_exact_match")
    base = decision.get("base_exact_match")
    weak = adapter is not None and base is not None and float(adapter) <= float(base)
    return _record("WARN" if weak else "PASS", {"adapter_exact_match": adapter, "base_exact_match": base, "adapter_worse_or_equal": weak}, "v2a_150 stays blocked if adapter does not beat base.", "blocked" if weak else "guarded")


def _check_combined_measured(root: Path, reports_dir: str | Path) -> dict[str, Any]:
    invalid_pass = _combined_pass_without_sources(root / reports_dir)
    combined_reports = sorted(str(path) for path in (root / reports_dir).glob("*_combined.json")) if (root / reports_dir).exists() else []
    blocked = []
    for path in combined_reports:
        report = _read_optional(path)
        if report and report.get("status") == "BLOCKED":
            blocked.append(path)
    status = "FAIL" if invalid_pass else ("BLOCKED" if blocked or not combined_reports else "PASS")
    return _record(status, {"combined_reports": combined_reports, "blocked_reports": blocked, "invalid_pass_reports": invalid_pass}, "Combined must be measured with solver+adapter source_breakdown or explicitly BLOCKED.", "blocked" if status == "BLOCKED" else ("guarded" if status == "PASS" else "unresolved"))


def _solver_score_record(ladder: dict[str, Any] | None) -> dict[str, Any]:
    if not ladder:
        return _record("BLOCKED", {"ladder_present": False}, "Solver raw exact-match must be reported separately from answerable accuracy.", "blocked")
    solver = ladder.get("decision", {}).get("solver_exact_match")
    below = solver is None or float(solver) < 0.93
    return _record("WARN" if below else "PASS", {"solver_exact_match": solver, "threshold": 0.93}, "Below-0.93 raw solver score is evidence gap, not leaderboard claim.", "blocked" if below else "guarded")


def _old_v1_family_record(family: str, failure_v2: dict[str, Any] | None) -> dict[str, Any]:
    if not failure_v2:
        return _record("BLOCKED", {"failure_report_v2_present": False, "family": family}, "Use V2 answerable_wrong before treating old V1 family as repair target.", "blocked")
    bucket = failure_v2.get("by_family", {}).get(family)
    wrong = 0 if not bucket else int(bucket.get("answerable_wrong", 0))
    status = "WARN" if wrong > 0 else "PASS"
    repair = "unresolved" if wrong > 0 else "fixed"
    label = "true_solver_repair_target" if wrong > 0 else "resolved_by_policy_aware_analysis"
    return _record(status, {"family": family, "answerable_wrong": wrong, "v2_status": label}, "Old V1 ABSTAIN-contaminated family blockers require V2 confirmation.", repair)


def _check_failure_report_v2_guard(root: Path, failure_v2: dict[str, Any] | None) -> dict[str, Any]:
    script_exists = (root / "kaggle_anti086/training/day2_failure_report_v2.py").exists()
    tests_text = _read_text(root / "tests/test_day2_failure_report_v2.py")
    tests_cover = "abstain_correct" in tests_text and "abstain_unsafe_answer" in tests_text
    report_ok = bool(failure_v2 and failure_v2.get("schema_version") == 2)
    ok = script_exists and tests_cover and report_ok
    return _record("PASS" if ok else "FAIL", {"script_exists": script_exists, "tests_cover_abstain_policy": tests_cover, "report_schema_v2": report_ok}, "V1 misclassification must be guarded by policy-aware V2 report and tests.", "fixed" if ok else "unresolved")


def _combined_pass_without_sources(path: str | Path) -> list[str]:
    root = Path(path)
    bad = []
    if root.is_file():
        paths = [root]
    elif root.exists():
        paths = sorted(root.glob("*_combined.json"))
    else:
        paths = []
    for report_path in paths:
        report = _read_optional(report_path)
        if not report or report.get("status") != "PASS":
            continue
        breakdown = report.get("source_breakdown", {})
        if int(breakdown.get("solver", 0)) <= 0 or int(breakdown.get("adapter", 0)) <= 0:
            bad.append(str(report_path))
    return bad


def _summarize(modes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    counts = {"fixed_count": 0, "guarded_count": 0, "blocked_count": 0, "unresolved_count": 0}
    critical = []
    for key, item in modes.items():
        repair = item["repair_status"]
        if repair == "fixed":
            counts["fixed_count"] += 1
        elif repair == "guarded":
            counts["guarded_count"] += 1
        elif repair == "blocked":
            counts["blocked_count"] += 1
            critical.append(key)
        else:
            counts["unresolved_count"] += 1
            critical.append(key)
    return counts | {"critical_blockers": critical}


def _next_action(summary: dict[str, Any]) -> str:
    if summary["unresolved_count"]:
        return "fix_unresolved_day2_infrastructure_guards_then_rerun_audit"
    if summary["blocked_count"]:
        return "resolve_blocked_kaggle_evidence_paths_then_rerun_day2"
    return "rerun_day2_solver_only_and_ladder_evidence"


def _record(status: str, evidence: dict[str, Any], required_guard: str, repair_status: str) -> dict[str, Any]:
    return {"status": status, "evidence": evidence, "required_guard": required_guard, "repair_status": repair_status}


def _read_optional(path: str | Path) -> dict[str, Any] | None:
    target = Path(path)
    if not target.exists():
        return None
    try:
        return read_json(target)
    except Exception as exc:
        return {"_read_error": f"{type(exc).__name__}:{exc}"}


def _read_text(path: str | Path) -> str:
    target = Path(path)
    if not target.exists():
        return ""
    return target.read_text(encoding="utf-8", errors="replace")


def _file_contains(path: str | Path, needle: str) -> bool:
    return needle in _read_text(path)


def _tracked_paths(root: Path, pattern: str) -> list[str]:
    try:
        result = subprocess.run(["git", "ls-files", f"*{pattern}*"], cwd=root, text=True, capture_output=True, check=False)
    except Exception:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Day2 21-failure hardening audit.")
    parser.add_argument("--out", default="artifacts/sprint11/day2_21_failure_hardening_audit.json")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--failure-report-v2", default="artifacts/sprint11/day2_failure_report_v2.json")
    parser.add_argument("--eval-ladder-report", default="artifacts/sprint11/day2_eval_ladder_report.json")
    parser.add_argument("--manifest", default="artifacts/sprint11/day2_eval_manifest.json")
    parser.add_argument("--reports-dir", default="artifacts/sprint11/day2_reports")
    args = parser.parse_args(argv)
    report = build_hardening_audit(
        repo_root=args.repo_root,
        failure_report_v2_path=args.failure_report_v2,
        eval_ladder_report_path=args.eval_ladder_report,
        manifest_path=args.manifest,
        reports_dir=args.reports_dir,
    )
    write_json_checked(args.out, report, field_name="day2_21_failure_hardening_audit")
    print(json.dumps({"status": report["status"], "out": args.out, "summary": report["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
