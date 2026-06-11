from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable, Mapping, Sequence


TARGET_FAMILIES: tuple[str, ...] = (
    "custom_numeral",
    "symbol_mapping",
    "bit_manipulation",
    "char_cipher",
    "word_cipher",
    "permutation_sorting",
    "gravity_numeric",
    "unit_conversion",
    "numeric_formula",
    "equation_operator",
    "sequence_pattern",
)

CURRENT_CORRECT_DEFAULT = 1383
TOTAL_ROWS_DEFAULT = 1792
TARGET_CORRECT_0_93_DEFAULT = 1667

ROW_ID_KEYS = ("row_id", "id", "problem_id")
FAMILY_KEYS = ("family", "task_family", "category")
PROMPT_KEYS = ("prompt", "question", "input", "problem")
GOLD_KEYS = ("expected", "gold", "answer", "label", "target")
PREDICTION_KEYS = (
    "prediction",
    "output",
    "solver_output",
    "selected_answer",
    "raw_output",
    "extracted_answer",
    "normalized_answer",
    "solver_candidate",
)
CORRECT_KEYS = ("correct", "is_correct", "exact_match")

PREDICTION_GLOB = "artifacts/sprint11/day2_predictions/*solver_only_predictions.jsonl"
JSON_ARTIFACTS: tuple[str, ...] = (
    "artifacts/sprint11/day2_abstain_fallback_mining.json",
    "artifacts/sprint11/day2_unsafe_abstain_forensic_audit.json",
    "artifacts/sprint11/day2_final_decision_report.json",
    "artifacts/sprint11/day3_equation_operator_repair_report.json",
    "artifacts/sprint11/day3_equation_operator_post_audit.json",
    "artifacts/sprint11/day3_equation_operator_quarantine_report.json",
    "artifacts/sprint11/day3_sequence_pattern_repair_report.json",
    "artifacts/sprint11/day3_sequence_pattern_post_audit.json",
)

SYNTHETIC_SAFE_FAMILIES: frozenset[str] = frozenset(
    {
        "custom_numeral",
        "symbol_mapping",
        "bit_manipulation",
        "char_cipher",
        "word_cipher",
        "permutation_sorting",
        "unit_conversion",
        "equation_operator",
        "sequence_pattern",
    }
)

LOW_RISK_SYNTHETIC_FAMILIES: frozenset[str] = frozenset(
    {
        "symbol_mapping",
        "bit_manipulation",
        "char_cipher",
        "unit_conversion",
    }
)

MEDIUM_RISK_SYNTHETIC_FAMILIES: frozenset[str] = frozenset(
    {
        "word_cipher",
        "numeric_formula",
        "permutation_sorting",
        "custom_numeral",
    }
)

HIGH_RISK_SYNTHETIC_FAMILIES: frozenset[str] = frozenset(
    {
        "gravity_numeric",
        "equation_operator",
        "sequence_pattern",
    }
)


@dataclass(frozen=True)
class RowAssessment:
    row_id: str
    family: str
    prompt: str
    gold: str | None
    prediction: str | None
    row_type: str
    current_outcome: str
    source: str


def read_json(path: Path, parse_warnings: list[str] | None = None) -> dict[str, Any] | None:
    if not path.exists():
        _warn(parse_warnings, f"Missing JSON artifact: {path}")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _warn(parse_warnings, f"Malformed JSON artifact {path}: {exc}")
        return None
    if not isinstance(payload, dict):
        _warn(parse_warnings, f"JSON artifact {path} is not an object.")
        return None
    return payload


def read_jsonl(path: Path, parse_warnings: list[str] | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        _warn(parse_warnings, f"Missing JSONL artifact: {path}")
        return []
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        _warn(parse_warnings, f"Unreadable JSONL artifact {path}: {exc}")
        return []
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            _warn(parse_warnings, f"Malformed JSONL line {path}:{index}: {exc.msg}")
            continue
        if not isinstance(payload, dict):
            _warn(parse_warnings, f"JSONL line {path}:{index} is not an object.")
            continue
        rows.append(payload)
    return rows


def safe_get(row: Mapping[str, Any], candidate_keys: Sequence[str], default: Any = None) -> Any:
    for key in candidate_keys:
        if key in row:
            return row[key]
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        for key in candidate_keys:
            if key in metadata:
                return metadata[key]
    return default


def normalize_family(value: Any) -> str:
    text = "" if value is None else str(value)
    normalized = re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")
    aliases = {
        "custom_numerals": "custom_numeral",
        "custom_numeral_system": "custom_numeral",
        "symbols": "symbol_mapping",
        "symbol_map": "symbol_mapping",
        "bit": "bit_manipulation",
        "binary": "bit_manipulation",
        "character_cipher": "char_cipher",
        "letter_cipher": "char_cipher",
        "words_cipher": "word_cipher",
        "perm_sort": "permutation_sorting",
        "sorting": "permutation_sorting",
        "gravity": "gravity_numeric",
        "unit": "unit_conversion",
        "units": "unit_conversion",
        "formula": "numeric_formula",
        "numeric_formulas": "numeric_formula",
        "equation": "equation_operator",
        "equation_ops": "equation_operator",
        "operator": "equation_operator",
        "sequence": "sequence_pattern",
        "sequences": "sequence_pattern",
    }
    return aliases.get(normalized, normalized)


def normalize_answer(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def is_abstain(value: Any) -> bool:
    if value is None:
        return True
    text = normalize_answer(value)
    return text == "" or text.upper() in {"ABSTAIN", "NONE", "NULL", "N/A", "NA", "NO_ANSWER"}


def is_empty_or_abstain_output(value: Any) -> bool:
    return is_abstain(value)


def extract_row_id(row: Mapping[str, Any]) -> str:
    value = safe_get(row, ROW_ID_KEYS)
    return normalize_answer(value) if value is not None else ""


def extract_prompt(row: Mapping[str, Any]) -> str:
    return normalize_answer(safe_get(row, PROMPT_KEYS, ""))


def extract_gold(row: Mapping[str, Any]) -> str | None:
    value = safe_get(row, GOLD_KEYS)
    if value is None:
        return None
    return normalize_answer(value)


def extract_prediction(row: Mapping[str, Any]) -> str | None:
    value = safe_get(row, PREDICTION_KEYS)
    if value is None:
        return None
    return normalize_answer(value)


def extract_correct_flag(row: Mapping[str, Any]) -> bool | None:
    value = safe_get(row, CORRECT_KEYS)
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "t", "yes", "y", "1"}:
            return True
        if lowered in {"false", "f", "no", "n", "0"}:
            return False
    return None


def classify_row(row: Mapping[str, Any], *, source: str = "unknown") -> RowAssessment:
    family = normalize_family(safe_get(row, FAMILY_KEYS, "unknown"))
    row_id = extract_row_id(row)
    prompt = extract_prompt(row)
    gold = extract_gold(row)
    prediction = extract_prediction(row)
    correct_flag = extract_correct_flag(row)

    if _gold_key_present(row) and gold is not None and is_abstain(gold):
        row_type = "expected_abstain"
    elif gold is None:
        row_type = "expected_abstain" if _source_indicates_abstain(row, source) else "unknown_gold"
    elif not is_abstain(gold):
        row_type = "concrete_answer"
    else:
        row_type = "unknown_gold"

    prediction_is_abstain = is_empty_or_abstain_output(prediction)
    if row_type == "concrete_answer":
        if correct_flag is True or (prediction is not None and normalize_answer(prediction) == normalize_answer(gold)):
            current_outcome = "correct"
        elif prediction_is_abstain:
            current_outcome = "wrong_or_abstain"
        else:
            current_outcome = "wrong"
    elif row_type == "expected_abstain":
        current_outcome = "safe_abstain" if prediction_is_abstain else "unsafe_answer_on_abstain"
    else:
        current_outcome = "unknown"

    return RowAssessment(
        row_id=row_id,
        family=family,
        prompt=prompt,
        gold=gold,
        prediction=prediction,
        row_type=row_type,
        current_outcome=current_outcome,
        source=source,
    )


def detect_prompt_patterns(prompt: str, family: str = "") -> list[str]:
    text = prompt or ""
    lowered = text.lower()
    patterns: list[str] = []
    if "->" in text or "=>" in text:
        patterns.append("contains_arrow")
    if r"\boxed" in text:
        patterns.append("contains_box")
    if "sequence" in lowered or re.search(r"\bnext\b|\bterm\b", lowered):
        patterns.append("contains_sequence_word")
    if any(token in text for token in ("@", "#", "⊙", "?")) or " op " in f" {lowered} ":
        patterns.append("contains_equation_operator_symbols")
    if re.search(r"\b[01]{4,}\b", text):
        patterns.append("contains_binary_digits")
    if re.search(r"[⊙⊕⊗★◆◇■□▲△●○]", text) or family == "custom_numeral":
        patterns.append("contains_custom_symbols")
    if re.search(r"\b(cm|mm|km|kg|g|m/s|mph|hours?|minutes?|seconds?|liters?|metres?|meters?)\b", lowered):
        patterns.append("contains_unit_words")
    if re.search(r"\b(cipher|decode|encode|encrypted|shift|caesar|word code|letter code)\b", lowered):
        patterns.append("contains_cipher_words")
    if re.search(r"\b(sort|permutation|ascending|descending|order)\b", lowered):
        patterns.append("contains_sorting_words")
    if re.search(r"\bf\(|formula|mapping|maps to|rule\b", lowered):
        patterns.append("contains_formula_mapping")
    if re.search(r"\b(?:sequence|equation|custom numeral|custom_numeral|[a-z_ ]+)\s+\d+\s*:", lowered):
        patterns.append("indexed_placeholder_pattern")
    if re.search(r"\b(example|query|q:|a:)\b", lowered):
        patterns.append("examples_query_pattern")
    if "|" in text or re.search(r"\btable\b", lowered):
        patterns.append("table_pattern")
    if len(text.split()) > 35:
        patterns.append("story_pattern")
    if len(text.split()) <= 8:
        patterns.append("minimal_pattern")
    return patterns or ["unclassified_pattern"]


def build_audit_report(root: Path, *, out: Path | None = None) -> dict[str, Any]:
    root = Path(root)
    parse_warnings: list[str] = []
    missing_artifacts: list[str] = []
    source_counts: dict[str, dict[str, int]] = defaultdict(lambda: {family: 0 for family in TARGET_FAMILIES})
    artifact_risks: dict[str, str] = {}
    assessments: list[RowAssessment] = []

    start_state_path = root / "artifacts/sprint11/day1_teacher_start_state.json"
    start_state = read_json(start_state_path, parse_warnings)
    start_state_present = start_state is not None
    if not start_state_path.exists():
        missing_artifacts.append(_rel(root, start_state_path))

    prediction_paths = sorted(root.glob(PREDICTION_GLOB))
    if not prediction_paths:
        missing_artifacts.append(PREDICTION_GLOB)
    for path in prediction_paths:
        rows = read_jsonl(path, parse_warnings)
        for row in rows:
            assessment = classify_row(row, source="prediction_jsonl")
            if assessment.family in TARGET_FAMILIES:
                assessments.append(assessment)
                source_counts["prediction_jsonl"][assessment.family] += 1

    json_sources_found = 0
    for artifact in JSON_ARTIFACTS:
        path = root / artifact
        if not path.exists():
            missing_artifacts.append(artifact)
            continue
        payload = read_json(path, parse_warnings)
        if payload is None:
            continue
        json_sources_found += 1
        source_name = _source_name_from_path(path)
        source_counts[source_name].update(_extract_family_counts(payload))
        _merge_artifact_risks(artifact_risks, _extract_family_risks(payload))
        # JSON reports provide cross-artifact evidence and examples, but the
        # primary row census is the Day2 solver-only prediction JSONL set.
        # Counting report snippets as rows would double-count the same local
        # probes and can turn expected-ABSTAIN examples into fake recovery pool.

    families = _build_family_reports(assessments, source_counts, artifact_risks)
    consistency = _build_consistency(source_counts)
    repo_state = inspect_repo_state(root)
    global_math = _build_global_math(start_state, families)

    local_candidates = sorted(
        family for family, report in families.items() if report["safe_to_attempt_local_recovery"]
    )
    synthetic_candidates = sorted(
        family for family, report in families.items() if report["synthetic_generation_allowed"]
    )
    evidence_backed_synthetic = _evidence_backed_synthetic_candidates(families)
    theoretical_synthetic = _theoretical_synthetic_candidates(families, evidence_backed_synthetic)
    blocked_local = sorted(
        family
        for family, report in families.items()
        if not report["safe_to_attempt_local_recovery"] and report["teacher_priority"] == "BLOCKED"
    )
    high_risk = sorted(family for family, report in families.items() if report["risk_level"] == "HIGH")

    found_any_evidence = bool(prediction_paths) or json_sources_found > 0
    count_mismatch = any(item["count_mismatch"] for item in consistency.values())
    protected_diffs = any(repo_state["protected_file_diffs"].values())
    if not found_any_evidence:
        status = "FAIL"
    elif missing_artifacts or parse_warnings or count_mismatch or protected_diffs:
        status = "WARN"
    else:
        status = "PASS"

    phase2_teacher_safe = bool(evidence_backed_synthetic or theoretical_synthetic) and status != "FAIL"
    phase2_interpretation = _build_phase2_interpretation(
        global_math=global_math,
        evidence_backed=evidence_backed_synthetic,
        theoretical=theoretical_synthetic,
        status=status,
    )
    priority_order = _build_phase2_teacher_priority_order(families)
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_by": "DAY1_FAMILY_RECOVERY_AUDIT",
        "status": status,
        "start_state_present": start_state_present,
        "missing_artifacts": sorted(set(missing_artifacts)),
        "parse_warnings": sorted(set(parse_warnings)),
        "repo_state": repo_state,
        "families": families,
        "global_recovery_math": global_math,
        "local_recovery_candidate_families": local_candidates,
        "synthetic_teacher_candidate_families": synthetic_candidates,
        "evidence_backed_synthetic_teacher_candidates": evidence_backed_synthetic,
        "theoretical_synthetic_teacher_candidates": theoretical_synthetic,
        "blocked_local_recovery_families": blocked_local,
        "high_risk_families": high_risk,
        "cross_artifact_consistency": consistency,
        "phase2_interpretation": phase2_interpretation,
        "phase2_teacher_priority_order": priority_order,
        "decision": {
            "phase1_complete": True,
            "safe_to_start_phase2_local_recovery": False,
            "safe_to_start_phase2_teacher_bank": phase2_teacher_safe,
            "safe_to_start_phase2_verified_data_factory": phase2_teacher_safe,
            "safe_to_train_lora": False,
            "safe_to_package": False,
            "safe_to_submit": False,
            "next_phase": "PHASE2_DETERMINISTIC_TEACHER_BANK",
        },
        "blocked_actions": {
            "train_lora": "blocked_until_verified_data_factory_passes",
            "v2a_150": "blocked",
            "package": "blocked",
            "submission": "blocked",
            "leaderboard_claim": "blocked",
        },
        "model_results_faked": False,
        "leaderboard_claim": False,
        "no_0_93_evidence": True,
        "no_0_95_evidence": True,
    }
    if protected_diffs:
        report["parse_warnings"].append("Protected files have existing diffs; auditor did not modify them.")
        report["parse_warnings"] = sorted(set(report["parse_warnings"]))

    if out is not None:
        write_report(report, out)
    return report


def inspect_repo_state(root: Path) -> dict[str, Any]:
    if not (root / ".git").exists():
        return {
            "branch": "UNKNOWN_NOT_GIT_REPO",
            "dirty_files": [],
            "protected_file_diffs": {
                "adapter_scripts": False,
                "router.py": False,
                "solver_ensemble.py": False,
                "submission_scripts": False,
                "verifier.py": False,
            },
            "notes": ["git status unavailable: not a git repository at audit root."],
        }

    status = _run_git(root, ["status", "--short"])
    diff = _run_git(root, ["diff", "--name-only"])
    branch = _run_git(root, ["rev-parse", "--abbrev-ref", "HEAD"])

    dirty_files = _parse_dirty_files(status[1]) if status[0] == 0 else []
    diff_files = [line.strip().replace("\\", "/") for line in diff[1].splitlines() if line.strip()] if diff[0] == 0 else []
    protected_paths = {
        "router.py": "kaggle_anti086/solvers/router.py",
        "solver_ensemble.py": "kaggle_anti086/solvers/solver_ensemble.py",
        "verifier.py": "kaggle_anti086/solvers/verifier.py",
        "submission_scripts": "submission",
        "adapter_scripts": "adapter",
    }
    protected = {
        label: any(_matches_protected(path, marker) for path in diff_files)
        for label, marker in protected_paths.items()
    }
    notes: list[str] = []
    if status[0] != 0:
        notes.append(f"git status unavailable: {status[2] or status[1]}")
    if diff[0] != 0:
        notes.append(f"git diff unavailable: {diff[2] or diff[1]}")
    if any(protected.values()):
        notes.append("Protected files have existing diffs; auditor did not modify them.")
    return {
        "branch": branch[1].strip() if branch[0] == 0 else "UNKNOWN_NOT_GIT_REPO",
        "dirty_files": dirty_files,
        "protected_file_diffs": protected,
        "notes": sorted(notes),
    }


def write_report(report: Mapping[str, Any], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Day1 Phase1 family recovery auditor.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, default=Path("artifacts/sprint11/day1_family_recovery_audit.json"))
    args = parser.parse_args(argv)
    report = build_audit_report(args.root, out=args.out)
    print(json.dumps({"status": report["status"], "out": str(args.out)}, sort_keys=True))
    return 0 if report["status"] in {"PASS", "WARN", "FAIL"} else 1


def _build_family_reports(
    assessments: Sequence[RowAssessment],
    source_counts: Mapping[str, Mapping[str, int]],
    artifact_risks: Mapping[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[RowAssessment]] = {family: [] for family in TARGET_FAMILIES}
    for assessment in assessments:
        if assessment.family in grouped:
            grouped[assessment.family].append(assessment)

    reports: dict[str, dict[str, Any]] = {}
    for family in TARGET_FAMILIES:
        rows = grouped[family]
        row_type_counts = Counter(row.row_type for row in rows)
        outcome_counts = Counter(row.current_outcome for row in rows)
        pattern_counts: Counter[str] = Counter()
        examples_by_pattern: dict[str, list[dict[str, str]]] = defaultdict(list)
        examples = {
            "concrete_wrong_or_abstain": [],
            "expected_abstain": [],
            "unsafe_answer_on_abstain": [],
            "unknown_gold": [],
        }

        for row in rows:
            for pattern in detect_prompt_patterns(row.prompt, family):
                pattern_counts[pattern] += 1
                if len(examples_by_pattern[pattern]) < 3:
                    examples_by_pattern[pattern].append(_row_example(row))
            if row.row_type == "concrete_answer" and row.current_outcome in {"wrong", "wrong_or_abstain"}:
                _append_capped(examples["concrete_wrong_or_abstain"], _row_example(row))
            if row.row_type == "expected_abstain":
                _append_capped(examples["expected_abstain"], _row_example(row))
            if row.current_outcome == "unsafe_answer_on_abstain":
                _append_capped(examples["unsafe_answer_on_abstain"], _row_example(row))
            if row.row_type == "unknown_gold":
                _append_capped(examples["unknown_gold"], _row_example(row))

        concrete_rows = row_type_counts["concrete_answer"]
        expected_abstain_rows = row_type_counts["expected_abstain"]
        unknown_gold_rows = row_type_counts["unknown_gold"]
        current_wrong = outcome_counts["wrong"]
        current_abstain_on_concrete = outcome_counts["wrong_or_abstain"]
        current_wrong_or_abstain = current_wrong + current_abstain_on_concrete
        unsafe_abstain = outcome_counts["unsafe_answer_on_abstain"]
        safe_local = concrete_rows > 0 and current_wrong_or_abstain > 0
        estimated_pool = current_wrong_or_abstain if safe_local else 0

        if family in {"equation_operator", "sequence_pattern"} and concrete_rows == 0:
            safe_local = False
            estimated_pool = 0
        risk_level = _risk_level(
            family,
            unsafe_abstain,
            unknown_gold_rows,
            len(rows),
            None if artifact_risks is None else artifact_risks.get(family),
        )
        local_recovery_risk = _local_recovery_risk(
            expected_abstain_rows=expected_abstain_rows,
            unsafe_abstain=unsafe_abstain,
            current_wrong_or_abstain=current_wrong_or_abstain,
            artifact_risk=None if artifact_risks is None else artifact_risks.get(family),
        )
        synthetic_teacher_risk = _synthetic_teacher_risk(
            family=family,
            concrete_rows=concrete_rows,
            current_correct=outcome_counts["correct"],
        )
        synthetic_safe = _synthetic_generation_allowed(family)
        priority = _teacher_priority(family, estimated_pool, safe_local, expected_abstain_rows, concrete_rows)
        local_router_allowed = safe_local and estimated_pool > 0 and local_recovery_risk == "LOW"

        reports[family] = {
            "rows_seen": len(rows),
            "expected_abstain_rows": expected_abstain_rows,
            "concrete_answer_rows": concrete_rows,
            "unknown_gold_rows": unknown_gold_rows,
            "current_correct": outcome_counts["correct"],
            "current_wrong": current_wrong,
            "current_abstain_on_concrete": current_abstain_on_concrete,
            "safe_abstain_rows": outcome_counts["safe_abstain"],
            "unsafe_answer_on_abstain_rows": unsafe_abstain,
            "current_wrong_or_abstain": current_wrong_or_abstain,
            "estimated_local_recovery_pool": estimated_pool,
            "safe_to_attempt_local_recovery": safe_local,
            "safe_to_generate_synthetic_data": synthetic_safe,
            "teacher_priority": priority,
            "risk_level": risk_level,
            "local_recovery_risk": local_recovery_risk,
            "synthetic_teacher_risk": synthetic_teacher_risk,
            "local_router_integration_allowed": local_router_allowed,
            "synthetic_generation_allowed": synthetic_safe,
            "synthetic_generation_reason": _synthetic_generation_reason(
                family=family,
                allowed=synthetic_safe,
                risk=synthetic_teacher_risk,
                concrete_rows=concrete_rows,
                current_correct=outcome_counts["correct"],
            ),
            "local_recovery_reason": _local_recovery_reason(
                expected_abstain_rows=expected_abstain_rows,
                current_wrong_or_abstain=current_wrong_or_abstain,
                unsafe_abstain=unsafe_abstain,
                artifact_risk=None if artifact_risks is None else artifact_risks.get(family),
                local_router_allowed=local_router_allowed,
            ),
            "recommended_next_action": _recommended_action(family, safe_local, synthetic_safe, risk_level, priority),
            "dominant_prompt_patterns": [name for name, _count in pattern_counts.most_common(5)],
            "pattern_counts": dict(sorted(pattern_counts.items())),
            "examples_by_pattern": {key: value for key, value in sorted(examples_by_pattern.items())},
            "examples": examples,
            "counts_by_source": {
                source: counts.get(family, 0)
                for source, counts in sorted(source_counts.items())
            },
        }
    return reports


def _build_global_math(start_state: Mapping[str, Any] | None, families: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    current_correct = int(_safe_number(start_state, "current_correct", CURRENT_CORRECT_DEFAULT))
    total_rows = int(_safe_number(start_state, "total_rows", TOTAL_ROWS_DEFAULT))
    target_correct = int(_safe_number(start_state, "target_correct_0_93", TARGET_CORRECT_0_93_DEFAULT))
    current_accuracy = current_correct / total_rows if total_rows else 0.0
    pool_total = sum(int(report["estimated_local_recovery_pool"]) for report in families.values())
    estimated_correct = current_correct + pool_total
    estimated_accuracy = estimated_correct / total_rows if total_rows else 0.0
    remaining_gap = max(0, target_correct - estimated_correct)
    return {
        "current_correct": current_correct,
        "total_rows": total_rows,
        "current_accuracy": current_accuracy,
        "target_correct_0_93": target_correct,
        "additional_needed_0_93": max(0, target_correct - current_correct),
        "estimated_local_recovery_pool_total": pool_total,
        "estimated_correct_if_all_safe_local_recovery_solved": estimated_correct,
        "estimated_accuracy_if_all_safe_local_recovery_solved": estimated_accuracy,
        "remaining_gap_to_0_93_after_safe_local_recovery": remaining_gap,
        "can_reach_0_93_with_safe_local_recovery_only": estimated_correct >= target_correct,
    }


def _evidence_backed_synthetic_candidates(families: Mapping[str, Mapping[str, Any]]) -> list[str]:
    candidates = []
    for family, report in families.items():
        concrete_rows = int(report["concrete_answer_rows"])
        current_correct = int(report["current_correct"])
        if concrete_rows > 0 and current_correct == concrete_rows and report["synthetic_generation_allowed"]:
            candidates.append(family)
    return sorted(candidates)


def _theoretical_synthetic_candidates(
    families: Mapping[str, Mapping[str, Any]],
    evidence_backed: Sequence[str],
) -> list[str]:
    evidence_set = set(evidence_backed)
    candidates = []
    for family, report in families.items():
        if family in evidence_set:
            continue
        if report["synthetic_generation_allowed"]:
            candidates.append(family)
    return sorted(candidates)


def _build_phase2_interpretation(
    *,
    global_math: Mapping[str, Any],
    evidence_backed: Sequence[str],
    theoretical: Sequence[str],
    status: str,
) -> dict[str, Any]:
    teacher_open = bool(evidence_backed or theoretical) and status != "FAIL"
    pool_total = int(global_math["estimated_local_recovery_pool_total"])
    local_status = "BLOCKED_NO_SAFE_POOL" if pool_total == 0 else "REVIEW_REQUIRED"
    return {
        "local_recovery_route_status": local_status,
        "teacher_distillation_route_status": "OPEN" if teacher_open else "BLOCKED_NO_CANDIDATES",
        "local_recovery_summary": (
            f"Safe local recovery pool is {pool_total}; local router integration is blocked for Phase 1."
        ),
        "teacher_distillation_summary": (
            f"{len(evidence_backed)} evidence-backed and {len(theoretical)} theoretical synthetic teacher "
            "families are available for Phase 2 interpretation."
        ),
        "reason": (
            "Concrete rows are already solved; remaining rows are expected-ABSTAIN or high-risk. "
            "Use solved deterministic families as verified synthetic teachers, not local router recovery."
        ),
        "phase2_allowed_work": [
            "PHASE2_DETERMINISTIC_TEACHER_BANK",
            "PHASE2_VERIFIED_SYNTHETIC_DATA_FACTORY",
        ],
        "phase2_blocked_work": [
            "local_router_integration",
            "lora_training",
            "adapter_packaging",
            "submission",
            "leaderboard_claim",
        ],
    }


def _build_phase2_teacher_priority_order(families: Mapping[str, Mapping[str, Any]]) -> list[dict[str, str]]:
    priority_seed = {
        "symbol_mapping": ("P0", 0, "verified_symbol_mapping_teacher"),
        "bit_manipulation": ("P0", 1, "verified_bit_manipulation_teacher"),
        "char_cipher": ("P0", 2, "verified_character_cipher_teacher"),
        "unit_conversion": ("P0", 3, "verified_unit_conversion_teacher"),
        "numeric_formula": ("P0", 4, "numeric_formula_safe_strict_ambiguity_rejection"),
        "word_cipher": ("P1", 5, "word_cipher_safe_subset_teacher"),
        "custom_numeral": ("P1", 6, "custom_numeral_synthetic_teacher"),
        "permutation_sorting": ("P1", 7, "permutation_sorting_synthetic_teacher"),
        "gravity_numeric": ("P2", 8, "gravity_numeric_explicit_constants_only"),
        "equation_operator": ("P2", 9, "equation_operator_synthetic_only_no_local_recovery"),
        "sequence_pattern": ("P2", 10, "sequence_pattern_synthetic_only_no_local_recovery"),
    }
    items: list[tuple[int, int, str, dict[str, str]]] = []
    for family, report in families.items():
        if not report["synthetic_generation_allowed"]:
            continue
        priority, order, action = priority_seed.get(family, ("P2", 99, f"{family}_synthetic_review"))
        solved = int(report["current_correct"])
        concrete = int(report["concrete_answer_rows"])
        expected_abstain = int(report["expected_abstain_rows"])
        if concrete > 0 and solved == concrete:
            reason = (
                f"{solved} concrete rows solved; deterministic {family} behavior is locally evidenced; "
                f"synthetic teacher risk is {report['synthetic_teacher_risk']}."
            )
        else:
            reason = (
                f"{expected_abstain} expected-ABSTAIN local rows and no concrete solved local teacher rows; "
                "synthetic-only work allowed with no local recovery claim."
            )
        items.append(
            (
                {"P0": 0, "P1": 1, "P2": 2}[priority],
                order,
                family,
                {
                    "family": family,
                    "priority": priority,
                    "reason": reason,
                    "recommended_phase2_action": action,
                },
            )
        )
    return [entry[3] for entry in sorted(items, key=lambda entry: (entry[0], entry[1], entry[2]))]


def _build_consistency(source_counts: Mapping[str, Mapping[str, int]]) -> dict[str, dict[str, Any]]:
    consistency: dict[str, dict[str, Any]] = {}
    source_aliases = {
        "prediction_jsonl": "prediction_jsonl",
        "day2_abstain_fallback_mining": "abstain_fallback_mining",
        "day2_unsafe_abstain_forensic_audit": "unsafe_abstain_forensic",
        "day2_final_decision_report": "day2_final_decision",
    }
    for family in TARGET_FAMILIES:
        compact_counts: dict[str, int | None] = {
            "prediction_jsonl": None,
            "abstain_fallback_mining": None,
            "unsafe_abstain_forensic": None,
            "day2_final_decision": None,
            "day3_reports": None,
        }
        day3_total = 0
        day3_seen = False
        for source, counts in source_counts.items():
            value = counts.get(family, 0)
            if source.startswith("day3_"):
                day3_total += value
                day3_seen = True
            elif source in source_aliases:
                compact_counts[source_aliases[source]] = value
        compact_counts["day3_reports"] = day3_total if day3_seen else None
        non_null = [value for value in compact_counts.values() if value is not None]
        mismatch = len(set(non_null)) > 1 if len(non_null) > 1 else False
        notes = []
        if mismatch:
            notes.append(
                "Counts differ across artifacts; likely causes include different artifact subset, "
                "abstain-only source, family naming mismatch, artifact missing, or previous Day3 reclassification."
            )
        consistency[family] = {
            "counts_by_source": compact_counts,
            "count_mismatch": mismatch,
            "count_mismatch_notes": notes,
        }
    return consistency


def _extract_family_counts(payload: Any) -> dict[str, int]:
    counts = {family: 0 for family in TARGET_FAMILIES}
    explicit_seen: set[str] = set()
    for key, value in _walk_items(payload):
        family = normalize_family(key)
        if family in counts and isinstance(value, int) and not isinstance(value, bool):
            counts[family] = max(counts[family], value)
            explicit_seen.add(family)
    for item in _walk_mappings(payload):
        value_family = normalize_family(safe_get(item, FAMILY_KEYS, ""))
        if value_family in counts:
            count_value = _first_int(
                item,
                (
                    "rows_seen",
                    "rows",
                    "count",
                    "total",
                    "n",
                    "remaining_wrong_or_abstain",
                    "remaining_plausible_recoverable",
                    "previous_plan_rows",
                    "equation_operator_previous_plan_rows",
                    "sequence_pattern_previous_plan_rows",
                ),
            )
            if count_value is not None:
                counts[value_family] = max(counts[value_family], count_value)
                explicit_seen.add(value_family)
        for nested_key, nested_value in item.items():
            nested_family = normalize_family(nested_key)
            if nested_family in counts and isinstance(nested_value, int) and not isinstance(nested_value, bool):
                counts[nested_family] = max(counts[nested_family], nested_value)
                explicit_seen.add(nested_family)

    for key, value in _walk_items(payload):
        family = normalize_family(key)
        if family in counts and isinstance(value, int) and not isinstance(value, bool):
            counts[family] = max(counts[family], value)
            explicit_seen.add(family)
    for row in _extract_row_dicts(payload):
        family = normalize_family(safe_get(row, FAMILY_KEYS, ""))
        if family in counts and family not in explicit_seen:
            counts[family] += 1
    return counts


def _extract_family_risks(payload: Any) -> dict[str, str]:
    risks: dict[str, str] = {}
    for item in _walk_mappings(payload):
        family = normalize_family(safe_get(item, FAMILY_KEYS, ""))
        if family not in TARGET_FAMILIES:
            continue
        risk_value = safe_get(item, ("precision_risk", "estimated_precision_risk", "risk", "risk_level"))
        risk = _normalize_risk(risk_value)
        if risk is None:
            continue
        current = risks.get(family)
        if current is None or _risk_rank(risk) > _risk_rank(current):
            risks[family] = risk
    return risks


def _merge_artifact_risks(target: dict[str, str], source: Mapping[str, str]) -> None:
    for family, risk in source.items():
        current = target.get(family)
        if current is None or _risk_rank(risk) > _risk_rank(current):
            target[family] = risk


def _normalize_risk(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if text.startswith("HIGH"):
        return "HIGH"
    if text.startswith("MEDIUM"):
        return "MEDIUM"
    if text.startswith("LOW"):
        return "LOW"
    return None


def _risk_rank(value: str) -> int:
    return {"LOW": 1, "MEDIUM": 2, "HIGH": 3}.get(value, 0)


def _extract_row_dicts(payload: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(payload, list):
        for item in payload:
            rows.extend(_extract_row_dicts(item))
    elif isinstance(payload, Mapping):
        if _looks_like_row(payload):
            rows.append(dict(payload))
        for value in payload.values():
            if isinstance(value, (list, dict)):
                rows.extend(_extract_row_dicts(value))
    return rows


def _looks_like_row(row: Mapping[str, Any]) -> bool:
    keys = set(row)
    return bool(
        keys.intersection(FAMILY_KEYS)
        and (
            keys.intersection(PROMPT_KEYS)
            or keys.intersection(GOLD_KEYS)
            or keys.intersection(PREDICTION_KEYS)
            or keys.intersection(ROW_ID_KEYS)
        )
    )


def _run_git(root: Path, args: Sequence[str]) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            check=False,
            text=True,
            capture_output=True,
        )
    except OSError as exc:
        return 127, "", str(exc)
    return completed.returncode, completed.stdout, completed.stderr.strip()


def _parse_dirty_files(output: str) -> list[str]:
    files: list[str] = []
    for line in output.splitlines():
        if len(line) < 4:
            continue
        files.append(line[3:].strip().replace("\\", "/"))
    return sorted(files)


def _matches_protected(path: str, marker: str) -> bool:
    normalized = path.replace("\\", "/")
    if marker.endswith(".py"):
        return normalized == marker or normalized.endswith("/" + Path(marker).name)
    return marker in normalized.lower()


def _gold_key_present(row: Mapping[str, Any]) -> bool:
    if any(key in row for key in GOLD_KEYS):
        return True
    metadata = row.get("metadata")
    return isinstance(metadata, Mapping) and any(key in metadata for key in GOLD_KEYS)


def _source_indicates_abstain(row: Mapping[str, Any], source: str) -> bool:
    text = json.dumps(row, sort_keys=True, default=str).lower()
    return "abstain" in source.lower() or "expected_abstain" in text


def _risk_level(
    family: str,
    unsafe_abstain: int,
    unknown_gold_rows: int,
    rows_seen: int,
    artifact_risk: str | None = None,
) -> str:
    if artifact_risk == "HIGH":
        return "HIGH"
    if artifact_risk == "MEDIUM":
        return "MEDIUM"
    if unsafe_abstain > 0 or family == "numeric_formula":
        return "HIGH"
    if rows_seen > 0 and unknown_gold_rows >= max(1, rows_seen // 2):
        return "MEDIUM"
    return "LOW"


def _local_recovery_risk(
    *,
    expected_abstain_rows: int,
    unsafe_abstain: int,
    current_wrong_or_abstain: int,
    artifact_risk: str | None,
) -> str:
    if expected_abstain_rows > 0 or unsafe_abstain > 0 or artifact_risk == "HIGH" or current_wrong_or_abstain == 0:
        return "HIGH"
    if artifact_risk == "MEDIUM":
        return "MEDIUM"
    return "MEDIUM"


def _synthetic_teacher_risk(*, family: str, concrete_rows: int, current_correct: int) -> str:
    perfect_concrete = concrete_rows > 0 and current_correct == concrete_rows
    if family in LOW_RISK_SYNTHETIC_FAMILIES and perfect_concrete:
        return "LOW"
    if family in MEDIUM_RISK_SYNTHETIC_FAMILIES:
        return "MEDIUM"
    if family in HIGH_RISK_SYNTHETIC_FAMILIES:
        return "HIGH"
    return "MEDIUM"


def _synthetic_generation_allowed(family: str) -> bool:
    return family in (
        LOW_RISK_SYNTHETIC_FAMILIES
        | MEDIUM_RISK_SYNTHETIC_FAMILIES
        | HIGH_RISK_SYNTHETIC_FAMILIES
    )


def _synthetic_generation_reason(
    *,
    family: str,
    allowed: bool,
    risk: str,
    concrete_rows: int,
    current_correct: int,
) -> str:
    if not allowed:
        return "blocked_no_safe_deterministic_synthetic_generator_specified"
    if family in {"equation_operator", "sequence_pattern"}:
        return f"synthetic_only_no_local_recovery_claim; strict verified generator required; synthetic_teacher_risk={risk}"
    if family == "numeric_formula":
        return "allowed_only_as_numeric_formula_safe_with_strict_ambiguity_rejection"
    if family == "gravity_numeric":
        return "allowed_only_when_formulas_constants_and_units_are_explicitly_verified"
    if concrete_rows > 0 and current_correct == concrete_rows:
        return (
            f"{current_correct} concrete rows solved locally; use as evidence-backed verified synthetic teacher source; "
            f"synthetic_teacher_risk={risk}"
        )
    return f"synthetic generation allowed as theoretical teacher source; synthetic_teacher_risk={risk}"


def _local_recovery_reason(
    *,
    expected_abstain_rows: int,
    current_wrong_or_abstain: int,
    unsafe_abstain: int,
    artifact_risk: str | None,
    local_router_allowed: bool,
) -> str:
    if local_router_allowed:
        return "concrete wrong local recovery pool exists with low local recovery risk"
    if expected_abstain_rows > 0:
        return f"local_router_integration_blocked: {expected_abstain_rows} expected-ABSTAIN rows present"
    if unsafe_abstain > 0:
        return f"local_router_integration_blocked: {unsafe_abstain} unsafe answers on expected-ABSTAIN rows"
    if artifact_risk in {"HIGH", "MEDIUM"}:
        return f"local_router_integration_blocked: Day2/Day3 artifact risk is {artifact_risk}"
    if current_wrong_or_abstain == 0:
        return "local_router_integration_blocked: no concrete wrong_or_abstain local recovery pool"
    return "local_router_integration_blocked: safety evidence insufficient"


def _synthetic_safe(family: str, risk_level: str) -> bool:
    del risk_level
    return _synthetic_generation_allowed(family)


def _teacher_priority(
    family: str,
    estimated_pool: int,
    safe_local: bool,
    expected_abstain_rows: int,
    concrete_rows: int,
) -> str:
    if not safe_local:
        return "BLOCKED" if expected_abstain_rows > 0 and concrete_rows == 0 else "P2"
    if family == "custom_numeral" and estimated_pool > 0:
        return "P0"
    if estimated_pool >= 20:
        return "P0"
    if estimated_pool >= 5:
        return "P1"
    return "P2"


def _recommended_action(
    family: str,
    safe_local: bool,
    synthetic_safe: bool,
    risk_level: str,
    priority: str,
) -> str:
    if safe_local:
        return "inspect_examples_then_implement_deterministic_teacher" if risk_level != "HIGH" else "forensic_review_before_teacher_work"
    if synthetic_safe:
        return "synthetic_teacher_only_no_local_router_integration"
    if priority == "BLOCKED":
        return "blocked_no_safe_local_recovery_evidence"
    return f"hold_{family}_until_more_evidence"


def _row_example(row: RowAssessment) -> dict[str, str]:
    return {
        "row_id": row.row_id,
        "prompt": row.prompt[:500],
        "gold": "" if row.gold is None else row.gold,
        "prediction": "" if row.prediction is None else row.prediction,
        "row_type": row.row_type,
        "current_outcome": row.current_outcome,
        "source": row.source,
    }


def _append_capped(items: list[dict[str, str]], value: dict[str, str], cap: int = 5) -> None:
    if len(items) < cap:
        items.append(value)


def _safe_number(payload: Mapping[str, Any] | None, key: str, default: int) -> int | float:
    if payload is None:
        return default
    value = payload.get(key, default)
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else default


def _walk_items(payload: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            yield str(key), value
            yield from _walk_items(value)
    elif isinstance(payload, list):
        for item in payload:
            yield from _walk_items(item)


def _walk_mappings(payload: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        yield payload
        for value in payload.values():
            yield from _walk_mappings(value)
    elif isinstance(payload, list):
        for item in payload:
            yield from _walk_mappings(item)


def _first_int(row: Mapping[str, Any], keys: Sequence[str]) -> int | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _source_name_from_path(path: Path) -> str:
    return path.stem


def _rel(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _warn(parse_warnings: list[str] | None, message: str) -> None:
    if parse_warnings is not None:
        parse_warnings.append(message)


if __name__ == "__main__":
    raise SystemExit(main())
