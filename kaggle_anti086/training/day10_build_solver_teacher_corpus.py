from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.v2_corpus_io import file_record, read_json, read_jsonl, write_json_checked, write_jsonl_checked
from kaggle_anti086.eval.day5_eval_factory import build_eval_rows
from kaggle_anti086.solvers.answer_normalizer import answers_match, normalize_answer
from kaggle_anti086.solvers.solver_ensemble import ANSWER_TYPE_BY_FAMILY, SolverEnsemble
from kaggle_anti086.training.day10_teacher_source_generators import (
    GENERATION_POLICY_VERSION,
    build_day10_repair_source_rows,
    parameter_tuple_hash as source_parameter_tuple_hash,
)


SUPPORTED_DIRECT_FAMILIES = {
    "bit_manipulation",
    "symbol_mapping",
    "char_cipher",
    "word_cipher",
    "unit_conversion",
    "numeric_formula",
    "gravity_numeric",
    "roman_numeral",
    "format_only",
}
UNSUPPORTED_OR_ABSTAIN_FAMILIES = {"custom_numeral", "sequence_pattern", "permutation_sorting", "equation_operator", "unknown"}
REQUIRED_FAMILIES = SUPPORTED_DIRECT_FAMILIES | {"abstain/unsupported"}
DEFAULT_EVAL_GLOBS = (
    "artifacts/sprint11/day5_*eval*.jsonl",
    "artifacts/sprint11/day5_*answerable*.jsonl",
    "artifacts/sprint11/day9_*eval*.jsonl",
    "artifacts/sprint11/day9_*locked*.jsonl",
)
FORBIDDEN_TARGET_RE = re.compile(
    r"\b(?:because|therefore|first|we\s+need|reasoning|explanation|expected(?:\s+answer)?|gold(?:\s+answer)?|metadata)\b|```|\n|\banswer\s*:",
    re.IGNORECASE,
)
ROW_ID_RE = re.compile(r"\b(?:day5|day9|day10|train_v2|source)[-_][A-Za-z0-9_.:-]+\b", re.IGNORECASE)
NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
BINARY_RE = re.compile(r"\b[01]{4,}\b")

FULL_TRAINING_MIN_DIRECT_ROWS = 4096
SMOKE_MIN_DIRECT_ROWS = 2000
FORMAT_ONLY_MIN_DIRECT_ROWS = 100
MAX_FAMILY_SHARE = 0.25
MAX_SUBFAMILY_SHARE = 0.12
MAX_TEMPLATE_SHARE = 0.10
MAX_NEAR_DUPLICATE_RATE = 0.02
REQUIRED_DIRECT_FAMILY_MINIMUMS = {
    "numeric_formula": 400,
    "gravity_numeric": 250,
    "unit_conversion": 400,
    "roman_numeral": 250,
    "bit_manipulation": 400,
    "symbol_mapping": 400,
    "char_cipher": 400,
    "word_cipher": 250,
    "format_only": 100,
}
DIRECT_DUPLICATE_ANSWER_CAPS = {
    "numeric_formula": 0.03,
    "gravity_numeric": 0.03,
    "unit_conversion": 0.03,
    "bit_manipulation": 0.03,
    "roman_numeral": 0.03,
    "symbol_mapping": 0.05,
    "char_cipher": 0.05,
    "word_cipher": 0.05,
    "format_only": 0.08,
}


def build_solver_teacher_corpus(
    *,
    rows: int = 768,
    seed: int = 1110,
    eval_paths: Iterable[str | Path] | None = None,
    source_mode: str = "private_like",
    target_direct_rows: int | None = None,
    target_abstain_rows: int = 512,
) -> dict[str, Any]:
    forbidden = collect_forbidden_eval_surface(eval_paths)
    if source_mode == "day10_repair":
        source_bundle = build_day10_repair_source_rows(
            target_direct_rows=target_direct_rows or rows,
            target_abstain_rows=target_abstain_rows,
            seed=seed,
        )
        target_counts = source_bundle["direct_target_counts"]
        direct_result = _solve_source_rows(
            source_bundle["direct_rows"],
            forbidden,
            target_counts=target_counts,
            direct_only=True,
        )
        abstain_result = _solve_source_rows(
            source_bundle["abstain_rows"],
            forbidden,
            target_counts=source_bundle["abstain_target_counts"],
            direct_only=False,
        )
        teacher_rows = direct_result["rows"]
        abstain_rows = abstain_result["rows"]
        rejected = direct_result["rejected"] + abstain_result["rejected"]
        generation_stats = _merge_generation_stats(direct_result["generation_stats"], abstain_result["generation_stats"])
    else:
        source_rows = build_day10_source_rows(rows=max(rows * 2, 256), seed=seed)
        solved = _solve_source_rows(source_rows, forbidden, target_counts={}, direct_only=False, stop_after=rows)
        teacher_rows = solved["rows"]
        abstain_rows = []
        rejected = solved["rejected"]
        generation_stats = solved["generation_stats"]
        source_bundle = {
            "generation_policy_version": "day10_pass10a_private_like_v1",
            "seed": seed,
            "source_mode": source_mode,
            "target_direct_rows": rows,
            "target_abstain_rows": 0,
        }

    audit = build_teacher_audit(teacher_rows, rejected, abstain_rows=abstain_rows)
    overlap = build_overlap_audit(teacher_rows + abstain_rows, forbidden)
    learnability = build_learnability_audit(teacher_rows)
    manifest = build_teacher_manifest(teacher_rows, audit, overlap, learnability, rows_requested=target_direct_rows or rows, seed=seed, abstain_rows=abstain_rows)
    mix_repair = build_mix_repair_report(
        rows_before=995 if source_mode == "day10_repair" else len(teacher_rows),
        direct_rows=teacher_rows,
        abstain_rows=abstain_rows,
        audit=audit,
        overlap=overlap,
        learnability=learnability,
        generation_stats=generation_stats,
        source_bundle=source_bundle,
    )
    if source_mode == "day10_repair" and mix_repair["status"] != "PASS":
        manifest["status"] = "FAIL"
    return {
        "rows": teacher_rows,
        "abstain_rows": abstain_rows,
        "rejected": rejected,
        "audit": audit,
        "overlap": overlap,
        "learnability": learnability,
        "mix_repair": mix_repair,
        "manifest": manifest,
    }


def build_day10_source_rows(*, rows: int, seed: int) -> list[dict[str, Any]]:
    raw_rows = build_eval_rows("private_like", rows, seed)
    output = []
    for idx, row in enumerate(raw_rows):
        copied = json.loads(json.dumps(row))
        family = str(copied["family"])
        original_signature = str(copied.get("metadata", {}).get("rule_signature", copied["rule_id"]))
        day10_signature = f"day10_teacher_{family}_{idx:05d}_{_short_hash(original_signature)}"
        copied["id"] = f"day10_source_{family}_{idx:05d}_{_short_hash(copied['id'])}"
        copied["rule_id"] = f"day10_source_rule_{family}_{idx:05d}_{_slug(day10_signature)}"
        copied["leakage_group"] = f"day10_source_lg_{family}_{idx:05d}_{_slug(day10_signature)}"
        copied["prompt"] = _rewrite_prompt(copied["prompt"], idx, family)
        copied["source"] = "day10_solver_teacher_source_factory"
        copied["split"] = "train"
        metadata = dict(copied.get("metadata", {}))
        metadata.update(
            {
                "created_by": "SPRINT-11_DAY10_PASS10A",
                "source_eval_family": "private_like_pattern_reuse_no_eval_row_reuse",
                "rule_signature": day10_signature,
                "original_day5_like_rule_signature_hash": _short_hash(original_signature, length=16),
                "teacher_source_index": idx,
            }
        )
        copied["metadata"] = metadata
        output.append(copied)
    return output


def _solve_source_rows(
    source_rows: list[dict[str, Any]],
    forbidden: dict[str, set[str]],
    *,
    target_counts: dict[str, int],
    direct_only: bool,
    stop_after: int | None = None,
) -> dict[str, Any]:
    teacher_rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_prompt_hashes: set[str] = set()
    seen_normalized_prompt_hashes: set[str] = set()
    seen_parameter_hashes: set[str] = set()
    answer_counts_by_family: dict[str, Counter[str]] = defaultdict(Counter)
    family_counts: Counter[str] = Counter()
    generation_stats = _new_generation_stats(target_counts, source_rows)
    ensemble = SolverEnsemble()

    for source in source_rows:
        family = str(source.get("family", "unknown"))
        if target_counts and family_counts[family] >= target_counts.get(family, 0):
            continue
        result = solve_teacher_row(source, ensemble)
        if result["row"] is None:
            _record_rejection(generation_stats, family, "solver_rejected")
            rejected.append(result["rejected"])
            continue
        row = result["row"]
        if direct_only and row["answer"] == "ABSTAIN":
            _record_rejection(generation_stats, family, "solver_rejected")
            rejected.append({"source_id": source.get("id"), "family": family, "reason": "abstain_not_allowed_in_direct_sft"})
            continue
        if not direct_only and row["answer"] != "ABSTAIN":
            _record_rejection(generation_stats, family, "solver_rejected")
            rejected.append({"source_id": source.get("id"), "family": family, "reason": "non_abstain_not_allowed_in_abstain_policy"})
            continue
        if _target_format_ok(str(row["answer"])) is False:
            _record_rejection(generation_stats, family, "format_rejected")
            rejected.append({"source_id": source.get("id"), "family": family, "reason": "teacher_answer_format_rejected"})
            continue
        if _row_overlaps(row, forbidden):
            _record_rejection(generation_stats, family, "overlap_rejected")
            rejected.append({"source_id": source.get("id"), "family": family, "reason": "eval_surface_overlap"})
            continue
        p_hash = prompt_hash(row["prompt"])
        np_hash = normalized_prompt_hash(row["prompt"])
        param_hash = _metadata_value(row, "source_parameter_tuple_hash")
        if p_hash in seen_prompt_hashes or np_hash in seen_normalized_prompt_hashes or (param_hash and param_hash in seen_parameter_hashes):
            _record_rejection(generation_stats, family, "duplicate_rejected")
            rejected.append({"source_id": source.get("id"), "family": family, "reason": "duplicate_prompt_or_parameter"})
            continue
        if direct_only and row["answer"] != "ABSTAIN":
            cap = _answer_duplicate_cap(family, target_counts.get(family, max(1, len(source_rows))))
            if answer_counts_by_family[family][row["answer"]] >= cap:
                _record_rejection(generation_stats, family, "duplicate_rejected")
                rejected.append({"source_id": source.get("id"), "family": family, "answer": row["answer"], "reason": "family_answer_duplicate_cap"})
                continue
            answer_counts_by_family[family][row["answer"]] += 1
        teacher_rows.append(row)
        generation_stats[family]["solver_verified"] += 1
        family_counts[family] += 1
        seen_prompt_hashes.add(p_hash)
        seen_normalized_prompt_hashes.add(np_hash)
        if param_hash:
            seen_parameter_hashes.add(param_hash)
        if stop_after is not None and len(teacher_rows) >= stop_after and _has_required_coverage(teacher_rows):
            break

    for family, stats in generation_stats.items():
        stats["remaining_needed"] = max(0, int(target_counts.get(family, 0)) - stats["solver_verified"])
    return {"rows": teacher_rows, "rejected": rejected, "generation_stats": generation_stats}


def _new_generation_stats(target_counts: dict[str, int], source_rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    families = set(target_counts) | {str(row.get("family", "unknown")) for row in source_rows}
    stats = {
        family: {
            "target": int(target_counts.get(family, 0)),
            "generation_attempted": 0,
            "solver_verified": 0,
            "solver_rejected": 0,
            "duplicate_rejected": 0,
            "overlap_rejected": 0,
            "format_rejected": 0,
            "remaining_needed": int(target_counts.get(family, 0)),
        }
        for family in sorted(families)
    }
    for row in source_rows:
        stats[str(row.get("family", "unknown"))]["generation_attempted"] += 1
    return stats


def _record_rejection(stats: dict[str, dict[str, int]], family: str, key: str) -> None:
    stats.setdefault(
        family,
        {
            "target": 0,
            "generation_attempted": 0,
            "solver_verified": 0,
            "solver_rejected": 0,
            "duplicate_rejected": 0,
            "overlap_rejected": 0,
            "format_rejected": 0,
            "remaining_needed": 0,
        },
    )
    stats[family][key] += 1


def _merge_generation_stats(left: dict[str, dict[str, int]], right: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    merged: dict[str, dict[str, int]] = {}
    for family in sorted(set(left) | set(right)):
        merged[family] = {}
        for key in {"target", "generation_attempted", "solver_verified", "solver_rejected", "duplicate_rejected", "overlap_rejected", "format_rejected", "remaining_needed"}:
            merged[family][key] = int(left.get(family, {}).get(key, 0)) + int(right.get(family, {}).get(key, 0))
    return merged


def _row_overlaps(row: dict[str, Any], forbidden: dict[str, set[str]]) -> bool:
    return any(
        (
            str(row.get("source_id", "")) in forbidden.get("source_id", set()),
            prompt_hash(str(row.get("prompt", ""))) in forbidden.get("prompt_hash", set()),
            normalized_prompt_hash(str(row.get("prompt", ""))) in forbidden.get("normalized_prompt_hash", set()),
            _metadata_value(row, "source_rule_signature") in forbidden.get("rule_signature", set()),
            _metadata_value(row, "source_leakage_group") in forbidden.get("leakage_group", set()),
            str(row.get("source_id", "")) in forbidden.get("eval_row_id", set()),
            _metadata_value(row, "source_parameter_tuple_hash") in forbidden.get("parameter_tuple_hash", set()),
            _metadata_value(row, "source_prompt_template_signature") in forbidden.get("prompt_template_signature", set()),
        )
    )


def _answer_duplicate_cap(family: str, target_count: int) -> int:
    share = DIRECT_DUPLICATE_ANSWER_CAPS.get(family, 0.03)
    return max(3, int(max(1, target_count) * share))


def solve_teacher_row(source_row: dict[str, Any], ensemble: SolverEnsemble | None = None) -> dict[str, Any]:
    ensemble = ensemble or SolverEnsemble()
    solver_input = _strip_gold_for_teacher(source_row)
    result = ensemble.run_all(solver_input)
    expected_behavior = str(source_row.get("metadata", {}).get("expected_solver_behavior", "answer"))
    if result.abstained or not result.candidates:
        if expected_behavior == "abstain" or str(source_row.get("family")) in UNSUPPORTED_OR_ABSTAIN_FAMILIES:
            return {"row": build_teacher_row(source_row, answer="ABSTAIN", result=result, candidate=None, expected_behavior="abstain"), "rejected": None}
        return {"row": None, "rejected": {"source_id": source_row.get("id"), "family": source_row.get("family"), "reason": result.reason or "teacher_abstained"}}
    candidate = result.candidates[0]
    if not candidate.verified:
        return {"row": None, "rejected": {"source_id": source_row.get("id"), "family": source_row.get("family"), "reason": "teacher_candidate_unverified"}}
    if candidate.risk == "high":
        return {"row": None, "rejected": {"source_id": source_row.get("id"), "family": source_row.get("family"), "reason": "teacher_candidate_high_risk"}}
    answer = str(candidate.answer).strip()
    if not answer or not _target_format_ok(answer):
        return {"row": None, "rejected": {"source_id": source_row.get("id"), "family": source_row.get("family"), "answer": answer, "reason": "teacher_answer_format_rejected"}}
    return {"row": build_teacher_row(source_row, answer=answer, result=result, candidate=candidate, expected_behavior="answer"), "rejected": None}


def build_teacher_row(source_row: dict[str, Any], *, answer: str, result: Any, candidate: Any | None, expected_behavior: str) -> dict[str, Any]:
    family = str(source_row.get("family", "unknown"))
    source_id = str(source_row["id"])
    answer_type = ANSWER_TYPE_BY_FAMILY.get(family)
    normalized = normalize_answer(answer, expected_type=answer_type).normalized
    gold = str(source_row.get("answer", ""))
    teacher_matches_gold = None
    if gold and gold.upper() != "ABSTAIN" and answer != "ABSTAIN":
        teacher_matches_gold = answers_match(answer, gold, answer_type=answer_type)
    prompt = str(source_row["prompt"]).strip()
    row_id = f"day10_teacher_{family}_{_short_hash(source_id)}"
    return {
        "id": row_id,
        "source_id": source_id,
        "family": family,
        "subfamily": str(source_row.get("subfamily", "")),
        "prompt": prompt,
        "answer": "ABSTAIN" if answer == "ABSTAIN" else normalized,
        "expected_behavior": expected_behavior,
        "solver_name": "solver_ensemble",
        "solver_source": "solver_ensemble" if candidate is None else str(candidate.source),
        "solver_confidence": 1.0 if answer == "ABSTAIN" else float(candidate.confidence),
        "verified": True,
        "risk": "low" if candidate is None else str(candidate.risk),
        "metadata": {
            "created_by": "SPRINT-11G_DAY10_PASS10A",
            "teacher": "SolverEnsemble",
            "no_chain_of_thought": True,
            "target_style": "final_answer_only",
            "source_rule_id": source_row.get("rule_id"),
            "source_leakage_group": source_row.get("leakage_group"),
            "source_rule_signature": source_row.get("metadata", {}).get("rule_signature", ""),
            "source_parameter_tuple_hash": source_row.get("metadata", {}).get("parameter_tuple_hash", ""),
            "source_prompt_template_id": source_row.get("metadata", {}).get("prompt_template_id", ""),
            "source_prompt_template_signature": prompt_template_signature(prompt),
            "source_generator_id": source_row.get("metadata", {}).get("generator_id", ""),
            "source_prompt_hash": prompt_hash(prompt),
            "source_normalized_prompt_hash": normalized_prompt_hash(prompt),
            "teacher_result_reason": getattr(result, "reason", ""),
            "candidate_count": 0 if getattr(result, "abstained", False) else len(getattr(result, "candidates", [])),
            "teacher_matches_source_gold_audit_only": teacher_matches_gold,
            "expected_solver_behavior": expected_behavior,
            "abstain_policy": "teacher_verified_abstain_only" if answer == "ABSTAIN" else "",
            "gold_used_for_target": False,
        },
    }


def build_teacher_audit(rows: list[dict[str, Any]], rejected: list[dict[str, Any]], *, abstain_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    abstain_rows = abstain_rows or []
    failures: list[str] = []
    ids = [row["id"] for row in rows]
    source_ids = [row["source_id"] for row in rows]
    prompts = [row["prompt"] for row in rows]
    if len(ids) != len(set(ids)):
        failures.append("duplicate_teacher_id")
    if len(source_ids) != len(set(source_ids)):
        failures.append("duplicate_source_id")
    if len(prompts) != len(set(prompts)):
        failures.append("duplicate_prompt")
    bad_targets = [row["id"] for row in rows if not _target_format_ok(str(row["answer"]))]
    if bad_targets:
        failures.append("bad_target_format")
    prompt_leaks = [row["id"] for row in rows if _prompt_has_leak(str(row["prompt"]), row)]
    if prompt_leaks:
        failures.append("prompt_leak_detected")
    direct_abstain = [row["id"] for row in rows if row["answer"] == "ABSTAIN"]
    if direct_abstain:
        failures.append("abstain_in_direct_answer_corpus")
    family_counts = Counter(row["family"] for row in rows)
    if not SUPPORTED_DIRECT_FAMILIES <= set(family_counts):
        failures.append("required_family_coverage_missing")
    if any(row["verified"] is not True for row in rows):
        failures.append("unverified_teacher_row")
    return {
        "status": "PASS" if not failures else "FAIL",
        "row_count": len(rows),
        "rejected_count": len([item for item in rejected if item]),
        "family_counts": dict(sorted(family_counts.items())),
        "abstain_count": len(abstain_rows),
        "direct_abstain_count": len(direct_abstain),
        "bad_target_ids": bad_targets[:20],
        "prompt_leak_ids": prompt_leaks[:20],
        "failures": failures,
        "no_training_performed": True,
        "no_adapter_packaged": True,
        "no_submission_created": True,
        "no_leaderboard_evidence": True,
        "no_0_95_evidence": True,
    }


def build_overlap_audit(rows: list[dict[str, Any]], forbidden: dict[str, set[str]]) -> dict[str, Any]:
    source_ids = {str(row["source_id"]) for row in rows}
    prompt_hashes = {prompt_hash(str(row["prompt"])) for row in rows}
    normalized_hashes = {normalized_prompt_hash(str(row["prompt"])) for row in rows}
    rule_signatures = {str(row.get("metadata", {}).get("source_rule_signature", "")) for row in rows if row.get("metadata", {}).get("source_rule_signature")}
    leakage_groups = {str(row.get("metadata", {}).get("source_leakage_group", "")) for row in rows if row.get("metadata", {}).get("source_leakage_group")}
    parameter_hashes = {str(row.get("metadata", {}).get("source_parameter_tuple_hash", "")) for row in rows if row.get("metadata", {}).get("source_parameter_tuple_hash")}
    template_signatures = {str(row.get("metadata", {}).get("source_prompt_template_signature", "")) for row in rows if row.get("metadata", {}).get("source_prompt_template_signature")}
    overlaps = {
        "source_id_overlap_count": len(source_ids & forbidden["source_id"]),
        "prompt_hash_overlap_count": len(prompt_hashes & forbidden["prompt_hash"]),
        "normalized_prompt_hash_overlap_count": len(normalized_hashes & forbidden["normalized_prompt_hash"]),
        "rule_signature_overlap_count": len(rule_signatures & forbidden["rule_signature"]),
        "leakage_group_overlap_count": len(leakage_groups & forbidden["leakage_group"]),
        "eval_row_id_overlap_count": len(source_ids & forbidden["eval_row_id"]),
        "parameter_tuple_hash_overlap_count": len(parameter_hashes & forbidden.get("parameter_tuple_hash", set())),
        "prompt_template_signature_overlap_count": len(template_signatures & forbidden.get("prompt_template_signature", set())),
    }
    failures = [key for key, value in overlaps.items() if value]
    return {
        "status": "PASS" if not failures else "FAIL",
        **overlaps,
        "checked_eval_row_count": len(forbidden["eval_row_id"]),
        "checked_eval_files": sorted(forbidden.get("files", set())),
        "failures": failures,
    }


def build_learnability_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    family_counts = Counter(row["family"] for row in rows)
    template_counts = Counter(prompt_template_signature(row["prompt"]) for row in rows)
    query_position_counts = Counter(query_position_bucket(row["prompt"]) for row in rows)
    answer_counts_by_family: dict[str, Counter[str]] = defaultdict(Counter)
    subfamilies_by_family: dict[str, Counter[str]] = defaultdict(Counter)
    number_operator_counts = Counter(number_operator_signature(row["prompt"]) for row in rows)
    answer_length_counts = Counter(len(str(row["answer"])) for row in rows)
    generator_counts = Counter(str(row.get("metadata", {}).get("source_generator_id", "")) for row in rows)
    parameter_hashes = {str(row.get("metadata", {}).get("source_parameter_tuple_hash", "")) for row in rows if row.get("metadata", {}).get("source_parameter_tuple_hash")}
    near_duplicates = near_duplicate_prompt_count(rows)
    for row in rows:
        if row["expected_behavior"] != "abstain" and row["answer"] != "ABSTAIN":
            answer_counts_by_family[row["family"]][row["answer"]] += 1
        subfamilies_by_family[row["family"]][row["subfamily"]] += 1
    failures: list[str] = []
    warnings: list[str] = []
    total = max(1, len(rows))
    max_template_share = max(template_counts.values(), default=0) / total
    if max_template_share > MAX_TEMPLATE_SHARE:
        failures.append("one_template_dominates")
    near_duplicate_rate = near_duplicates / total
    if near_duplicate_rate > MAX_NEAR_DUPLICATE_RATE:
        failures.append("near_duplicate_prompt_rate_high")
    family_cap_violations = {family: count for family, count in family_counts.items() if count / total > MAX_FAMILY_SHARE}
    if family_cap_violations:
        failures.append("family_cap_exceeded")
    subfamily_cap_violations = {
        f"{family}/{subfamily}": count
        for family, counter in subfamilies_by_family.items()
        for subfamily, count in counter.items()
        if count / total > MAX_SUBFAMILY_SHARE
    }
    if subfamily_cap_violations:
        failures.append("subfamily_cap_exceeded")
    duplicate_answer_violations = {}
    for family, counter in answer_counts_by_family.items():
        family_total = sum(counter.values())
        cap = DIRECT_DUPLICATE_ANSWER_CAPS.get(family, 0.03)
        if family_total >= 20 and counter.most_common(1)[0][1] / family_total > cap:
            duplicate_answer_violations[family] = counter.most_common(1)[0]
    if duplicate_answer_violations:
        failures.append("duplicate_answer_rate_excessive")
    subfamily_balance_warnings = {
        family: counter.most_common(1)[0]
        for family, counter in subfamilies_by_family.items()
        if sum(counter.values()) >= 20 and counter.most_common(1)[0][1] / sum(counter.values()) > 0.75
    }
    if subfamily_balance_warnings:
        warnings.append("subfamily_balance_skew")
    if len(query_position_counts) < 3:
        failures.append("query_position_diversity_low")
    if len(number_operator_counts) < 8:
        failures.append("number_operator_diversity_low")
    if len(answer_length_counts) < 6:
        failures.append("answer_length_diversity_low")
    if len(parameter_hashes) < len(rows) * 0.98:
        failures.append("generated_source_entropy_low")
    return {
        "status": "PASS" if not failures else "FAIL",
        "row_count": len(rows),
        "prompt_template_count": len(template_counts),
        "max_prompt_template_share": max_template_share,
        "query_position_distribution": dict(sorted(query_position_counts.items())),
        "answer_unique_count": len({row["answer"] for row in rows}),
        "answer_length_distribution": dict(sorted(answer_length_counts.items())),
        "number_operator_diversity": dict(sorted(number_operator_counts.items())),
        "generated_source_entropy": len(parameter_hashes) / total,
        "generator_distribution": dict(sorted(generator_counts.items())),
        "near_duplicate_prompt_count": near_duplicates,
        "near_duplicate_prompt_rate": near_duplicate_rate,
        "duplicate_answer_violations": duplicate_answer_violations,
        "abstain_rows_excluded_from_default_direct_sft": all(row["expected_behavior"] == "abstain" for row in rows if row["answer"] == "ABSTAIN"),
        "subfamily_balance_warnings": subfamily_balance_warnings,
        "family_counts": dict(sorted(family_counts.items())),
        "family_cap_violations": family_cap_violations,
        "subfamily_cap_violations": subfamily_cap_violations,
        "warnings": warnings,
        "failures": failures,
    }


def build_teacher_manifest(rows: list[dict[str, Any]], audit: dict[str, Any], overlap: dict[str, Any], learnability: dict[str, Any], *, rows_requested: int, seed: int, abstain_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    abstain_rows = abstain_rows or []
    status = "PASS" if audit["status"] == overlap["status"] == learnability["status"] == "PASS" else "FAIL"
    return {
        "schema_version": 1,
        "created_by": "SPRINT-11G_DAY10_PASS10A",
        "status": status,
        "rows_requested": rows_requested,
        "row_count": len(rows),
        "direct_answer_rows": len(rows),
        "abstain_policy_rows": len(abstain_rows),
        "seed": seed,
        "family_counts": dict(sorted(Counter(row["family"] for row in rows).items())),
        "subfamily_counts": dict(sorted(Counter(f"{row['family']}/{row['subfamily']}" for row in rows).items())),
        "solver_source_counts": dict(sorted(Counter(row["solver_source"] for row in rows).items())),
        "risk_counts": dict(sorted(Counter(row["risk"] for row in rows).items())),
        "abstain_count": len(abstain_rows),
        "audit_status": audit["status"],
        "overlap_status": overlap["status"],
        "learnability_status": learnability["status"],
        "training_allowed": False,
        "packaging_allowed": False,
        "submission_allowed": False,
        "no_training_performed": True,
        "no_adapter_packaged": True,
        "no_submission_created": True,
        "no_leaderboard_evidence": True,
        "no_0_95_evidence": True,
    }


def build_mix_repair_report(
    *,
    rows_before: int,
    direct_rows: list[dict[str, Any]],
    abstain_rows: list[dict[str, Any]],
    audit: dict[str, Any],
    overlap: dict[str, Any],
    learnability: dict[str, Any],
    generation_stats: dict[str, dict[str, int]],
    source_bundle: dict[str, Any],
    direct_record: dict[str, Any] | None = None,
    abstain_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    direct_count = len(direct_rows)
    family_counts = Counter(row["family"] for row in direct_rows)
    subfamily_counts = Counter(f"{row['family']}/{row['subfamily']}" for row in direct_rows)
    remaining_blockers: list[str] = []
    family_minimum_failures = {
        family: {"required": required, "actual": family_counts.get(family, 0)}
        for family, required in REQUIRED_DIRECT_FAMILY_MINIMUMS.items()
        if family_counts.get(family, 0) < required
    }
    if family_minimum_failures:
        remaining_blockers.append("family_minimums_not_met")
    if direct_count < SMOKE_MIN_DIRECT_ROWS:
        remaining_blockers.append("direct_rows_below_smoke_threshold")
    if direct_count < FULL_TRAINING_MIN_DIRECT_ROWS:
        remaining_blockers.append("direct_rows_below_full_training_threshold")
    if family_counts.get("format_only", 0) < FORMAT_ONLY_MIN_DIRECT_ROWS:
        remaining_blockers.append("format_only_coverage_below_minimum")
    if audit.get("status") != "PASS":
        remaining_blockers.append("teacher_audit_not_pass")
    if overlap.get("status") != "PASS":
        remaining_blockers.append("overlap_audit_not_pass")
    if learnability.get("status") != "PASS":
        remaining_blockers.append("learnability_audit_not_pass")
    family_cap_violations = learnability.get("family_cap_violations", {})
    subfamily_cap_violations = learnability.get("subfamily_cap_violations", {})
    if family_cap_violations:
        remaining_blockers.append("family_cap_violations")
    if subfamily_cap_violations:
        remaining_blockers.append("subfamily_cap_violations")
    target_direct_rows = int(source_bundle.get("target_direct_rows", direct_count))
    target_abstain_rows = int(source_bundle.get("target_abstain_rows", len(abstain_rows)))
    if direct_count < target_direct_rows:
        remaining_blockers.append("target_direct_rows_not_met")
    if len(abstain_rows) < target_abstain_rows:
        remaining_blockers.append("target_abstain_rows_not_met")
    status = "PASS" if not remaining_blockers else "FAIL"
    return {
        "status": status,
        "generation_policy_version": source_bundle.get("generation_policy_version", GENERATION_POLICY_VERSION),
        "generator_seed": source_bundle.get("seed"),
        "source_mode": source_bundle.get("source_mode", "unknown"),
        "target_direct_rows": target_direct_rows,
        "target_abstain_rows": target_abstain_rows,
        "row_count_before": rows_before,
        "row_count_after": direct_count,
        "direct_answer_rows": direct_count,
        "abstain_policy_rows": len(abstain_rows),
        "smoke_training_eligible": status == "PASS" and direct_count >= SMOKE_MIN_DIRECT_ROWS,
        "full_training_eligible": status == "PASS" and direct_count >= FULL_TRAINING_MIN_DIRECT_ROWS,
        "by_family": dict(sorted(family_counts.items())),
        "by_subfamily": dict(sorted(subfamily_counts.items())),
        "format_only_count": family_counts.get("format_only", 0),
        "family_minimum_failures": family_minimum_failures,
        "family_cap_violations": family_cap_violations,
        "subfamily_cap_violations": subfamily_cap_violations,
        "template_dominance_warnings": ["one_template_dominates"] if "one_template_dominates" in learnability.get("failures", []) else [],
        "near_duplicate_prompt_warnings": ["near_duplicate_prompt_rate_high"] if "near_duplicate_prompt_rate_high" in learnability.get("failures", []) else [],
        "near_duplicate_prompt_rate": learnability.get("near_duplicate_prompt_rate", 0.0),
        "overlap_status": overlap.get("status"),
        "learnability_status": learnability.get("status"),
        "teacher_audit_status": audit.get("status"),
        "generation_stats_by_family": generation_stats,
        "remaining_blockers": sorted(set(remaining_blockers)),
        "warnings": learnability.get("warnings", []),
        "direct_output_sha256": "" if direct_record is None else direct_record.get("sha256", ""),
        "abstain_output_sha256": "" if abstain_record is None else abstain_record.get("sha256", ""),
        "output_hashes": {
            "direct": "" if direct_record is None else direct_record.get("sha256", ""),
            "abstain_policy": "" if abstain_record is None else abstain_record.get("sha256", ""),
        },
        "packaging_allowed": False,
        "submission_allowed": False,
        "no_training_performed": True,
        "no_leaderboard_evidence": True,
        "no_0_95_evidence": True,
    }


def collect_forbidden_eval_surface(paths: Iterable[str | Path] | None = None) -> dict[str, set[str]]:
    discovered = [Path(path) for path in paths] if paths else _discover_eval_paths()
    values: dict[str, set[str]] = {
        "source_id": set(),
        "prompt_hash": set(),
        "normalized_prompt_hash": set(),
        "rule_signature": set(),
        "leakage_group": set(),
        "eval_row_id": set(),
        "parameter_tuple_hash": set(),
        "prompt_template_signature": set(),
        "files": set(),
    }
    for path in discovered:
        if not path.exists() or path.suffix != ".jsonl":
            continue
        try:
            rows = read_jsonl(path)
        except Exception:
            continue
        values["files"].add(str(path))
        for row in rows:
            prompt = str(row.get("prompt", ""))
            row_id = str(row.get("id", ""))
            if row_id:
                values["source_id"].add(row_id)
                values["eval_row_id"].add(row_id)
            if prompt:
                values["prompt_hash"].add(prompt_hash(prompt))
                values["normalized_prompt_hash"].add(normalized_prompt_hash(prompt))
                values["prompt_template_signature"].add(prompt_template_signature(prompt))
            signature = str(row.get("metadata", {}).get("rule_signature", ""))
            if signature:
                values["rule_signature"].add(signature)
            parameter_hash = str(row.get("metadata", {}).get("parameter_tuple_hash", ""))
            if parameter_hash:
                values["parameter_tuple_hash"].add(parameter_hash)
            leakage = str(row.get("leakage_group", ""))
            if leakage:
                values["leakage_group"].add(leakage)
    return values


def _discover_eval_paths() -> list[Path]:
    paths: set[Path] = set()
    for pattern in DEFAULT_EVAL_GLOBS:
        paths.update(Path(".").glob(pattern))
    manifest = Path("artifacts/sprint11/day5_eval_manifest.json")
    if manifest.exists():
        try:
            data = read_json(manifest)
            for item in data.get("files", {}).values():
                if isinstance(item, dict) and item.get("path"):
                    paths.add(Path(str(item["path"])))
        except Exception:
            pass
    return sorted(paths)


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(str(prompt).encode("utf-8")).hexdigest()


def normalized_prompt_hash(prompt: str) -> str:
    return prompt_hash(normalize_prompt_surface(prompt))


def normalize_prompt_surface(prompt: str) -> str:
    text = str(prompt).lower()
    text = re.sub(r"\[[^\]]+\]", "[tag]", text)
    text = NUMBER_RE.sub("<num>", text)
    text = BINARY_RE.sub("<bin>", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def prompt_template_signature(prompt: str) -> str:
    text = normalize_prompt_surface(prompt)
    text = re.sub(r"[a-z]{4,}", "<word>", text)
    return _short_hash(text, length=12)


def query_position_bucket(prompt: str) -> str:
    lowered = prompt.lower()
    markers = [idx for idx in (lowered.find("query"), lowered.find("input"), lowered.find("solve"), lowered.find("convert"), lowered.find("extract")) if idx >= 0]
    if not markers:
        return "missing_query_marker"
    pos = min(markers) / max(1, len(prompt))
    if pos < 0.33:
        return "early"
    if pos < 0.66:
        return "middle"
    return "late"


def number_operator_signature(prompt: str) -> str:
    text = str(prompt)
    has_decimal = bool(re.search(r"\d+\.\d+", text))
    has_binary = bool(BINARY_RE.search(text))
    has_arrow = "->" in text
    has_question = "?" in text
    symbol_count = len(re.findall(r"[!@#$%^&*{}\[\]<>?/|~+=:;.-]", text))
    return f"decimal={int(has_decimal)}|binary={int(has_binary)}|arrow={int(has_arrow)}|q={int(has_question)}|sym={min(symbol_count, 9)}"


def near_duplicate_prompt_count(rows: list[dict[str, Any]]) -> int:
    signatures = [normalize_prompt_surface(row["prompt"]) for row in rows]
    counts = Counter(signatures)
    return sum(count - 1 for count in counts.values() if count > 1)


def write_day10_outputs(
    result: dict[str, Any],
    *,
    out_direct: str | Path,
    out_manifest: str | Path,
    out_audit: str | Path,
    out_overlap: str | Path,
    out_learnability: str | Path,
    out_abstain_policy: str | Path | None = None,
    out_mix_repair: str | Path | None = None,
) -> dict[str, Any]:
    direct_record = write_jsonl_checked(out_direct, result["rows"], field_name="day10_solver_teacher_direct")
    abstain_record = None
    if out_abstain_policy is not None:
        abstain_record = write_jsonl_checked(out_abstain_policy, result.get("abstain_rows", []), field_name="day10_solver_teacher_abstain_policy")
    if out_mix_repair is not None:
        mix = dict(result["mix_repair"])
        mix["direct_output_sha256"] = direct_record.get("sha256", "")
        mix["abstain_output_sha256"] = "" if abstain_record is None else abstain_record.get("sha256", "")
        mix["output_hashes"] = {
            "direct": direct_record.get("sha256", ""),
            "abstain_policy": "" if abstain_record is None else abstain_record.get("sha256", ""),
        }
        result["mix_repair"] = mix
    result["manifest"]["files"] = {"direct": direct_record}
    if abstain_record is not None:
        result["manifest"]["files"]["abstain_policy"] = abstain_record
    write_json_checked(out_audit, result["audit"], field_name="day10_solver_teacher_audit")
    write_json_checked(out_overlap, result["overlap"], field_name="day10_teacher_overlap_audit")
    write_json_checked(out_learnability, result["learnability"], field_name="day10_teacher_learnability_audit")
    if out_mix_repair is not None:
        write_json_checked(out_mix_repair, result["mix_repair"], field_name="day10_corpus_mix_repair_report")
    result["manifest"]["files"].update(
        {
            "audit": file_record(out_audit),
            "overlap_audit": file_record(out_overlap),
            "learnability_audit": file_record(out_learnability),
        }
    )
    if out_mix_repair is not None:
        result["manifest"]["files"]["mix_repair"] = file_record(out_mix_repair)
    write_json_checked(out_manifest, result["manifest"], field_name="day10_solver_teacher_manifest")
    return result["manifest"]


def _strip_gold_for_teacher(row: dict[str, Any]) -> dict[str, Any]:
    copied = dict(row)
    copied.pop("answer", None)
    return copied


def _rewrite_prompt(prompt: str, idx: int, family: str) -> str:
    prompt = re.sub(r"^\[[^\]]+\]\s*", "", str(prompt)).strip()
    surface_tag = _alpha_tag(idx)
    variants = (
        f"Context label {surface_tag}: infer the transformation from examples. {prompt}",
        f"{prompt}\nOutput only the final transformed value for the {surface_tag} query.",
        f"Examples define a hidden rule for {family}; surface tag {surface_tag}. Query is included below.\n{prompt}",
        f"{prompt}\nFor tag {surface_tag}, no explanation; output the final answer token/string.",
    )
    return variants[idx % len(variants)]


def _target_format_ok(answer: str) -> bool:
    text = str(answer).strip()
    if not text:
        return False
    if FORBIDDEN_TARGET_RE.search(text):
        return False
    if ROW_ID_RE.search(text):
        return False
    return True


def _prompt_has_leak(prompt: str, row: dict[str, Any]) -> bool:
    text = str(prompt)
    if str(row.get("id", "")) and str(row["id"]) in text:
        return True
    if str(row.get("source_id", "")) and str(row["source_id"]) in text:
        return True
    if re.search(r"\b(?:expected(?:\s+answer)?|gold(?:\s+answer)?|metadata)\b", text, re.IGNORECASE):
        return True
    return False


def _has_required_coverage(rows: list[dict[str, Any]]) -> bool:
    families = {row["family"] for row in rows}
    has_abstain = any(row["answer"] == "ABSTAIN" or row["expected_behavior"] == "abstain" for row in rows)
    return SUPPORTED_DIRECT_FAMILIES <= families and has_abstain


def _short_hash(value: str, *, length: int = 10) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:length]


def _metadata_value(row: dict[str, Any], key: str) -> str:
    value = row.get("metadata", {}).get(key, "")
    return "" if value is None else str(value)


def _alpha_tag(idx: int) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    n = max(0, int(idx))
    chars = []
    while True:
        chars.append(alphabet[n % len(alphabet)])
        n = n // len(alphabet) - 1
        if n < 0:
            break
    return "tag" + "".join(reversed(chars))


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(value))[:96].strip("_")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Day 10 solver-teacher corpus without gold-label target leakage.")
    parser.add_argument("--rows", type=int, default=768)
    parser.add_argument("--source-mode", choices=["private_like", "day10_repair"], default="day10_repair")
    parser.add_argument("--target-direct-rows", type=int, default=6144)
    parser.add_argument("--target-abstain-rows", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1110)
    parser.add_argument("--out-direct", default="artifacts/sprint11/day10_solver_teacher_direct.jsonl")
    parser.add_argument("--out-abstain-policy", default="artifacts/sprint11/day10_solver_teacher_abstain_policy.jsonl")
    parser.add_argument("--out-manifest", default="artifacts/sprint11/day10_solver_teacher_manifest.json")
    parser.add_argument("--out-audit", default="artifacts/sprint11/day10_solver_teacher_audit.json")
    parser.add_argument("--out-overlap", default="artifacts/sprint11/day10_teacher_overlap_audit.json")
    parser.add_argument("--out-learnability", default="artifacts/sprint11/day10_teacher_learnability_audit.json")
    parser.add_argument("--out-mix-repair", default="artifacts/sprint11/day10_corpus_mix_repair_report.json")
    args = parser.parse_args(argv)
    result = build_solver_teacher_corpus(
        rows=args.rows,
        seed=args.seed,
        source_mode=args.source_mode,
        target_direct_rows=args.target_direct_rows,
        target_abstain_rows=args.target_abstain_rows,
    )
    manifest = write_day10_outputs(
        result,
        out_direct=args.out_direct,
        out_abstain_policy=args.out_abstain_policy if args.source_mode == "day10_repair" else None,
        out_manifest=args.out_manifest,
        out_audit=args.out_audit,
        out_overlap=args.out_overlap,
        out_learnability=args.out_learnability,
        out_mix_repair=args.out_mix_repair if args.source_mode == "day10_repair" else None,
    )
    print(json.dumps({"status": manifest["status"], "row_count": manifest["row_count"], "out": args.out_direct}, sort_keys=True))
    return 0 if manifest["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
