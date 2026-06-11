# kaggle_anti086/training/day1x_stress_diversity_leakage_and_final_manifest.py
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


BOX_RE = re.compile(r"\\boxed\{([^{}]*)\}")
PLACEHOLDER_PATTERNS = (
    "ABSTAIN",
    "TODO",
    "UNKNOWN",
    "N/A",
    "NONE",
    "NULL",
    "PLACEHOLDER",
    "EXPECTED-ABSTAIN",
    "EXPECTED_ABSTAIN",
)

ORIGINAL_FAMILIES = {
    "symbol_mapping",
    "bit_manipulation",
    "char_cipher",
    "unit_conversion",
    "numeric_formula_safe",
    "word_cipher",
    "custom_numeral",
    "permutation_sorting",
    "gravity_numeric",
    "equation_operator",
    "sequence_pattern",
}

COMPOSED_FAMILIES = {
    "composed_custom_numeral_arithmetic",
    "composed_symbol_equation",
    "composed_cipher_mapping",
    "composed_unit_formula",
    "composed_bit_conversion",
    "composed_sequence_operator",
    "composed_permutation_mapping",
    "composed_word_cipher",
    "composed_gravity_unit",
}

CORE_FILES = {
    "train_100k": "day1x_verified_sft_train_100k.jsonl",
    "eval_10k": "day1x_verified_sft_eval_10k.jsonl",
    "probe_10k": "day1x_verified_probe_10k.jsonl",
}

HOLDOUT_FILES = {
    "holdout_rule": "day1x_holdout_rule.jsonl",
    "holdout_prompt_style": "day1x_holdout_prompt_style.jsonl",
    "holdout_composed": "day1x_holdout_composed.jsonl",
    "holdout_format_traps": "day1x_holdout_format_traps.jsonl",
    "holdout_hard_p2": "day1x_holdout_hard_p2.jsonl",
}

VARIANT_FILES = {
    "direct": "day1x_sft_direct.jsonl",
    "short_trace": "day1x_sft_short_trace.jsonl",
    "mixed_curriculum": "day1x_sft_mixed_curriculum.jsonl",
    "format_heavy": "day1x_sft_format_heavy.jsonl",
    "composed_heavy": "day1x_sft_composed_heavy.jsonl",
    "public_style_heavy": "day1x_sft_public_style_heavy.jsonl",
    "hard_p2_heavy": "day1x_sft_hard_p2_heavy.jsonl",
}

DPO_FILES = {
    "dpo_train": "day1x_dpo_train_pairs_20k.jsonl",
    "dpo_eval": "day1x_dpo_eval_pairs_2k.jsonl",
}


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                yield {
                    "__parse_error__": True,
                    "__line_no__": line_no,
                    "__error__": repr(exc),
                    "__raw__": line[:500],
                }
                continue
            if isinstance(obj, dict):
                yield obj
            else:
                yield {
                    "__parse_error__": True,
                    "__line_no__": line_no,
                    "__error__": "jsonl row is not object",
                    "__raw__": line[:500],
                }


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in iter_jsonl(path))


def norm_ws(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def norm_prompt(text: Any) -> str:
    return norm_ws(text).lower()


def short_hash(text: str, n: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def token_prefix(text: str, n: int = 16) -> str:
    tokens = re.findall(r"[A-Za-z0-9_+\-*/=<>]+", norm_prompt(text))
    return " ".join(tokens[:n])


def word_ngrams(text: str, n: int = 5) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9_+\-*/=<>]+", norm_prompt(text))
    if len(tokens) < n:
        return [" ".join(tokens)] if tokens else []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def get_prompt(row: Dict[str, Any]) -> str:
    if "prompt" in row:
        return norm_ws(row.get("prompt"))
    messages = row.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return norm_ws(msg.get("content"))
    return ""


def get_assistant_text(row: Dict[str, Any]) -> str:
    if "target_text" in row:
        return norm_ws(row.get("target_text"))
    if "chosen" in row:
        return norm_ws(row.get("chosen"))
    messages = row.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return norm_ws(msg.get("content"))
    return ""


def get_answer(row: Dict[str, Any]) -> str:
    return norm_ws(row.get("answer"))


def boxed_values(text: str) -> List[str]:
    return [norm_ws(x) for x in BOX_RE.findall(text or "")]


def ends_with_final_box(text: str) -> bool:
    text = norm_ws(text)
    boxes = list(BOX_RE.finditer(text))
    if not boxes:
        return False
    return boxes[-1].end() == len(text)


def has_placeholder(text: str) -> bool:
    upper = str(text or "").upper()
    return any(p in upper for p in PLACEHOLDER_PATTERNS)


def validate_sft_like_row(row: Dict[str, Any], split_name: str) -> List[str]:
    problems: List[str] = []

    if row.get("__parse_error__"):
        return [f"parse_error:{row.get('__error__')}"]

    prompt = get_prompt(row)
    answer = get_answer(row)
    assistant = get_assistant_text(row)
    family = norm_ws(row.get("family"))

    if not prompt:
        problems.append("empty_prompt")
    if not assistant:
        problems.append("empty_assistant_or_target_text")
    if not answer:
        problems.append("empty_answer")
    if not family:
        problems.append("missing_family")

    if has_placeholder(prompt) or has_placeholder(assistant) or has_placeholder(answer):
        problems.append("placeholder_or_abstain_present")

    boxes = boxed_values(assistant)
    if len(boxes) != 1:
        problems.append(f"box_count_{len(boxes)}")
    else:
        if boxes[0] != answer:
            problems.append("boxed_answer_mismatch")
        if not ends_with_final_box(assistant):
            problems.append("text_after_final_box")

    if row.get("verification_status") not in (None, "PASS"):
        problems.append("verification_status_not_pass")

    try:
        ambiguity_count = int(row.get("ambiguity_count", 0) or 0)
    except Exception:
        ambiguity_count = 999999
    if ambiguity_count != 0:
        problems.append("ambiguity_count_nonzero")

    if row.get("trainable") is False:
        problems.append("trainable_false")

    if split_name.startswith("holdout") and "holdout_reason" not in row:
        problems.append("holdout_missing_reason")

    return problems


def validate_dpo_pair(row: Dict[str, Any], split_name: str) -> List[str]:
    problems: List[str] = []
    if row.get("__parse_error__"):
        return [f"parse_error:{row.get('__error__')}"]

    prompt = norm_ws(row.get("prompt"))
    chosen = norm_ws(row.get("chosen"))
    rejected = norm_ws(row.get("rejected"))
    answer = get_answer(row)
    negative_type = norm_ws(row.get("negative_type"))
    reason_rejected = norm_ws(row.get("reason_rejected"))

    if not prompt:
        problems.append("empty_prompt")
    if not chosen:
        problems.append("empty_chosen")
    if not rejected:
        problems.append("empty_rejected")
    if not answer:
        problems.append("empty_answer")
    if not negative_type:
        problems.append("missing_negative_type")
    if not reason_rejected:
        problems.append("missing_reason_rejected")

    if chosen == rejected:
        problems.append("chosen_equals_rejected")

    chosen_boxes = boxed_values(chosen)
    rejected_boxes = boxed_values(rejected)

    if len(chosen_boxes) != 1:
        problems.append(f"chosen_box_count_{len(chosen_boxes)}")
    else:
        if chosen_boxes[0] != answer:
            problems.append("chosen_boxed_answer_mismatch")
        if not ends_with_final_box(chosen):
            problems.append("chosen_text_after_box")

    format_negative = negative_type.startswith("format_") or negative_type == "right_answer_wrong_format"

    rejected_is_well_formed = len(rejected_boxes) == 1 and ends_with_final_box(rejected) and not has_placeholder(rejected)
    rejected_same_answer = len(rejected_boxes) == 1 and rejected_boxes[0] == answer

    if rejected_is_well_formed and rejected_same_answer and not format_negative:
        problems.append("rejected_accidentally_correct")

    if rejected_is_well_formed and rejected_same_answer and format_negative:
        problems.append("format_negative_is_not_actually_malformed")

    if has_placeholder(chosen):
        problems.append("chosen_placeholder_or_abstain")

    if negative_type == "format_abstain" and "ABSTAIN" not in rejected.upper():
        problems.append("format_abstain_missing_abstain")

    return problems


def audit_jsonl_file(path: Path, split_name: str, mode: str) -> Dict[str, Any]:
    counters = Counter()
    family_counts = Counter()
    style_counts = Counter()
    difficulty_counts = Counter()
    source_counts = Counter()
    answer_counts = Counter()
    prompt_hashes = Counter()
    id_hashes = Counter()
    prefix_counts = Counter()
    trace_lengths: List[int] = []
    examples_failed: List[Dict[str, Any]] = []

    total = 0
    for row in iter_jsonl(path):
        total += 1
        row_id = norm_ws(row.get("id"))
        prompt = get_prompt(row) if mode != "dpo" else norm_ws(row.get("prompt"))
        answer = get_answer(row)
        family = norm_ws(row.get("family"))
        style = norm_ws(row.get("prompt_style"))
        difficulty = norm_ws(row.get("difficulty"))
        source = norm_ws(row.get("source") or row.get("variant") or row.get("split"))

        if row_id:
            id_hashes[row_id] += 1
        if prompt:
            prompt_hashes[norm_prompt(prompt)] += 1
            prefix_counts[token_prefix(prompt)] += 1
        if answer:
            answer_counts[answer] += 1
        if family:
            family_counts[family] += 1
        if style:
            style_counts[style] += 1
        if difficulty:
            difficulty_counts[difficulty] += 1
        if source:
            source_counts[source] += 1

        assistant = get_assistant_text(row)
        if assistant:
            trace_lengths.append(len(assistant.split()))

        problems = validate_dpo_pair(row, split_name) if mode == "dpo" else validate_sft_like_row(row, split_name)

        if problems:
            counters["rows_with_problems"] += 1
            for p in problems:
                counters[p] += 1
            if len(examples_failed) < 20:
                examples_failed.append({
                    "id": row_id,
                    "family": family,
                    "prompt": prompt[:300],
                    "problems": problems,
                })

        # For SFT-like accepted rows, any ABSTAIN/placeholder anywhere is fatal.
        # For DPO rows, the rejected response is intentionally allowed to be malformed.
        # Therefore only the DPO chosen response is treated as accepted/trainable output.
        if mode == "dpo":
            chosen_text = norm_ws(row.get("chosen"))
            if has_placeholder(chosen_text):
                counters["chosen_placeholder_or_abstain"] += 1
        else:
            if has_placeholder(json.dumps(row, ensure_ascii=False)):
                counters["placeholder_or_abstain_anywhere"] += 1

    duplicate_prompt_count = sum(c - 1 for c in prompt_hashes.values() if c > 1)
    duplicate_id_count = sum(c - 1 for c in id_hashes.values() if c > 1)

    return {
        "path": str(path),
        "split": split_name,
        "mode": mode,
        "row_count": total,
        "rows_with_problems": counters["rows_with_problems"],
        "format_error_count": (
            counters["box_count_0"]
            + counters["box_count_2"]
            + counters["box_count_3"]
            + counters["text_after_final_box"]
            + counters["boxed_answer_mismatch"]
            + counters["empty_assistant_or_target_text"]
            + counters["chosen_box_count_0"]
            + counters["chosen_box_count_2"]
            + counters["chosen_text_after_box"]
            + counters["chosen_boxed_answer_mismatch"]
        ),
        "verification_fail_count": counters["verification_status_not_pass"],
        "ambiguity_accepted": counters["ambiguity_count_nonzero"],
        "abstain_accepted": counters["placeholder_or_abstain_present"] + counters["placeholder_or_abstain_anywhere"],
        "duplicate_prompt_count": duplicate_prompt_count,
        "duplicate_id_count": duplicate_id_count,
        "family_counts": dict(family_counts),
        "prompt_style_counts": dict(style_counts),
        "difficulty_counts": dict(difficulty_counts),
        "source_counts": dict(source_counts),
        "answer_top20": answer_counts.most_common(20),
        "dominant_prompt_prefix_top20": prefix_counts.most_common(20),
        "trace_length": {
            "min": min(trace_lengths) if trace_lengths else 0,
            "max": max(trace_lengths) if trace_lengths else 0,
            "mean": statistics.mean(trace_lengths) if trace_lengths else 0,
            "median": statistics.median(trace_lengths) if trace_lengths else 0,
        },
        "raw_problem_counts": dict(counters),
        "examples_failed": examples_failed,
    }


def collect_prompt_sets(out_dir: Path, file_map: Dict[str, str]) -> Dict[str, set]:
    sets: Dict[str, set] = {}
    for split, filename in file_map.items():
        path = out_dir / filename
        prompts = set()
        if path.exists():
            for row in iter_jsonl(path):
                p = get_prompt(row) if "dpo" not in split else norm_ws(row.get("prompt"))
                if p:
                    prompts.add(norm_prompt(p))
        sets[split] = prompts
    return sets


def compute_split_leakage(prompt_sets: Dict[str, set]) -> Tuple[int, List[Dict[str, Any]]]:
    leakage = 0
    examples: List[Dict[str, Any]] = []
    items = list(prompt_sets.items())
    for i in range(len(items)):
        split_a, prompts_a = items[i]
        for j in range(i + 1, len(items)):
            split_b, prompts_b = items[j]
            overlap = prompts_a.intersection(prompts_b)
            if overlap:
                leakage += len(overlap)
                examples.append({
                    "split_a": split_a,
                    "split_b": split_b,
                    "overlap_count": len(overlap),
                    "examples": sorted(list(overlap))[:5],
                })
    return leakage, examples[:30]


def sum_counter_from_reports(reports: Iterable[Dict[str, Any]], key: str) -> int:
    return int(sum(int(r.get(key, 0) or 0) for r in reports))


def build_teacher_stress_report(out_dir: Path) -> Dict[str, Any]:
    reports: Dict[str, Dict[str, Any]] = {}
    all_reports: List[Dict[str, Any]] = []
    missing_files: List[str] = []

    for split, filename in {**CORE_FILES, **HOLDOUT_FILES, **VARIANT_FILES}.items():
        path = out_dir / filename
        if not path.exists():
            missing_files.append(filename)
            continue
        report = audit_jsonl_file(path, split, mode="sft")
        reports[split] = report
        all_reports.append(report)

    for split, filename in DPO_FILES.items():
        path = out_dir / filename
        if not path.exists():
            missing_files.append(filename)
            continue
        report = audit_jsonl_file(path, split, mode="dpo")
        reports[split] = report
        all_reports.append(report)

    total_rows = sum_counter_from_reports(all_reports, "row_count")
    rows_with_problems = sum_counter_from_reports(all_reports, "rows_with_problems")
    format_error_count = sum_counter_from_reports(all_reports, "format_error_count")
    verification_fail_count = sum_counter_from_reports(all_reports, "verification_fail_count")
    ambiguity_accepted = sum_counter_from_reports(all_reports, "ambiguity_accepted")
    abstain_accepted = sum_counter_from_reports(all_reports, "abstain_accepted")

    per_family = Counter()
    per_style = Counter()
    per_difficulty = Counter()
    for r in all_reports:
        per_family.update(r.get("family_counts", {}))
        per_style.update(r.get("prompt_style_counts", {}))
        per_difficulty.update(r.get("difficulty_counts", {}))

    pass_rate = 1.0 if total_rows == 0 else (total_rows - rows_with_problems) / total_rows

    blocked_reasons: List[str] = []
    if missing_files:
        blocked_reasons.append(f"missing_files:{missing_files}")
    if format_error_count:
        blocked_reasons.append("format_errors_nonzero")
    if verification_fail_count:
        blocked_reasons.append("verification_failures_nonzero")
    if ambiguity_accepted:
        blocked_reasons.append("ambiguity_accepted_nonzero")
    if abstain_accepted:
        blocked_reasons.append("abstain_or_placeholder_nonzero")
    if pass_rate < 0.995:
        blocked_reasons.append(f"pass_rate_below_0.995:{pass_rate:.6f}")

    composed_seen = set(k for k in per_family if k.startswith("composed_"))
    if len(composed_seen.intersection(COMPOSED_FAMILIES)) < 9:
        blocked_reasons.append("not_all_9_composed_families_seen")

    original_seen = set(k for k in per_family if k in ORIGINAL_FAMILIES)
    if len(original_seen) < 11:
        blocked_reasons.append("not_all_11_original_families_seen")

    status = "PASS" if not blocked_reasons else "FAIL"

    return {
        "schema_version": 1,
        "created_by": "DAY1X_TEACHER_STRESS_REPORT",
        "status": status,
        "total_rows_checked": total_rows,
        "rows_with_problems": rows_with_problems,
        "verification_pass_rate": pass_rate,
        "format_error_count": format_error_count,
        "verification_fail_count": verification_fail_count,
        "ambiguity_accepted": ambiguity_accepted,
        "abstain_accepted": abstain_accepted,
        "original_families_seen": sorted(original_seen),
        "composed_families_seen": sorted(composed_seen.intersection(COMPOSED_FAMILIES)),
        "per_family_counts": dict(per_family),
        "per_prompt_style_counts": dict(per_style),
        "per_difficulty_counts": dict(per_difficulty),
        "file_reports": reports,
        "blocked_reasons": blocked_reasons,
        "package_authorized": False,
        "submission_authorized": False,
        "leaderboard_claim": False,
        "no_0_93_evidence": True,
        "no_0_95_evidence": True,
    }


def build_diversity_and_leakage_reports(out_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    core_prompt_sets = collect_prompt_sets(out_dir, {**CORE_FILES, **HOLDOUT_FILES})
    split_leakage_count, split_leakage_examples = compute_split_leakage(core_prompt_sets)

    train_path = out_dir / CORE_FILES["train_100k"]
    train_audit = audit_jsonl_file(train_path, "train_100k", mode="sft") if train_path.exists() else {}

    train_rows = int(train_audit.get("row_count", 0) or 0)
    family_counts = Counter(train_audit.get("family_counts", {}))
    style_counts = Counter(train_audit.get("prompt_style_counts", {}))
    difficulty_counts = Counter(train_audit.get("difficulty_counts", {}))

    max_family_fraction = max((c / train_rows for c in family_counts.values()), default=0.0)
    max_style_fraction = max((c / train_rows for c in style_counts.values()), default=0.0)
    max_difficulty_fraction = max((c / train_rows for c in difficulty_counts.values()), default=0.0)

    dominant_prefixes = train_audit.get("dominant_prompt_prefix_top20", [])
    dominant_prefix_fraction = 0.0
    if dominant_prefixes and train_rows:
        dominant_prefix_fraction = dominant_prefixes[0][1] / train_rows

    answer_top20 = train_audit.get("answer_top20", [])
    answer_spike_fraction = 0.0
    if answer_top20 and train_rows:
        answer_spike_fraction = answer_top20[0][1] / train_rows

    diversity_blockers: List[str] = []
    if train_rows < 100000:
        diversity_blockers.append("train_100k_missing_or_under_count")
    if len(set(family_counts).intersection(ORIGINAL_FAMILIES)) < 11:
        diversity_blockers.append("original_family_coverage_incomplete")
    if len(set(family_counts).intersection(COMPOSED_FAMILIES)) < 9:
        diversity_blockers.append("composed_family_coverage_incomplete")
    if max_style_fraction > 0.35:
        diversity_blockers.append(f"prompt_style_dominates:{max_style_fraction:.4f}")
    if max_family_fraction > 0.20:
        diversity_blockers.append(f"family_dominates:{max_family_fraction:.4f}")
    if dominant_prefix_fraction > 0.10:
        diversity_blockers.append(f"template_prefix_dominates:{dominant_prefix_fraction:.4f}")
    if answer_spike_fraction > 0.08:
        diversity_blockers.append(f"answer_distribution_spike:{answer_spike_fraction:.4f}")

    leakage_blockers: List[str] = []
    if split_leakage_count:
        leakage_blockers.append("core_split_or_holdout_prompt_leakage")
    if int(train_audit.get("duplicate_prompt_count", 0) or 0):
        leakage_blockers.append("train_duplicate_prompt_count_nonzero")
    if int(train_audit.get("duplicate_id_count", 0) or 0):
        leakage_blockers.append("train_duplicate_id_count_nonzero")

    diversity_status = "PASS" if not diversity_blockers else "FAIL"
    leakage_status = "PASS" if not leakage_blockers else "FAIL"

    leakage_report = {
        "schema_version": 1,
        "created_by": "DAY1X_PROMPT_TEMPLATE_LEAKAGE_REPORT",
        "status": leakage_status,
        "split_leakage_count": split_leakage_count,
        "split_leakage_examples": split_leakage_examples,
        "train_duplicate_prompt_count": train_audit.get("duplicate_prompt_count", 0),
        "train_duplicate_id_count": train_audit.get("duplicate_id_count", 0),
        "dominant_prompt_prefix_top20": dominant_prefixes,
        "dominant_prefix_fraction": dominant_prefix_fraction,
        "blocked_reasons": leakage_blockers,
    }

    diversity_report = {
        "schema_version": 1,
        "created_by": "DAY1X_DIVERSITY_REPORT",
        "status": diversity_status,
        "train_rows": train_rows,
        "rows_by_family": dict(family_counts),
        "rows_by_prompt_style": dict(style_counts),
        "rows_by_difficulty": dict(difficulty_counts),
        "max_family_fraction": max_family_fraction,
        "max_prompt_style_fraction": max_style_fraction,
        "max_difficulty_fraction": max_difficulty_fraction,
        "answer_top20": answer_top20,
        "answer_spike_fraction": answer_spike_fraction,
        "trace_length": train_audit.get("trace_length", {}),
        "blocked_reasons": diversity_blockers,
    }

    original_count = sum(family_counts.get(f, 0) for f in ORIGINAL_FAMILIES)
    composed_count = sum(family_counts.get(f, 0) for f in COMPOSED_FAMILIES)
    public_style_count = sum(c for style, c in style_counts.items() if str(style).startswith("public_style"))
    format_stress_count = sum(c for key, c in train_audit.get("source_counts", {}).items() if "format" in str(key).lower())

    family_composition_report = {
        "schema_version": 1,
        "created_by": "DAY1X_FAMILY_COMPOSITION_REPORT",
        "status": "PASS" if not diversity_blockers else "FAIL",
        "train_rows": train_rows,
        "original_family_rows": original_count,
        "composed_family_rows": composed_count,
        "public_style_rows": public_style_count,
        "format_stress_rows": format_stress_count,
        "original_fraction": original_count / train_rows if train_rows else 0.0,
        "composed_fraction": composed_count / train_rows if train_rows else 0.0,
        "public_style_fraction": public_style_count / train_rows if train_rows else 0.0,
        "rows_by_family": dict(family_counts),
        "original_families_present": sorted(set(family_counts).intersection(ORIGINAL_FAMILIES)),
        "composed_families_present": sorted(set(family_counts).intersection(COMPOSED_FAMILIES)),
        "blocked_reasons": diversity_blockers,
    }

    return leakage_report, diversity_report, family_composition_report


def load_optional_json(out_dir: Path, filename: str) -> Dict[str, Any]:
    path = out_dir / filename
    if not path.exists():
        return {"status": "MISSING", "path": str(path)}
    try:
        return read_json(path)
    except Exception as exc:
        return {"status": "BROKEN", "path": str(path), "error": repr(exc)}


def build_final_manifest(
    out_dir: Path,
    stress_report: Dict[str, Any],
    leakage_report: Dict[str, Any],
    diversity_report: Dict[str, Any],
    family_report: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    expanded_manifest = load_optional_json(out_dir, "day1x_expanded_data_manifest.json")
    expanded_decision = load_optional_json(out_dir, "day1x_decision_report_100k_stage.json")
    dpo_variant_manifest = load_optional_json(out_dir, "day1x_dpo_variant_manifest.json")
    dpo_variant_decision = load_optional_json(out_dir, "day1x_decision_report_dpo_variant_stage.json")

    blocked: List[str] = []

    required_pass_items = {
        "expanded_manifest": expanded_manifest,
        "expanded_decision": expanded_decision,
        "dpo_variant_manifest": dpo_variant_manifest,
        "dpo_variant_decision": dpo_variant_decision,
        "stress_report": stress_report,
        "leakage_report": leakage_report,
        "diversity_report": diversity_report,
        "family_composition_report": family_report,
    }

    for name, report in required_pass_items.items():
        if report.get("status") != "PASS":
            blocked.append(f"{name}_not_pass:{report.get('status')}")

    unsafe_flags = []
    for name, report in required_pass_items.items():
        if report.get("package_authorized") is True:
            unsafe_flags.append(f"{name}.package_authorized_true")
        if report.get("submission_authorized") is True:
            unsafe_flags.append(f"{name}.submission_authorized_true")
        if report.get("leaderboard_claim") is True:
            unsafe_flags.append(f"{name}.leaderboard_claim_true")
    blocked.extend(unsafe_flags)

    safety_counters = {
        "format_error_count": int(stress_report.get("format_error_count", 0) or 0),
        "verification_fail_count": int(stress_report.get("verification_fail_count", 0) or 0),
        "ambiguity_accepted": int(stress_report.get("ambiguity_accepted", 0) or 0),
        "abstain_accepted": int(stress_report.get("abstain_accepted", 0) or 0),
        "split_leakage_count": int(leakage_report.get("split_leakage_count", 0) or 0),
        "duplicate_prompt_count": int(leakage_report.get("train_duplicate_prompt_count", 0) or 0),
        "duplicate_id_count": int(leakage_report.get("train_duplicate_id_count", 0) or 0),
    }

    for key, value in safety_counters.items():
        if value:
            blocked.append(f"{key}_nonzero:{value}")

    status = "PASS" if not blocked else "FAIL"

    selected_stage = (
        expanded_manifest.get("selected_stage_for_day2")
        or expanded_decision.get("selected_stage_for_day2")
        or "unknown"
    )

    recommended_variants = (
        dpo_variant_manifest.get("recommended_training_variants")
        or dpo_variant_decision.get("recommended_training_variants")
        or []
    )

    manifest = {
        "schema_version": 1,
        "created_by": "DAY1X_STRESS_DIVERSITY_LEAKAGE_AND_FINAL_MANIFEST",
        "status": status,
        "selected_stage_for_day2": selected_stage,
        "stress_status": stress_report.get("status"),
        "diversity_status": diversity_report.get("status"),
        "leakage_status": leakage_report.get("status"),
        "family_composition_status": family_report.get("status"),
        "dpo_variant_status": dpo_variant_manifest.get("status"),
        "expanded_data_status": expanded_manifest.get("status"),
        "safety_counters": safety_counters,
        "counts": {
            "train_100k": count_jsonl(out_dir / CORE_FILES["train_100k"]),
            "eval_10k": count_jsonl(out_dir / CORE_FILES["eval_10k"]),
            "probe_10k": count_jsonl(out_dir / CORE_FILES["probe_10k"]),
            "dpo_train_pairs": count_jsonl(out_dir / DPO_FILES["dpo_train"]),
            "dpo_eval_pairs": count_jsonl(out_dir / DPO_FILES["dpo_eval"]),
            "holdouts_total": sum(count_jsonl(out_dir / f) for f in HOLDOUT_FILES.values()),
        },
        "original_families_present": family_report.get("original_families_present", []),
        "composed_families_present": family_report.get("composed_families_present", []),
        "recommended_training_variants": recommended_variants,
        "ready_for_day2_lora_training": status == "PASS",
        "ready_for_next_phase": status == "PASS",
        "next_phase": "DAY2_LORA_TRAINING_AND_ADAPTER_EVAL",
        "training_authorized": status == "PASS",
        "package_authorized": False,
        "submission_authorized": False,
        "leaderboard_claim": False,
        "no_0_93_evidence": True,
        "no_0_95_evidence": True,
        "blocked_reasons": blocked,
    }

    decision = {
        "schema_version": 1,
        "created_by": "DAY1X_FINAL_DECISION_REPORT",
        "status": status,
        "day1x_option_c_complete": status == "PASS",
        "selected_stage_for_day2": selected_stage,
        "recommended_training_variants": recommended_variants,
        "ready_for_day2_lora_training": status == "PASS",
        "next_phase": "DAY2_LORA_TRAINING_AND_ADAPTER_EVAL",
        "blocked_reasons": blocked,
        "training_authorized": status == "PASS",
        "package_authorized": False,
        "submission_authorized": False,
        "leaderboard_claim": False,
        "no_0_93_evidence": True,
        "no_0_95_evidence": True,
    }

    return manifest, decision


def run(out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    stress_report = build_teacher_stress_report(out_dir)
    write_json(out_dir / "day1x_teacher_stress_report.json", stress_report)

    leakage_report, diversity_report, family_report = build_diversity_and_leakage_reports(out_dir)
    write_json(out_dir / "day1x_prompt_template_leakage_report.json", leakage_report)
    write_json(out_dir / "day1x_diversity_report.json", diversity_report)
    write_json(out_dir / "day1x_family_composition_report.json", family_report)

    manifest, decision = build_final_manifest(
        out_dir=out_dir,
        stress_report=stress_report,
        leakage_report=leakage_report,
        diversity_report=diversity_report,
        family_report=family_report,
    )

    write_json(out_dir / "day1x_final_manifest.json", manifest)
    write_json(out_dir / "day1x_final_decision_report.json", decision)

    return {
        "stress_report": stress_report,
        "leakage_report": leakage_report,
        "diversity_report": diversity_report,
        "family_report": family_report,
        "manifest": manifest,
        "decision": decision,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="artifacts/sprint11")
    args = parser.parse_args()

    result = run(Path(args.out_dir))

    summary = {
        "teacher_stress_status": result["stress_report"]["status"],
        "leakage_status": result["leakage_report"]["status"],
        "diversity_status": result["diversity_report"]["status"],
        "family_composition_status": result["family_report"]["status"],
        "final_manifest_status": result["manifest"]["status"],
        "final_decision_status": result["decision"]["status"],
        "ready_for_day2_lora_training": result["decision"]["ready_for_day2_lora_training"],
        "blocked_reasons": result["decision"]["blocked_reasons"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

