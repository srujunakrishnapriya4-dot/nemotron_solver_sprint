from __future__ import annotations

import json
import re
from collections import Counter
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


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON file is not an object: {path}")
    return data


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
                row = json.loads(line)
            except Exception as exc:
                yield {
                    "__parse_error__": True,
                    "__line_no__": line_no,
                    "__error__": repr(exc),
                    "__raw__": line[:500],
                }
                continue
            if not isinstance(row, dict):
                yield {
                    "__parse_error__": True,
                    "__line_no__": line_no,
                    "__error__": "jsonl row is not a JSON object",
                    "__raw__": line[:500],
                }
                continue
            yield row


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in iter_jsonl(path))


def norm_ws(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "")).strip()


def has_placeholder(text: Any) -> bool:
    upper = str(text or "").upper()
    return any(p in upper for p in PLACEHOLDER_PATTERNS)


def boxed_values(text: str) -> List[str]:
    return [norm_ws(x) for x in BOX_RE.findall(text or "")]


def ends_with_final_box(text: str) -> bool:
    text = norm_ws(text)
    boxes = list(BOX_RE.finditer(text))
    return bool(boxes) and boxes[-1].end() == len(text)


def get_user_prompt(row: Dict[str, Any]) -> str:
    if "prompt" in row:
        return norm_ws(row.get("prompt"))
    messages = row.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return norm_ws(msg.get("content"))
    return ""


def get_assistant_target(row: Dict[str, Any]) -> str:
    if "target_text" in row:
        return norm_ws(row.get("target_text"))
    messages = row.get("messages")
    if isinstance(messages, list):
        for msg in messages:
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

    if has_placeholder(prompt) or has_placeholder(target) or has_placeholder(answer):
        problems.append("placeholder_or_abstain_present")

    boxes = boxed_values(target)
    if len(boxes) != 1:
        problems.append(f"box_count_{len(boxes)}")
    else:
        if boxes[0] != answer:
            problems.append("boxed_answer_mismatch")
        if not ends_with_final_box(target):
            problems.append("text_after_final_box")

    if row.get("verification_status") not in (None, "PASS"):
        problems.append("verification_status_not_pass")

    try:
        ambiguity_count = int(row.get("ambiguity_count", 0) or 0)
    except Exception:
        ambiguity_count = 999999
    if ambiguity_count != 0:
        problems.append("ambiguity_nonzero")

    return problems


def validate_dpo_pair(row: Dict[str, Any]) -> List[str]:
    problems: List[str] = []

    if row.get("__parse_error__"):
        return [f"parse_error:{row.get('__error__')}"]

    prompt = norm_ws(row.get("prompt"))
    chosen = norm_ws(row.get("chosen"))
    rejected = norm_ws(row.get("rejected"))
    answer = get_answer(row)

    if not prompt:
        problems.append("empty_prompt")
    if not chosen:
        problems.append("empty_chosen")
    if not rejected:
        problems.append("empty_rejected")
    if not answer:
        problems.append("empty_answer")

    if chosen == rejected:
        problems.append("chosen_equals_rejected")

    if has_placeholder(chosen):
        problems.append("chosen_placeholder_or_abstain")

    chosen_boxes = boxed_values(chosen)
    if len(chosen_boxes) != 1:
        problems.append(f"chosen_box_count_{len(chosen_boxes)}")
    else:
        if chosen_boxes[0] != answer:
            problems.append("chosen_boxed_answer_mismatch")
        if not ends_with_final_box(chosen):
            problems.append("chosen_text_after_box")

    if not norm_ws(row.get("negative_type")):
        problems.append("missing_negative_type")
    if not norm_ws(row.get("reason_rejected")):
        problems.append("missing_reason_rejected")

    return problems


def summarize_sft_file(path: Path, sample_limit: Optional[int] = None) -> Dict[str, Any]:
    row_count = 0
    problem_count = 0
    problem_counter: Counter[str] = Counter()
    family_counter: Counter[str] = Counter()
    prompt_style_counter: Counter[str] = Counter()
    difficulty_counter: Counter[str] = Counter()
    examples_failed: List[Dict[str, Any]] = []
    max_target_tokens = 0
    total_target_tokens = 0

    for row in iter_jsonl(path):
        row_count += 1
        if sample_limit is not None and row_count > sample_limit:
            break

        family_counter[norm_ws(row.get("family"))] += 1
        prompt_style_counter[norm_ws(row.get("prompt_style"))] += 1
        difficulty_counter[norm_ws(row.get("difficulty"))] += 1

        target = get_assistant_target(row)
        tokens = len(target.split())
        max_target_tokens = max(max_target_tokens, tokens)
        total_target_tokens += tokens

        problems = validate_sft_row(row)
        if problems:
            problem_count += 1
            for p in problems:
                problem_counter[p] += 1
            if len(examples_failed) < 10:
                examples_failed.append({
                    "id": row.get("id"),
                    "family": row.get("family"),
                    "prompt": get_user_prompt(row)[:250],
                    "problems": problems,
                })

    return {
        "path": str(path),
        "exists": path.exists(),
        "row_count": row_count if sample_limit is None else min(row_count, sample_limit),
        "sample_limited": sample_limit is not None,
        "problem_count": problem_count,
        "problem_counts": dict(problem_counter),
        "family_counts": {k: v for k, v in family_counter.items() if k},
        "prompt_style_counts": {k: v for k, v in prompt_style_counter.items() if k},
        "difficulty_counts": {k: v for k, v in difficulty_counter.items() if k},
        "max_target_tokens": max_target_tokens,
        "mean_target_tokens": (total_target_tokens / row_count) if row_count else 0.0,
        "examples_failed": examples_failed,
    }


def summarize_dpo_file(path: Path, sample_limit: Optional[int] = None) -> Dict[str, Any]:
    row_count = 0
    problem_count = 0
    problem_counter: Counter[str] = Counter()
    family_counter: Counter[str] = Counter()
    negative_type_counter: Counter[str] = Counter()
    examples_failed: List[Dict[str, Any]] = []

    for row in iter_jsonl(path):
        row_count += 1
        if sample_limit is not None and row_count > sample_limit:
            break

        family_counter[norm_ws(row.get("family"))] += 1
        negative_type_counter[norm_ws(row.get("negative_type"))] += 1

        problems = validate_dpo_pair(row)
        if problems:
            problem_count += 1
            for p in problems:
                problem_counter[p] += 1
            if len(examples_failed) < 10:
                examples_failed.append({
                    "id": row.get("id"),
                    "family": row.get("family"),
                    "negative_type": row.get("negative_type"),
                    "problems": problems,
                })

    return {
        "path": str(path),
        "exists": path.exists(),
        "row_count": row_count if sample_limit is None else min(row_count, sample_limit),
        "sample_limited": sample_limit is not None,
        "problem_count": problem_count,
        "problem_counts": dict(problem_counter),
        "family_counts": {k: v for k, v in family_counter.items() if k},
        "negative_type_counts": {k: v for k, v in negative_type_counter.items() if k},
        "examples_failed": examples_failed,
    }


def resolve_existing_file(out_dir: Path, candidates: List[str]) -> Tuple[Optional[Path], List[str]]:
    checked = []
    for name in candidates:
        p = out_dir / name
        checked.append(str(p))
        if p.exists():
            return p, checked
    return None, checked


DAY2_DATASET_CANDIDATES = {
    "direct": ["day1x_sft_direct.jsonl"],
    "short_trace": ["day1x_sft_short_trace.jsonl"],
    "mixed_curriculum": ["day1x_sft_mixed_curriculum.jsonl"],
    "format_heavy": ["day1x_sft_format_heavy.jsonl"],
    "composed_heavy": ["day1x_sft_composed_heavy.jsonl"],
    "public_style_heavy": ["day1x_sft_public_style_heavy.jsonl", "day1x_public_style_heavy.jsonl"],
    "hard_p2_heavy": ["day1x_sft_hard_p2_heavy.jsonl", "day1x_hard_p2_heavy.jsonl"],
    "dpo_train": ["day1x_dpo_train_pairs_20k.jsonl"],
    "dpo_eval": ["day1x_dpo_eval_pairs_2k.jsonl"],
}


def discover_day2_datasets(out_dir: Path) -> Dict[str, Dict[str, Any]]:
    discovered: Dict[str, Dict[str, Any]] = {}
    for key, candidates in DAY2_DATASET_CANDIDATES.items():
        path, checked = resolve_existing_file(out_dir, candidates)
        discovered[key] = {
            "key": key,
            "path": str(path) if path else None,
            "exists": path is not None,
            "checked": checked,
            "kind": "dpo" if key.startswith("dpo_") else "sft",
        }
    return discovered
