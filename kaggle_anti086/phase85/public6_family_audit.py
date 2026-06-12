from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PUBLIC6 = [
    "bit_manipulation",
    "gravity_numeric_formula",
    "unit_conversion",
    "word_cipher",
    "roman_numeral",
    "equation_transform",
]


ROMAN_TABLE = [
    (1000, "M"),
    (900, "CM"),
    (500, "D"),
    (400, "CD"),
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
]


def norm_text(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "")).strip()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"missing_csv:{path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    required = {"id", "prompt"}
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"csv_missing_columns:{path}:{sorted(missing)}")

    return rows


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def roman_from_int(n: int) -> str:
    if n <= 0 or n >= 4000:
        raise ValueError(f"roman_out_of_range:{n}")

    out = []
    remaining = n
    for val, sym in ROMAN_TABLE:
        while remaining >= val:
            out.append(sym)
            remaining -= val
    return "".join(out)


def decimal_places_from_gold(gold: str) -> int:
    gold = norm_text(gold)
    if "." not in gold:
        return 0
    return len(gold.split(".")[-1])


def format_float_like_gold(value: float, gold: str) -> str:
    places = decimal_places_from_gold(gold)
    if places == 0:
        return str(int(round(value)))
    return f"{value:.{places}f}"


def safe_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def classify_family(prompt: str) -> Tuple[str, str]:
    p = prompt.lower()

    if "8-bit binary" in p or "bit manipulation" in p:
        return "bit_manipulation", "bit_manipulation_binary"

    if "gravitational constant" in p or "d = 0.5*g*t^2" in p or "0.5*g*t" in p:
        return "gravity_numeric_formula", "gravity_changed_g"

    if "secret unit conversion" in p or "now convert:" in p and "becomes" in p:
        return "unit_conversion", "secret_linear_scale"

    if "secret encryption" in p or "decrypt the following text" in p or "now decrypt:" in p:
        return "word_cipher", "word_or_char_cipher"

    if "wonderland numeral" in p or "different numeral system" in p:
        return "roman_numeral", "roman_integer_conversion"

    if "secret set of transformation rules" in p or "transformation rules is applied to equations" in p:
        digit_count = len(re.findall(r"\d", prompt))
        punct_count = len(re.findall(r"[^A-Za-z0-9\s]", prompt))
        if digit_count >= punct_count:
            return "equation_transform", "equation_transform_numeric_symbolic"
        return "equation_transform", "equation_transform_punctuation_symbolic"

    return "unknown", "unknown"


def parse_last_int_query(prompt: str) -> Optional[int]:
    tail = prompt[-600:]
    patterns = [
        r"Now\s+convert\s*:?\s*(-?\d+)",
        r"convert\s+(-?\d+)",
        r"for\s+(-?\d+)",
        r"given\s+(-?\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, tail, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
    nums = re.findall(r"-?\d+", tail)
    if nums:
        return int(nums[-1])
    return None


def roman_teacher(prompt: str, gold: str) -> Dict[str, Any]:
    n = parse_last_int_query(prompt)
    if n is None:
        return {"covered": False, "reason": "query_int_not_parsed"}

    try:
        pred = roman_from_int(n)
    except Exception as e:
        return {"covered": False, "reason": str(e)}

    ok = pred == norm_text(gold)
    return {
        "covered": ok,
        "prediction": pred,
        "reason": "verified" if ok else "prediction_mismatch",
    }


def parse_becomes_examples(prompt: str) -> List[Tuple[float, float]]:
    pairs = []
    # examples like "10.08 m becomes 6.69"
    for m in re.finditer(
        r"(-?\d+(?:\.\d+)?)\s*[A-Za-z/%°]*\s+becomes\s+(-?\d+(?:\.\d+)?)",
        prompt,
        flags=re.IGNORECASE,
    ):
        a = safe_float(m.group(1))
        b = safe_float(m.group(2))
        if a is not None and b is not None and abs(a) > 1e-12:
            pairs.append((a, b))
    return pairs


def parse_unit_query(prompt: str) -> Optional[float]:
    tail = prompt[-500:]
    m = re.search(r"Now\s+convert\s*:?\s*(-?\d+(?:\.\d+)?)", tail, flags=re.IGNORECASE)
    if m:
        return safe_float(m.group(1))
    nums = re.findall(r"-?\d+(?:\.\d+)?", tail)
    if nums:
        return safe_float(nums[-1])
    return None


def unit_conversion_teacher(prompt: str, gold: str) -> Dict[str, Any]:
    pairs = parse_becomes_examples(prompt)
    q = parse_unit_query(prompt)
    if len(pairs) < 2:
        return {"covered": False, "reason": "not_enough_examples"}
    if q is None:
        return {"covered": False, "reason": "query_value_not_parsed"}

    ratios = [b / a for a, b in pairs if abs(a) > 1e-12]
    if not ratios:
        return {"covered": False, "reason": "no_valid_ratios"}

    k = statistics.median(ratios)
    pred = format_float_like_gold(q * k, gold)
    ok = pred == norm_text(gold)
    return {
        "covered": ok,
        "prediction": pred,
        "scale": k,
        "reason": "verified" if ok else "prediction_mismatch",
    }


def parse_gravity_examples(prompt: str) -> List[Tuple[float, float]]:
    examples = []
    # Handles: "For t = 1.37s, distance = 14.92 m"
    for m in re.finditer(
        r"t\s*=\s*(-?\d+(?:\.\d+)?)\s*s?[^.\n\r]*?distance\s*=\s*(-?\d+(?:\.\d+)?)",
        prompt,
        flags=re.IGNORECASE,
    ):
        t = safe_float(m.group(1))
        d = safe_float(m.group(2))
        if t is not None and d is not None and abs(t) > 1e-12:
            examples.append((t, d))
    return examples


def parse_gravity_query(prompt: str) -> Optional[float]:
    tail = prompt[-600:]
    # Use last t=... occurrence, usually query.
    vals = re.findall(r"t\s*=\s*(-?\d+(?:\.\d+)?)\s*s?", tail, flags=re.IGNORECASE)
    if vals:
        return safe_float(vals[-1])
    return None


def gravity_teacher(prompt: str, gold: str) -> Dict[str, Any]:
    examples = parse_gravity_examples(prompt)
    q = parse_gravity_query(prompt)

    if len(examples) < 2:
        return {"covered": False, "reason": "not_enough_examples"}
    if q is None:
        return {"covered": False, "reason": "query_t_not_parsed"}

    gs = []
    for t, d in examples:
        if abs(t) > 1e-12:
            gs.append(2.0 * d / (t * t))

    if not gs:
        return {"covered": False, "reason": "no_valid_g_values"}

    g = statistics.median(gs)
    pred = format_float_like_gold(0.5 * g * q * q, gold)
    ok = pred == norm_text(gold)
    return {
        "covered": ok,
        "prediction": pred,
        "g": g,
        "reason": "verified" if ok else "prediction_mismatch",
    }


def parse_cipher_examples(prompt: str) -> Tuple[List[Tuple[str, str]], Optional[str]]:
    examples = []
    query = None

    lines = [x.strip() for x in prompt.splitlines() if x.strip()]
    for line in lines:
        q = re.search(r"Now\s+decrypt\s*:?\s*(.+)$", line, flags=re.IGNORECASE)
        if q:
            query = q.group(1).strip()
            continue

        # Handles "abc def -> cat dog"
        m = re.match(r"(.+?)\s*(?:->|=>|=)\s*(.+)$", line)
        if m:
            left = m.group(1).strip()
            right = m.group(2).strip()
            if left and right and "decrypt" not in left.lower():
                examples.append((left, right))

    return examples, query


def learn_char_map(examples: List[Tuple[str, str]]) -> Optional[Dict[str, str]]:
    cmap: Dict[str, str] = {}
    reverse: Dict[str, str] = {}

    for enc, dec in examples:
        enc_words = enc.split()
        dec_words = dec.split()
        if len(enc_words) != len(dec_words):
            continue

        for ew, dw in zip(enc_words, dec_words):
            if len(ew) != len(dw):
                continue

            for a, b in zip(ew, dw):
                if a in cmap and cmap[a] != b:
                    return None
                if b in reverse and reverse[b] != a:
                    return None
                cmap[a] = b
                reverse[b] = a

    return cmap


def word_cipher_teacher(prompt: str, gold: str) -> Dict[str, Any]:
    examples, query = parse_cipher_examples(prompt)
    if not examples:
        return {"covered": False, "reason": "no_examples_parsed"}
    if not query:
        return {"covered": False, "reason": "query_not_parsed"}

    word_map: Dict[str, str] = {}
    for enc, dec in examples:
        ew = enc.split()
        dw = dec.split()
        if len(ew) == len(dw):
            for a, b in zip(ew, dw):
                word_map[a] = b

    # First try exact word mapping.
    qwords = query.split()
    if qwords and all(w in word_map for w in qwords):
        pred = " ".join(word_map[w] for w in qwords)
        ok = pred == norm_text(gold)
        return {
            "covered": ok,
            "prediction": pred,
            "mode": "word_map",
            "reason": "verified" if ok else "prediction_mismatch",
        }

    # Then try character substitution.
    cmap = learn_char_map(examples)
    if not cmap:
        return {"covered": False, "reason": "char_map_conflict_or_empty"}

    decoded_chars = []
    missing = []
    for ch in query:
        if ch.isspace():
            decoded_chars.append(ch)
        elif ch in cmap:
            decoded_chars.append(cmap[ch])
        else:
            missing.append(ch)
            decoded_chars.append("?")

    if missing:
        return {
            "covered": False,
            "prediction": "".join(decoded_chars),
            "reason": "missing_chars",
            "missing_chars": sorted(set(missing)),
        }

    pred = norm_text("".join(decoded_chars))
    ok = pred == norm_text(gold)
    return {
        "covered": ok,
        "prediction": pred,
        "mode": "char_map",
        "reason": "verified" if ok else "prediction_mismatch",
    }


def answer_format_valid(family: str, answer: str) -> bool:
    ans = norm_text(answer)
    if not ans:
        return False

    if family == "bit_manipulation":
        return bool(re.fullmatch(r"[01]{8}", ans))

    if family == "gravity_numeric_formula":
        return bool(re.fullmatch(r"-?\d+(?:\.\d+)?", ans))

    if family == "unit_conversion":
        return bool(re.fullmatch(r"-?\d+(?:\.\d+)?", ans))

    if family == "roman_numeral":
        return bool(re.fullmatch(r"[IVXLCDM]+", ans))

    if family == "word_cipher":
        return bool(re.fullmatch(r"[A-Za-z ]+", ans))

    if family == "equation_transform":
        return len(ans) > 0

    return len(ans) > 0


def teacher_probe(family: str, prompt: str, answer: str) -> Dict[str, Any]:
    """
    PHASE 8.6 HARD OVERRIDE.

    Route all teacher coverage checks through public6_teachers.py.
    Do not use the old Phase 8.5 placeholder logic for word_cipher,
    bit_manipulation, or equation_transform.
    """
    try:
        from kaggle_anti086.phase85.public6_teachers import teacher_probe as phase86_teacher_probe
    except ModuleNotFoundError:
        from public6_teachers import teacher_probe as phase86_teacher_probe
    return phase86_teacher_probe(family, prompt, answer)

def summarize_rows(rows: List[Dict[str, str]], has_gold: bool) -> Dict[str, Any]:
    fams: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        rid = norm_text(row.get("id"))
        prompt = row.get("prompt", "")
        answer = row.get("answer", "") if has_gold else ""

        family, subtype = classify_family(prompt)
        if family not in fams:
            fams[family] = {
                "rows": 0,
                "subtypes": {},
                "answer_format_valid": 0,
                "teacher_verified": 0,
                "teacher_uncovered": 0,
                "teacher_reasons": {},
                "examples": [],
            }

        f = fams[family]
        f["rows"] += 1
        f["subtypes"][subtype] = f["subtypes"].get(subtype, 0) + 1

        if has_gold and answer_format_valid(family, answer):
            f["answer_format_valid"] += 1

        if has_gold:
            probe = teacher_probe(family, prompt, answer)
            reason = probe.get("reason", "unknown")
            f["teacher_reasons"][reason] = f["teacher_reasons"].get(reason, 0) + 1

            if probe.get("covered") is True:
                f["teacher_verified"] += 1
            else:
                f["teacher_uncovered"] += 1

        if len(f["examples"]) < 3:
            f["examples"].append(
                {
                    "id": rid,
                    "subtype": subtype,
                    "answer": answer,
                    "prompt_head": norm_text(prompt)[:350],
                }
            )

    total = len(rows)
    for family, f in fams.items():
        rows_n = f["rows"]
        f["row_fraction"] = rows_n / total if total else 0.0
        if has_gold:
            f["answer_format_valid_rate"] = f["answer_format_valid"] / rows_n if rows_n else 0.0
            f["teacher_verified_rate"] = f["teacher_verified"] / rows_n if rows_n else 0.0

    return {
        "total_rows": total,
        "families": fams,
    }


def build_training_mix_plan(train_summary: Dict[str, Any]) -> Dict[str, Any]:
    families = train_summary["families"]

    expected_missing = [f for f in PUBLIC6 if f not in families]
    unknown_rows = families.get("unknown", {}).get("rows", 0)

    # Conservative thresholds. These are intentionally not too easy.
    # Phase 9 should not be authorized if the major hard families have zero verified teacher coverage.
    thresholds = {
        "roman_numeral": 0.95,
        "unit_conversion": 0.75,
        "gravity_numeric_formula": 0.75,
        "word_cipher": 0.20,
        "bit_manipulation": 0.05,
        "equation_transform": 0.05,
    }

    coverage = {
        f: float(families.get(f, {}).get("teacher_verified_rate", 0.0))
        for f in PUBLIC6
    }

    blocked = []
    if expected_missing:
        blocked.append(f"missing_public6_families:{expected_missing}")
    if unknown_rows:
        blocked.append(f"unknown_family_rows:{unknown_rows}")

    for fam, min_cov in thresholds.items():
        if coverage.get(fam, 0.0) < min_cov:
            blocked.append(
                f"teacher_coverage_below_threshold:{fam}:{coverage.get(fam, 0.0):.4f}<required:{min_cov:.4f}"
            )

    total_public6_rows = sum(families.get(f, {}).get("rows", 0) for f in PUBLIC6)

    # Balanced by rows initially, with warning not to balance by raw token count blindly.
    suggested_family_weights = {}
    for fam in PUBLIC6:
        rows = families.get(fam, {}).get("rows", 0)
        suggested_family_weights[fam] = round(rows / total_public6_rows, 6) if total_public6_rows else 0.0

    variants = [
        {
            "name": "A1_public6_direct_qv_r32",
            "style": "direct",
            "families": PUBLIC6,
            "rank": 32,
            "target_modules": ["q_proj", "v_proj"],
            "phase9_priority": 1,
        },
        {
            "name": "A3_public6_mixed_curriculum_qv_r32",
            "style": "mixed_curriculum",
            "families": PUBLIC6,
            "rank": 32,
            "target_modules": ["q_proj", "v_proj"],
            "phase9_priority": 2,
        },
        {
            "name": "A5_equation_transform_heavy_qv_r32",
            "style": "equation_transform_heavy",
            "families": ["equation_transform", "bit_manipulation", "word_cipher"],
            "rank": 32,
            "target_modules": ["q_proj", "v_proj"],
            "phase9_priority": 3,
        },
    ]

    return {
        "schema_version": 1,
        "created_by": "PHASE85_PUBLIC6_TRAINING_MIX_PLANNER",
        "public6": PUBLIC6,
        "teacher_coverage_thresholds": thresholds,
        "teacher_coverage_observed": coverage,
        "suggested_family_weights_by_row_count": suggested_family_weights,
        "token_budget_rule": "family-balanced token budget; no family should exceed 25-30 percent tokens unless explicitly justified",
        "recommended_phase9_variants": variants,
        "phase9_authorized": len(blocked) == 0,
        "package_authorized": False,
        "submission_authorized": False,
        "leaderboard_claim": False,
        "blocked_reasons": blocked,
    }


def run_phase85(train_csv: Path, test_csv: Path, out_dir: Path) -> Dict[str, Any]:
    train_rows = read_csv_rows(train_csv)
    test_rows = read_csv_rows(test_csv)

    train_summary = summarize_rows(train_rows, has_gold=True)
    test_summary = summarize_rows(test_rows, has_gold=False)

    family_audit = {
        "schema_version": 1,
        "created_by": "PHASE85_PUBLIC6_FAMILY_AUDIT",
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
        "public6_expected": PUBLIC6,
        "train": train_summary,
        "test": test_summary,
        "package_authorized": False,
        "submission_authorized": False,
        "leaderboard_claim": False,
    }

    train_fams = set(train_summary["families"])
    missing = [f for f in PUBLIC6 if f not in train_fams]
    unknown = train_summary["families"].get("unknown", {}).get("rows", 0)

    family_audit["status"] = "PASS" if not missing and unknown == 0 else "FAIL"
    family_audit["blocked_reasons"] = []
    if missing:
        family_audit["blocked_reasons"].append(f"missing_public6_families:{missing}")
    if unknown:
        family_audit["blocked_reasons"].append(f"unknown_family_rows:{unknown}")

    coverage_report = {
        "schema_version": 1,
        "created_by": "PHASE85_PUBLIC6_SOLVER_COVERAGE_AUDIT",
        "families": {},
        "package_authorized": False,
        "submission_authorized": False,
        "leaderboard_claim": False,
    }

    for fam in PUBLIC6:
        f = train_summary["families"].get(fam, {})
        rows = int(f.get("rows", 0))
        verified = int(f.get("teacher_verified", 0))
        coverage_report["families"][fam] = {
            "train_rows": rows,
            "answer_format_valid": int(f.get("answer_format_valid", 0)),
            "answer_format_valid_rate": float(f.get("answer_format_valid_rate", 0.0)),
            "teacher_verified": verified,
            "teacher_uncovered": int(f.get("teacher_uncovered", 0)),
            "teacher_verified_rate": float(f.get("teacher_verified_rate", 0.0)),
            "subtypes": f.get("subtypes", {}),
            "teacher_reasons": f.get("teacher_reasons", {}),
            "examples": f.get("examples", []),
        }

    mix_plan = build_training_mix_plan(train_summary)

    coverage_report["phase9_authorized"] = mix_plan["phase9_authorized"]
    coverage_report["status"] = "PASS" if mix_plan["phase9_authorized"] else "WARN"
    coverage_report["blocked_reasons"] = mix_plan["blocked_reasons"]

    manifest = {
        "schema_version": 1,
        "created_by": "PHASE85_PUBLIC6_VERIFIED_SFT_MANIFEST",
        "status": "PASS" if mix_plan["phase9_authorized"] else "WARN",
        "phase9_authorized": mix_plan["phase9_authorized"],
        "notes": [
            "This manifest is an audit/coverage manifest, not a generated SFT dataset.",
            "Do not train Phase 9 if phase9_authorized is false.",
            "Implement missing deterministic teachers before increasing LoRA training.",
        ],
        "package_authorized": False,
        "submission_authorized": False,
        "leaderboard_claim": False,
        "blocked_reasons": mix_plan["blocked_reasons"],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "public_family_audit_report.json", family_audit)
    write_json(out_dir / "public6_solver_coverage_report.json", coverage_report)
    write_json(out_dir / "public6_teacher_gap_report.json", {
        "schema_version": 1,
        "created_by": "PHASE85_PUBLIC6_TEACHER_GAP_REPORT",
        "gaps": mix_plan["blocked_reasons"],
        "family_coverage": mix_plan["teacher_coverage_observed"],
        "package_authorized": False,
        "submission_authorized": False,
        "leaderboard_claim": False,
    })
    write_json(out_dir / "public6_training_mix_plan.json", mix_plan)
    write_json(out_dir / "public6_verified_sft_manifest.json", manifest)

    final = {
        "schema_version": 1,
        "created_by": "PHASE85_PUBLIC6_AUDIT_AND_MIX_GATE",
        "status": "PASS" if family_audit["status"] == "PASS" else "FAIL",
        "family_audit_status": family_audit["status"],
        "coverage_status": coverage_report["status"],
        "phase9_authorized": mix_plan["phase9_authorized"],
        "public6": PUBLIC6,
        "train_rows": train_summary["total_rows"],
        "test_rows": test_summary["total_rows"],
        "outputs": {
            "public_family_audit_report": str(out_dir / "public_family_audit_report.json"),
            "public6_solver_coverage_report": str(out_dir / "public6_solver_coverage_report.json"),
            "public6_teacher_gap_report": str(out_dir / "public6_teacher_gap_report.json"),
            "public6_training_mix_plan": str(out_dir / "public6_training_mix_plan.json"),
            "public6_verified_sft_manifest": str(out_dir / "public6_verified_sft_manifest.json"),
        },
        "package_authorized": False,
        "submission_authorized": False,
        "leaderboard_claim": False,
        "blocked_reasons": family_audit["blocked_reasons"] + mix_plan["blocked_reasons"],
    }
    write_json(out_dir / "phase85_public6_final_gate_report.json", final)
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="train.csv")
    parser.add_argument("--test-csv", default="test.csv")
    parser.add_argument("--out-dir", default="artifacts/sprint11")
    args = parser.parse_args()

    final = run_phase85(Path(args.train_csv), Path(args.test_csv), Path(args.out_dir))
    print(json.dumps({
        "status": final["status"],
        "family_audit_status": final["family_audit_status"],
        "coverage_status": final["coverage_status"],
        "phase9_authorized": final["phase9_authorized"],
        "train_rows": final["train_rows"],
        "test_rows": final["test_rows"],
        "blocked_reasons": final["blocked_reasons"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
