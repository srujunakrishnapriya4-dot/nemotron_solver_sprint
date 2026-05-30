from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.data.schema import validate_rows
from kaggle_anti086.kaggle_path_safety import safe_write_text
from kaggle_anti086.solvers.roman_solver import int_to_roman
from kaggle_anti086.solvers.numeric_parsing import format_decimal


def build_rows() -> list[dict]:
    rows: list[dict] = []
    rows.extend(_roman_rows())
    rows.extend(_unit_rows())
    rows.extend(_numeric_rows())
    rows.extend(_gravity_rows())
    rows.extend(_word_cipher_rows())
    return rows


def write_smoke(path: str | Path) -> list[dict]:
    rows = build_rows()
    report = validate_rows(rows, context="day3_smoke")
    if report["failure_count"]:
        raise SystemExit(json.dumps(report, indent=2, sort_keys=True))
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    safe_write_text(path, payload, field_name="day3_solver_smoke_eval")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build deterministic Day 3 solver smoke eval rows.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    rows = write_smoke(args.out)
    print(json.dumps({"status": "WROTE", "row_count": len(rows), "out": args.out}, sort_keys=True))
    return 0


def _base(row_id: str, family: str, subfamily: str, rule_id: str, prompt: str, answer: str, difficulty: int = 1) -> dict:
    return {
        "id": row_id,
        "family": family,
        "subfamily": subfamily,
        "rule_id": rule_id,
        "prompt": prompt,
        "answer": answer,
        "source": "day3_solver_smoke",
        "solver_name": "solver_ensemble",
        "verification_status": "verified",
        "difficulty": difficulty,
        "split": "dev",
        "leakage_group": f"{family}_{rule_id}",
    }


def _roman_rows() -> list[dict]:
    queries = [4, 9, 38, 44, 58, 94, 145, 399, 944, 1994, 2024, 3999]
    forms = [
        "11 -> XI; 15 -> XV; 94 -> XCIV. Convert {n}",
        "11 -> XI; 15 -> XV; 94 -> XCIV. write {n} in the same system",
        "11 -> XI; 15 -> XV; 94 -> XCIV. solve for {n}",
        "11 -> XI; 15 -> XV; 94 -> XCIV. output for {n}",
    ]
    return [
        _base(f"day3_roman_{idx}", "roman_numeral", "standard_roman", "roman_standard", forms[idx % len(forms)].format(n=n), int_to_roman(n))
        for idx, n in enumerate(queries)
    ]


def _unit_rows() -> list[dict]:
    rows = []
    for idx, x in enumerate(range(10, 22)):
        if idx % 3 == 0:
            y = format_decimal(Decimal(x) * Decimal("1.50"), 2)
            prompt = f"2 -> 3.00; 4 -> 6.00; 6 -> 9.00. Convert {x}"
            subfamily = "multiplicative_scale_round_2"
        elif idx % 3 == 1:
            y = str(x + 5)
            prompt = f"10 -> 15; 20 -> 25; 30 -> 35. Input: {x}"
            subfamily = "linear_offset_round_0"
        else:
            y = format_decimal(Decimal(x) * Decimal("2.25") + Decimal("1.10"), 2)
            prompt = f"2 -> 5.60; 4 -> 10.10; 6 -> 14.60. For {x}, output ?"
            subfamily = "linear_scale_offset_round_2"
        rows.append(_base(f"day3_unit_{idx}", "unit_conversion", subfamily, f"unit_{idx % 3}", prompt, y, 2))
    return rows


def _numeric_rows() -> list[dict]:
    rows = []
    for idx, x in enumerate(range(4, 16)):
        if idx % 2 == 0:
            prompt = f"1 -> 3.0; 2 -> 6.0; 3 -> 9.0. Query {x}?"
            answer = format_decimal(Decimal(x) * Decimal(3), 1)
            subfamily = "linear_scale"
        else:
            prompt = f"1 -> 2; 2 -> 8; 3 -> 18. Query {x}?"
            answer = str(2 * x * x)
            subfamily = "quadratic_scale"
        rows.append(_base(f"day3_numeric_{idx}", "numeric_formula", subfamily, f"numeric_{idx % 2}", prompt, answer, 2))
    return rows


def _gravity_rows() -> list[dict]:
    rows = []
    for idx, t in enumerate(range(4, 16)):
        prompt = f"time 1 -> 4.90; time 2 -> 19.60; time 3 -> 44.10. Query time {t}?"
        answer = format_decimal(Decimal("4.90") * Decimal(t * t), 2)
        rows.append(_base(f"day3_gravity_{idx}", "gravity_numeric", "gravity_distance", "gravity_distance", prompt, answer, 2))
    return rows


def _word_cipher_rows() -> list[dict]:
    dictionaries = [
        ("aaa bbb ccc", "cat dog moon", "aaa bbb", "cat dog"),
        ("red blu grn", "sun sea sky", "red grn", "sun sky"),
        ("wkgqa lsrqaq wneeke", "mouse chases mirror", "wkgqa wneeke", "mouse mirror"),
    ]
    rows = []
    for idx in range(12):
        cipher, plain, query, answer = dictionaries[idx % len(dictionaries)]
        if idx % 2 == 0:
            prompt = f'"{cipher}" -> "{plain}"; encrypted: {query} plaintext: ?'
        else:
            prompt = f"`{cipher}` = `{plain}`; encrypted: {query} plaintext: ?"
        rows.append(_base(f"day3_word_{idx}", "word_cipher", "word_substitution", f"word_{idx % len(dictionaries)}", prompt, answer, 2))
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
