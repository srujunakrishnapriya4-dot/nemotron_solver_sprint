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
from kaggle_anti086.solvers.bit_transform_solver import _format_bits, _rol
from kaggle_anti086.solvers.numeric_parsing import format_decimal
from kaggle_anti086.solvers.roman_solver import int_to_roman


def build_rows() -> list[dict]:
    rows: list[dict] = []
    rows += _bit_rows()
    rows += _symbol_rows()
    rows += _char_rows()
    rows += _word_rows()
    rows += _unit_rows()
    rows += _numeric_gravity_rows()
    rows += _roman_rows()
    assert len(rows) == 256
    return rows


def write_eval(path: str | Path) -> list[dict]:
    rows = build_rows()
    report = validate_rows(rows, context="day4_adversarial")
    if report["failure_count"]:
        raise SystemExit(json.dumps(report, indent=2, sort_keys=True))
    safe_write_text(path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), field_name="day4_adversarial_solver_eval")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Day 4 adversarial deterministic solver eval.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    rows = write_eval(args.out)
    print(json.dumps({"status": "WROTE", "row_count": len(rows), "out": args.out}, sort_keys=True))
    return 0


def _base(row_id: str, family: str, subfamily: str, prompt: str, answer: str, behavior: str, difficulty: int = 2) -> dict:
    return {
        "id": row_id,
        "family": family,
        "subfamily": subfamily,
        "rule_id": f"{family}_{subfamily}",
        "prompt": prompt,
        "answer": answer,
        "source": "day4_adversarial_solver_eval",
        "solver_name": "solver_ensemble",
        "verification_status": "verified",
        "difficulty": difficulty,
        "split": "dev",
        "leakage_group": f"{family}_{subfamily}",
        "metadata": {"expected_solver_behavior": behavior},
    }


def _bit_rows() -> list[dict]:
    rows = []
    for i in range(36):
        q = _format_bits((i * 37 + 19) % 256, 8)
        if i % 3 == 0:
            mask = 0b10100101
            ex = [(x, _format_bits(int(x, 2) ^ mask, 8)) for x in ["00000000", "11111111", "01010101"]]
            ans = _format_bits(int(q, 2) ^ mask, 8)
            sub = "xor_mask"
        elif i % 3 == 1:
            ex = [(x, x[::-1]) for x in ["00010010", "11100001", "01001100"]]
            ans = q[::-1]
            sub = "reverse"
        else:
            mask = 0b00110011
            ex = [(x, _format_bits(int(_rol(x, 2), 2) ^ mask, 8)) for x in ["00010010", "11100001", "01001100"]]
            ans = _format_bits(int(_rol(q, 2), 2) ^ mask, 8)
            sub = "rotate_then_xor"
        prompt = "; ".join(f"{a} -> {b}" for a, b in ex) + f". Input: {q}"
        rows.append(_base(f"day4_bit_answer_{i}", "bit_manipulation", sub, prompt, ans, "answer"))
    for i in range(12):
        prompt = "0101 -> 1010; 11110000 -> 00001111. Input: 0101" if i % 2 == 0 else "0101 -> 1010. Input: 1111"
        rows.append(_base(f"day4_bit_abstain_{i}", "bit_manipulation", "expected_abstain", prompt, "ABSTAIN", "abstain", 3))
    return rows


def _symbol_rows() -> list[dict]:
    rows = []
    for i in range(36):
        prompt = "!@ -> ab; @# -> bc; input: !# output ?"
        rows.append(_base(f"day4_symbol_answer_{i}", "symbol_mapping", "char_substitution", prompt, "ac", "answer"))
    for i in range(12):
        prompt = "!@ -> ab; !# -> xy; input: !# output ?" if i % 2 == 0 else "!@ -> ab; #$ -> cd; input: !Z output ?"
        rows.append(_base(f"day4_symbol_abstain_{i}", "symbol_mapping", "expected_abstain", prompt, "ABSTAIN", "abstain", 3))
    return rows


def _char_rows() -> list[dict]:
    rows = []
    for i in range(30):
        if i % 3 == 0:
            prompt = "abc -> def; xyz -> abc; query: mno"
            answer = "pqr"
            sub = "caesar_shift"
        elif i % 3 == 1:
            prompt = "abc -> cba; book -> koob; query: star"
            answer = "rats"
            sub = "reverse_string"
        else:
            prompt = "trb -> cat; hffk -> book; query: trbhffk"
            answer = "catbook"
            sub = "monoalphabetic_substitution"
        rows.append(_base(f"day4_char_answer_{i}", "char_cipher", sub, prompt, answer, "answer"))
    for i in range(10):
        prompt = "abc -> def; abc -> xyz; query: abc" if i % 2 == 0 else "abc -> def; query: zzz"
        rows.append(_base(f"day4_char_abstain_{i}", "char_cipher", "expected_abstain", prompt, "ABSTAIN", "abstain", 3))
    return rows


def _word_rows() -> list[dict]:
    rows = []
    for i in range(24):
        prompt = '"aaa bbb ccc" -> "cat dog moon"; encrypted: aaa ccc plaintext: ?'
        rows.append(_base(f"day4_word_answer_{i}", "word_cipher", "word_substitution", prompt, "cat moon", "answer"))
    for i in range(8):
        prompt = '"aaa bbb" -> "cat dog"; encrypted: aaa zzz plaintext: ?'
        rows.append(_base(f"day4_word_abstain_{i}", "word_cipher", "expected_abstain", prompt, "ABSTAIN", "abstain", 3))
    return rows


def _unit_rows() -> list[dict]:
    rows = []
    for i in range(24):
        q = Decimal(i + 10)
        prompt = "2 -> 3.00; 4 -> 6.00; 6 -> 9.00. Convert " + str(q)
        rows.append(_base(f"day4_unit_answer_{i}", "unit_conversion", "multiplicative_scale_round_2", prompt, format_decimal(q * Decimal("1.5"), 2), "answer"))
    for i in range(8):
        prompt = "1 -> 1; 2 -> 4. For 3, output ?"
        rows.append(_base(f"day4_unit_abstain_{i}", "unit_conversion", "expected_abstain", prompt, "ABSTAIN", "abstain", 3))
    return rows


def _numeric_gravity_rows() -> list[dict]:
    rows = []
    for i in range(12):
        q = i + 4
        rows.append(_base(f"day4_numeric_answer_{i}", "numeric_formula", "quadratic_scale", f"1 -> 2; 2 -> 8; 3 -> 18. Query {q}?", str(2 * q * q), "answer"))
    for i in range(12):
        t = i + 4
        rows.append(_base(f"day4_gravity_answer_{i}", "gravity_numeric", "gravity_distance", f"time 1 -> 4.90; time 2 -> 19.60; time 3 -> 44.10. Query time {t}?", format_decimal(Decimal("4.90") * Decimal(t * t), 2), "answer"))
    for i in range(8):
        rows.append(_base(f"day4_numeric_abstain_{i}", "gravity_numeric", "expected_abstain", "1 -> 2; 2 -> 4; 3 -> 6. Query 4?", "ABSTAIN", "abstain", 3))
    return rows


def _roman_rows() -> list[dict]:
    rows = []
    for i, n in enumerate([4, 9, 38, 44, 58, 94, 145, 399, 944, 1994, 2024, 3999, 27, 73, 166, 388, 512, 777]):
        rows.append(_base(f"day4_roman_answer_{i}", "roman_numeral", "standard_roman", f"11 -> XI; 15 -> XV. solve for {n}", int_to_roman(n), "answer"))
    for i in range(6):
        rows.append(_base(f"day4_roman_abstain_{i}", "roman_numeral", "expected_abstain", "11 -> XII; 15 -> XV. solve for 38", "ABSTAIN", "abstain", 3))
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
