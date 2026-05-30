from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.competition_sprint.bit_composition_solver import solve_bit_composition_problem  # noqa: E402


def test_bit_composition_solver_solves_xor_shift() -> None:
    examples = (("00000000", "11111111"), ("11111111", "00000000"), ("10101010", "01010101"))
    result = solve_bit_composition_problem(examples, "00110011")
    assert result.verified
    assert len(result.prediction) == 8


def test_bit_composition_solver_solves_rotates_shifts_masks_and_add_sub() -> None:
    rol = solve_bit_composition_problem((("00000001", "00000100"), ("10000000", "00000010"), ("00110000", "11000000")), "00001000")
    assert rol.verified
    assert rol.prediction == "00100000"
    xor_mask = solve_bit_composition_problem((("00000000", "10100101"), ("11111111", "01011010"), ("00110011", "10010110")), "01010101")
    assert xor_mask.verified
    assert xor_mask.prediction == "11110000"
    and_mask = solve_bit_composition_problem((("11111111", "00001111"), ("10101010", "00001010"), ("01010101", "00000101")), "11001100")
    assert and_mask.verified
    assert and_mask.prediction == "00001100"
    add_const = solve_bit_composition_problem((("00000000", "00000101"), ("00000001", "00000110"), ("11111111", "00000100")), "00001010")
    assert add_const.verified
    assert add_const.prediction == "00001111"


def test_bit_composition_solver_solves_reverse_and_depth_two_composition() -> None:
    reverse = solve_bit_composition_problem((("00010000", "00001000"), ("10000000", "00000001"), ("01100000", "00000110")), "00000011")
    assert reverse.verified
    assert reverse.prediction == "11000000"
    composed = solve_bit_composition_problem((("00000000", "11110000"), ("11111111", "00001111"), ("10101010", "01011010")), "00110011")
    assert composed.verified
    assert composed.prediction == "11000011"


def test_bit_composition_solver_rejects_one_example_and_reports_pruning() -> None:
    result = solve_bit_composition_problem((("00000000", "11111111"),), "00110011")
    assert not result.verified
    assert result.reason == "underdetermined_bit_rule"
    richer = solve_bit_composition_problem((("00000000", "00000000"), ("11111111", "11111111"), ("10101010", "10101010")), "01010101")
    assert richer.verified
    assert richer.metadata["pruned_duplicate_count"] >= 0
    assert "depth_solved_distribution" in richer.metadata


def test_bit_composition_solver_bit_position_boolean_synthesis() -> None:
    def transform(bits: str) -> str:
        b = [int(ch) for ch in bits]
        out = [
            b[0] ^ b[1],
            b[2] & b[3],
            b[4] | b[5],
            1 - b[6],
            (b[0] & b[1]) | (b[0] & b[2]) | (b[1] & b[2]),
            (b[3] & b[4]) | ((1 - b[3]) & b[5]),
            b[0] ^ b[2] ^ b[4],
            0,
        ]
        return "".join(str(bit) for bit in out)

    xs = tuple(format(value, "08b") for value in range(256))
    examples = tuple((x, transform(x)) for x in xs)
    result = solve_bit_composition_problem(examples, "10011100")
    assert result.verified
    assert result.prediction == transform("10011100")
    assert result.rule_id == "bit_position_boolean_synthesis"


def test_bit_composition_solver_affine_gf2_synthesis() -> None:
    def transform(bits: str) -> str:
        b = [int(ch) for ch in bits]
        out = [
            b[0] ^ b[1] ^ 1,
            b[2] ^ b[3],
            b[4],
            b[5] ^ 1,
            b[6] ^ b[7],
            b[0] ^ b[7],
            b[1] ^ b[6],
            b[2] ^ b[5],
        ]
        return "".join(str(bit) for bit in out)

    xs = tuple(format(value, "08b") for value in range(256))
    examples = tuple((x, transform(x)) for x in xs)
    result = solve_bit_composition_problem(examples, "10101100")
    assert result.verified
    assert result.prediction == transform("10101100")


def test_bit_composition_solver_truth_table_popcount_and_exact_k() -> None:
    def transform(bits: str) -> str:
        ones = bits.count("1")
        return "".join(
            [
                "1" if ones >= 4 else "0",
                "1" if ones <= 2 else "0",
                "1" if ones == 3 else "0",
                str(ones % 2),
                "0",
                "0",
                "0",
                "0",
            ]
        )

    examples = tuple((format(value, "08b"), transform(format(value, "08b"))) for value in range(256))
    result = solve_bit_composition_problem(examples, "00111100", budget=1200)
    assert result.verified
    assert result.rule_id == "truth_table_bit_mining"
    assert result.prediction == transform("00111100")


def test_bit_composition_solver_truth_table_target_disagreement() -> None:
    values = (1, 2, 3, 5, 8, 13, 21, 34)
    examples = tuple((format(value, "08b"), "0" + format(value, "08b")[1:][::-1]) for value in values)
    result = solve_bit_composition_problem(examples, "10001111", budget=1200)
    assert not result.verified
    assert result.reason == "ambiguous_fits"
