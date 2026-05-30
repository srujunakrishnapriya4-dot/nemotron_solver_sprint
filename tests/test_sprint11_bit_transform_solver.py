from kaggle_anti086.solvers.bit_transform_solver import BitTransformSolver


def _row(prompt: str) -> dict:
    return {"family": "bit_manipulation", "prompt": prompt}


def test_bit_xor_mask_works() -> None:
    result = BitTransformSolver().solve(_row("0000 -> 1010; 1111 -> 0101; 0011 -> 1001. Input: 0101"))
    assert not result.abstained
    assert result.candidates[0].answer == "1111"


def test_bit_reverse_works() -> None:
    result = BitTransformSolver().solve(_row("0001 -> 1000; 0110 -> 0110; 1100 -> 0011. Input: 1010"))
    assert not result.abstained
    assert result.candidates[0].answer == "0101"


def test_bit_rotate_works() -> None:
    result = BitTransformSolver().solve(_row("1001 -> 0011; 0110 -> 1100; 0011 -> 0110. Input: 1010"))
    assert not result.abstained
    assert result.candidates[0].answer == "0101"


def test_bit_rotate_xor_works() -> None:
    result = BitTransformSolver().solve(_row("00010010 -> 01111011; 11100001 -> 10110100; 01001100 -> 00000010. Input: 10101010"))
    assert not result.abstained
    assert result.candidates[0].answer == "10011001"


def test_bit_abstains_on_mixed_width_missing_query_and_no_match() -> None:
    assert BitTransformSolver().solve(_row("0101 -> 1010; 11110000 -> 00001111. Input: 0101")).abstained
    assert BitTransformSolver().solve(_row("0101 -> 1010; 1111 -> 0000.")).abstained
    assert BitTransformSolver().solve(_row("0000 -> 0001; 1111 -> 0010; 0101 -> 1110. Input: 0011")).abstained


def test_bit_ambiguity_with_different_query_outputs_abstains() -> None:
    result = BitTransformSolver().solve(_row("0101 -> 1010; 1010 -> 0101. Input: 1000"))
    assert result.abstained
    assert result.reason == "ambiguous_transform_disagreement"
