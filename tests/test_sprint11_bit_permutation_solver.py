from kaggle_anti086.solvers.bit_transform_solver import BitTransformSolver


def _row(prompt: str) -> dict:
    return {"family": "bit_manipulation", "prompt": prompt}


def _perm(bits: str, order: tuple[int, ...]) -> str:
    return "".join(bits[i] for i in order)


def test_known_bit_position_permutation_solves() -> None:
    order = (2, 0, 1, 3)
    examples = ["0000", "1000", "0100", "0010", "0001"]
    prompt = "; ".join(f"{bits} -> {_perm(bits, order)}" for bits in examples) + ". Input: 1100"
    result = BitTransformSolver().solve(_row(prompt))
    assert not result.abstained
    assert result.candidates[0].answer == _perm("1100", order)
    assert "permutation" in result.candidates[0].metadata["operation"]


def test_permutation_then_xor_solves() -> None:
    order = (1, 0, 3, 2)
    mask = int("1010", 2)
    examples = ["0000", "1000", "0100", "0010", "0001"]
    prompt = "; ".join(f"{bits} -> {int(_perm(bits, order), 2) ^ mask:04b}" for bits in examples) + ". Input: 1100"
    result = BitTransformSolver().solve(_row(prompt))
    assert not result.abstained
    assert result.candidates[0].answer == f"{int(_perm('1100', order), 2) ^ mask:04b}"


def test_permutation_too_few_examples_abstains() -> None:
    result = BitTransformSolver().solve(_row("1000 -> 0010; 0100 -> 1000. Input: 1100"))
    assert result.abstained


def test_simple_reverse_preferred_over_generic_permutation() -> None:
    result = BitTransformSolver().solve(_row("1000 -> 0001; 0100 -> 0010; 0010 -> 0100; 0001 -> 1000. Input: 1100"))
    assert not result.abstained
    assert result.candidates[0].subfamily == "reverse"


def test_mixed_width_still_abstains() -> None:
    assert BitTransformSolver().solve(_row("0101 -> 1010; 11110000 -> 00001111. Input: 0101")).abstained
