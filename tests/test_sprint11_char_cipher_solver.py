from kaggle_anti086.solvers.char_cipher_solver import CharCipherSolver
from kaggle_anti086.solvers.solver_ensemble import SolverEnsemble


def _row(prompt: str, family: str = "char_cipher") -> dict:
    return {"family": family, "prompt": prompt}


def test_char_caesar_works() -> None:
    result = CharCipherSolver().solve(_row("abc -> def; xyz -> abc; query: mno"))
    assert not result.abstained
    assert result.candidates[0].answer == "pqr"


def test_char_reverse_works() -> None:
    result = CharCipherSolver().solve(_row("abc -> cba; book -> koob; query: star"))
    assert not result.abstained
    assert result.candidates[0].answer == "rats"


def test_char_monoalphabetic_mapping_works() -> None:
    result = CharCipherSolver().solve(_row("trb -> cat; hffk -> book; query: trbhffk"))
    assert not result.abstained
    assert result.candidates[0].answer == "catbook"


def test_char_conflict_rejects_and_unknown_abstains() -> None:
    assert CharCipherSolver().solve(_row("abc -> def; abc -> xyz; query: abc")).abstained
    assert CharCipherSolver().solve(_row("abc -> def; query: zzz")).abstained


def test_char_does_not_override_complete_word_cipher() -> None:
    row = _row('"aaa bbb" -> "cat dog"; encrypted: aaa bbb plaintext: ?', family="word_cipher")
    best = SolverEnsemble().best_candidate(row)
    assert best is not None
    assert best.source == "word_cipher_solver"
    assert best.answer == "cat dog"
