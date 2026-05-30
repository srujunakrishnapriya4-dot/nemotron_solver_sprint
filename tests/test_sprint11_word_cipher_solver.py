from kaggle_anti086.solvers.word_cipher_solver import WordCipherSolver


def _row(prompt: str) -> dict:
    return {
        "id": "cipher_1",
        "family": "word_cipher",
        "subfamily": "word_substitution",
        "rule_id": "cipher_rule",
        "prompt": prompt,
        "answer": "mouse chases mirror",
        "source": "test",
        "solver_name": "word_cipher_solver",
        "verification_status": "verified",
        "difficulty": 2,
        "split": "dev",
        "leakage_group": "cipher",
    }


def test_word_cipher_full_dictionary_phrase_solves() -> None:
    prompt = (
        '"wkgqa lsrqaq wneeke" -> "mouse chases mirror"; '
        '"wkgqa nwrjnvaq nv brmrla" -> "mouse imagines in palace"; '
        'encrypted: wkgqa lsrqaq wneeke plaintext: ?'
    )
    result = WordCipherSolver().solve(_row(prompt))
    assert not result.abstained
    assert result.candidates[0].answer == "mouse chases mirror"
    assert result.candidates[0].verified is True


def test_word_cipher_conflicting_mapping_abstains() -> None:
    prompt = '"aaa bbb" -> "cat dog"; "aaa ccc" -> "mouse bird"; encrypted: aaa bbb plaintext: ?'
    assert WordCipherSolver().solve(_row(prompt)).abstained


def test_word_cipher_unknown_heavy_query_abstains() -> None:
    prompt = '"aaa bbb" -> "cat dog"; encrypted: aaa zzz plaintext: ?'
    assert WordCipherSolver().solve(_row(prompt)).abstained


def test_word_cipher_mismatched_examples_do_not_corrupt_mapping() -> None:
    prompt = '"aaa bbb ccc" -> "cat dog"; "aaa bbb" -> "cat dog"; encrypted: aaa bbb plaintext: ?'
    result = WordCipherSolver().solve(_row(prompt))
    assert not result.abstained
    assert result.candidates[0].answer == "cat dog"
