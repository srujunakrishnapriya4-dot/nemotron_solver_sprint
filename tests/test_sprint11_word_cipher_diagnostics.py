from kaggle_anti086.solvers.word_cipher_solver import WordCipherSolver


def _row(prompt: str) -> dict:
    return {
        "id": "word_diag",
        "family": "word_cipher",
        "subfamily": "word_substitution",
        "rule_id": "word_diag",
        "prompt": prompt,
        "answer": "cat dog",
        "source": "test",
        "solver_name": "word_cipher_solver",
        "verification_status": "verified",
        "difficulty": 1,
        "split": "dev",
        "leakage_group": "word_diag",
    }


def test_word_cipher_full_mapping_succeeds_with_metadata() -> None:
    result = WordCipherSolver().solve(_row('"aaa bbb" -> "cat dog"; encrypted: aaa bbb plaintext: ?'))
    assert not result.abstained
    metadata = result.candidates[0].metadata
    assert metadata["mapping_size"] == 2
    assert metadata["coverage_ratio"] == 1.0
    assert metadata["unknown_count"] == 0


def test_word_cipher_conflict_abstains_with_conflict_metadata() -> None:
    result = WordCipherSolver().solve(_row('"aaa bbb" -> "cat dog"; "aaa ccc" -> "mouse bird"; encrypted: aaa bbb plaintext: ?'))
    assert result.abstained
    assert result.reason == "mapping_conflict"
    assert result.metadata["conflicts"]


def test_word_cipher_unknown_abstains_with_coverage_ratio() -> None:
    result = WordCipherSolver().solve(_row('"aaa bbb" -> "cat dog"; encrypted: aaa zzz plaintext: ?'))
    assert result.abstained
    assert result.reason == "unknown_query_words"
    assert result.metadata["unknown_count"] == 1
    assert result.metadata["coverage_ratio"] == 0.5


def test_word_cipher_mismatched_pair_ignored_but_counted() -> None:
    result = WordCipherSolver().solve(_row('"aaa bbb ccc" -> "cat dog"; `aaa bbb` = `cat dog`; encrypted: aaa bbb plaintext: ?'))
    assert not result.abstained
    assert result.candidates[0].metadata["ignored_pairs"] == 1


def test_word_cipher_supported_pair_formats() -> None:
    prompts = [
        '"aaa bbb" -> "cat dog"; encrypted: aaa bbb plaintext: ?',
        "`aaa bbb` = `cat dog`; encrypted: aaa bbb plaintext: ?",
        "cipher: aaa bbb plain: cat dog; encrypted: aaa bbb plaintext: ?",
        "encrypted: aaa bbb plaintext: cat dog; encrypted: aaa bbb plaintext: ?",
    ]
    for prompt in prompts:
        result = WordCipherSolver().solve(_row(prompt))
        assert not result.abstained, prompt
        assert result.candidates[0].answer == "cat dog"
