from kaggle_anti086.solvers.symbol_mapping_solver import SymbolMappingSolver
from kaggle_anti086.solvers.word_cipher_solver import WordCipherSolver


def test_varied_symbol_substitution_preserves_punctuation() -> None:
    row = {"family": "symbol_mapping", "prompt": "!@# -> @&!; $%^ -> []{; @#$ -> &![; query: #^$ output ?"}
    result = SymbolMappingSolver().solve(row)
    assert not result.abstained
    assert result.candidates[0].answer == "!{["


def test_symbol_conflict_unknown_and_compression_abstain() -> None:
    solver = SymbolMappingSolver()
    assert solver.solve({"family": "symbol_mapping", "prompt": "!@ -> ab; !# -> xy; query: !# output ?"}).abstained
    assert solver.solve({"family": "symbol_mapping", "prompt": "!@ -> ab; #$ -> cd; query: !Z% output ?"}).abstained
    assert solver.solve({"family": "symbol_mapping", "prompt": "!!@@ -> !@; ##$$ -> #$. query: !!@@ output ?"}).abstained


def test_varied_word_dictionary_and_repeated_word_solve() -> None:
    row = {
        "family": "word_cipher",
        "prompt": "encrypted: wkgqa lsrqaq plaintext: mouse chases; encrypted: wneeke wkgqa plaintext: mirror mouse; encrypted: wkgqa wkgqa plaintext: ?",
    }
    result = WordCipherSolver().solve(row)
    assert not result.abstained
    assert result.candidates[0].answer == "mouse mouse"
    assert result.candidates[0].metadata["coverage"] == 1.0


def test_word_conflict_and_unknown_abstain_with_diagnostics() -> None:
    conflict = WordCipherSolver().solve({"family": "word_cipher", "prompt": '"aaa bbb" -> "cat dog"; "aaa ccc" -> "moon sun"; encrypted: aaa ccc plaintext: ?'})
    assert conflict.abstained
    assert conflict.metadata["conflicting_words"] == ["aaa"]
    unknown = WordCipherSolver().solve({"family": "word_cipher", "prompt": '"aaa bbb" -> "cat dog"; encrypted: aaa zzz plaintext: ?'})
    assert unknown.abstained
    assert unknown.metadata["coverage"] < 1.0
