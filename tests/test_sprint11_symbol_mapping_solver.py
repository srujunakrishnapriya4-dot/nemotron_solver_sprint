from kaggle_anti086.solvers.symbol_mapping_solver import SymbolMappingSolver


def _row(prompt: str) -> dict:
    return {"family": "symbol_mapping", "prompt": prompt}


def test_symbol_same_length_substitution_works() -> None:
    result = SymbolMappingSolver().solve(_row("!@ -> ab; @# -> bc; input: !# output ?"))
    assert not result.abstained
    assert result.candidates[0].answer == "ac"


def test_symbol_reversal_substitution_works() -> None:
    result = SymbolMappingSolver().solve(_row("!@ -> ba; @# -> cb; input: !# output ?"))
    assert not result.abstained
    assert result.candidates[0].answer == "ca"


def test_symbol_conflict_rejects() -> None:
    result = SymbolMappingSolver().solve(_row("!@ -> ab; !# -> xy; input: !# output ?"))
    assert result.abstained


def test_symbol_unknown_coverage_threshold_enforced() -> None:
    result = SymbolMappingSolver().solve(_row("!@ -> ab; #$ -> cd; input: !Z output ?"))
    assert result.abstained
    assert result.reason == "unknown_symbol_coverage_too_low"


def test_symbol_punctuation_preserved() -> None:
    result = SymbolMappingSolver().solve(_row("!@ -> @&; @# -> &[; input: !# output ?"))
    assert not result.abstained
    assert result.candidates[0].answer == "@["
