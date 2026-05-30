from kaggle_anti086.solvers.router import route_row


def test_router_binary_to_bit() -> None:
    result = route_row({"family": "unknown", "prompt": "0101 -> 1010; input: 1111"})
    assert result.candidate_solvers == ["bit_transform_solver"]


def test_router_roman_to_roman() -> None:
    assert route_row({"family": "unknown", "prompt": "11 -> XI; solve for 38"}).candidate_solvers == ["roman_solver"]


def test_router_numeric_unit_and_gravity() -> None:
    assert route_row({"family": "unit_conversion", "prompt": "10 converts to 15. Convert 20"}).candidate_solvers[0] == "unit_conversion_solver"
    assert route_row({"family": "gravity_numeric", "prompt": "falling distance time 2 -> 19.6"}).candidate_solvers[0] == "numeric_formula_solver"


def test_router_word_symbol_char_and_equation() -> None:
    assert route_row({"family": "word_cipher", "prompt": '"aaa" -> "cat"'}).candidate_solvers[0] == "word_cipher_solver"
    assert route_row({"family": "symbol_mapping", "prompt": "!@ -> ab; input: !#"}).candidate_solvers[0] == "symbol_mapping_solver"
    assert route_row({"family": "char_cipher", "prompt": "abc -> def; query: mno"}).candidate_solvers[0] == "char_cipher_solver"
    eq = route_row({"family": "equation_operator", "prompt": "a @ b = c"})
    assert eq.supported_by_solver is False
    assert eq.family == "equation_operator"
