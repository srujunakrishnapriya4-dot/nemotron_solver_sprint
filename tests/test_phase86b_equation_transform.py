from kaggle_anti086.phase85.public6_teachers import equation_transform_teacher


def test_equation_symbolic_direct_ordered_positions():
    p = (
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
        "Below are a few examples: "
        "abcde = bd "
        "vwxyz = wy "
        "12345 = 24 "
        "Now, determine the result for: pqrst"
    )
    r = equation_transform_teacher(p, "qs")
    assert r["covered"] is True
    assert r["prediction"] == "qs"


def test_equation_symbolic_reversed_positions():
    p = (
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
        "Below are a few examples: "
        "abcde = db "
        "vwxyz = yw "
        "12345 = 42 "
        "Now, determine the result for: pqrst"
    )
    r = equation_transform_teacher(p, "sq")
    assert r["covered"] is True
    assert r["prediction"] == "sq"


def test_equation_symbolic_repeated_positions():
    p = (
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
        "Below are a few examples: "
        "abcde = bb "
        "vwxyz = ww "
        "12345 = 22 "
        "Now, determine the result for: pqrst"
    )
    r = equation_transform_teacher(p, "qq")
    assert r["covered"] is True
    assert r["prediction"] == "qq"


def test_equation_numeric_failure_does_not_block_symbolic_fallback():
    p = (
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
        "Below are a few examples: "
        "34/44 = 34 "
        "41/32 = 41 "
        "87/64 = 87 "
        "Now, determine the result for: 69/52"
    )
    r = equation_transform_teacher(p, "69")
    assert r["covered"] is True
    assert r["prediction"] == "69"


def test_equation_symbolic_constant_rhs():
    p = (
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
        "Below are a few examples: "
        "abc = ?? "
        "def = ?? "
        "ghi = ?? "
        "Now, determine the result for: xyz"
    )
    r = equation_transform_teacher(p, "??")
    assert r["covered"] is True
    assert r["prediction"] == "??"
