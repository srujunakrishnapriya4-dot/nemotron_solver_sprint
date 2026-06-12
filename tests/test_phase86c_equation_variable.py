from kaggle_anti086.phase85.public6_teachers import equation_transform_teacher


def test_equation_var_delete_one_position():
    p = (
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
        "Below are a few examples: "
        "abcde = abde "
        "vwxyz = vwyz "
        "12345 = 1245 "
        "Now, determine the result for: pqrst"
    )
    r = equation_transform_teacher(p, "pqst")
    assert r["covered"] is True
    assert r["prediction"] == "pqst"


def test_equation_var_delete_two_positions():
    p = (
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
        "Below are a few examples: "
        "abcdef = abef "
        "vwxyzz = vwzz "
        "123456 = 1256 "
        "Now, determine the result for: pqrstu"
    )
    r = equation_transform_teacher(p, "pqtu")
    assert r["covered"] is True
    assert r["prediction"] == "pqtu"


def test_equation_var_prefix():
    p = (
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
        "Below are a few examples: "
        "abcde = abc "
        "vwxyz = vwx "
        "12345 = 123 "
        "Now, determine the result for: pqrst"
    )
    r = equation_transform_teacher(p, "pqr")
    assert r["covered"] is True
    assert r["prediction"] == "pqr"


def test_equation_var_suffix_reverse():
    p = (
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
        "Below are a few examples: "
        "abcde = ed "
        "vwxyz = zy "
        "12345 = 54 "
        "Now, determine the result for: pqrst"
    )
    r = equation_transform_teacher(p, "ts")
    assert r["covered"] is True
    assert r["prediction"] == "ts"
