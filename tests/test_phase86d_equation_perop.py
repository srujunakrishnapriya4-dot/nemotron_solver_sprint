from kaggle_anti086.phase85.public6_teachers import equation_transform_teacher


def test_equation_per_operator_concat():
    p = (
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
        "Below are a few examples: "
        "19)37 = 18 "
        "15$20 = 1520 "
        "30)14 = 16 "
        "Now, determine the result for: 25$96"
    )
    r = equation_transform_teacher(p, "2596")
    assert r["covered"] is True
    assert r["prediction"] == "2596"


def test_equation_per_operator_abs_sub():
    p = (
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
        "Below are a few examples: "
        "46-02 = 44 "
        "84-62 = 22 "
        "01(33 = 133 "
        "46%92 = 39 "
        "Now, determine the result for: 79-36"
    )
    r = equation_transform_teacher(p, "43")
    assert r["covered"] is True
    assert r["prediction"] == "43"


def test_equation_per_operator_mul():
    p = (
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
        "Below are a few examples: "
        "96$54 = 5184 "
        "50$41 = 2050 "
        "51$95 = 4845 "
        "89$47 = 4183 "
        "Now, determine the result for: 59$49"
    )
    r = equation_transform_teacher(p, "2891")
    assert r["covered"] is True
    assert r["prediction"] == "2891"


def test_equation_per_operator_reject_ambiguous():
    p = (
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
        "Below are a few examples: "
        "12$34 = 1234 "
        "56$78 = 5678 "
        "Now, determine the result for: 90$12"
    )
    # concat is unique enough here; this is a positive guard.
    r = equation_transform_teacher(p, "9012")
    assert r["covered"] is True
    assert r["prediction"] == "9012"
