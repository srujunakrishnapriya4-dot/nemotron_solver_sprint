from kaggle_anti086.phase85.public6_teachers import (
    word_cipher_teacher,
    bit_manipulation_teacher,
    equation_transform_teacher,
)


def test_word_cipher_inline_char_substitution():
    p = (
        "In Alice's Wonderland, secret encryption rules are used on text. "
        "Here are some examples: ucoov -> queen pwgtfyoqg -> discovers vorq -> near "
        "Now decrypt: ucoov vorq"
    )
    r = word_cipher_teacher(p, "queen near")
    assert r["covered"] is True
    assert r["prediction"] == "queen near"


def test_word_cipher_inline_phrase_word_map():
    p = (
        "In Alice's Wonderland, secret encryption rules are used on text. "
        "Here are some examples: abc def -> cat dog ghi jkl -> red sun "
        "Now decrypt: abc jkl"
    )
    r = word_cipher_teacher(p, "cat sun")
    assert r["covered"] is True
    assert r["prediction"] == "cat sun"


def test_bit_manipulation_xor_constant():
    p = (
        "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. "
        "Here are some examples of input -> output: "
        "00000000 -> 11111111 "
        "10101010 -> 01010101 "
        "11110000 -> 00001111 "
        "Now transform: 00110011"
    )
    r = bit_manipulation_teacher(p, "11001100")
    assert r["covered"] is True
    assert r["prediction"] == "11001100"


def test_bit_manipulation_rotate_left():
    p = (
        "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. "
        "Here are some examples of input -> output: "
        "10000001 -> 00000011 "
        "00010000 -> 00100000 "
        "01010101 -> 10101010 "
        "Now transform: 11110000"
    )
    r = bit_manipulation_teacher(p, "11100001")
    assert r["covered"] is True
    assert r["prediction"] == "11100001"


def test_equation_transform_numeric_add():
    p = (
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
        "Below are a few examples: "
        "12+23 = 35 "
        "40+17 = 57 "
        "8+9 = 17 "
        "Now, determine the result for: 31+44"
    )
    r = equation_transform_teacher(p, "75")
    assert r["covered"] is True
    assert r["prediction"] == "75"


def test_equation_transform_numeric_absdiff_with_operator():
    p = (
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
        "Below are a few examples: "
        "34/44 = 10/ "
        "41/32 = 9/ "
        "87/64 = 23/ "
        "Now, determine the result for: 69/52"
    )
    r = equation_transform_teacher(p, "17/")
    assert r["covered"] is True
    assert r["prediction"] == "17/"


def test_equation_transform_symbolic_position_substitution():
    p = (
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
        "Below are a few examples: "
        "abcde = XY "
        "axcye = XZ "
        "ppcqe = WQ "
        "Now, determine the result for: mmcne"
    )
    # positions 0 and 3, with char map m->?, n->? cannot be learned, so use known chars
    p2 = (
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
        "Below are a few examples: "
        "abcde = XY "
        "axcye = XZ "
        "ppcqe = WQ "
        "Now, determine the result for: apcye"
    )
    r = equation_transform_teacher(p2, "XZ")
    assert r["covered"] is True
    assert r["prediction"] == "XZ"
