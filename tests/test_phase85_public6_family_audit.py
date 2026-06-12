from pathlib import Path

from kaggle_anti086.phase85.public6_family_audit import (
    classify_family,
    roman_from_int,
    roman_teacher,
    unit_conversion_teacher,
    gravity_teacher,
    word_cipher_teacher,
    run_phase85,
)


def test_family_classifier_public6_patterns():
    assert classify_family("A secret bit manipulation rule transforms 8-bit binary numbers")[0] == "bit_manipulation"
    assert classify_family("The gravitational constant has been secretly changed. d = 0.5*g*t^2")[0] == "gravity_numeric_formula"
    assert classify_family("A secret unit conversion is applied. 10 m becomes 5. Now convert: 20 m")[0] == "unit_conversion"
    assert classify_family("Secret encryption rules are used on text. Now decrypt: abc")[0] == "word_cipher"
    assert classify_family("Numbers are secretly converted into a different numeral system")[0] == "roman_numeral"
    assert classify_family("A secret set of transformation rules is applied to equations")[0] == "equation_transform"


def test_roman_teacher():
    assert roman_from_int(94) == "XCIV"
    p = "Numbers are secretly converted into a different numeral system. Now convert: 38"
    r = roman_teacher(p, "XXXVIII")
    assert r["covered"] is True


def test_unit_conversion_teacher():
    p = """
    A secret unit conversion is applied.
    10 m becomes 5.00
    20 m becomes 10.00
    4 m becomes 2.00
    Now convert: 30 m
    """
    r = unit_conversion_teacher(p, "15.00")
    assert r["covered"] is True


def test_gravity_teacher():
    p = """
    The gravitational constant has been secretly changed.
    For t = 1.0s, distance = 5.00 m
    For t = 2.0s, distance = 20.00 m
    Now determine distance for t = 3.0s given d = 0.5*g*t^2.
    """
    r = gravity_teacher(p, "45.00")
    assert r["covered"] is True


def test_word_cipher_teacher_word_map():
    p = """
    abc def -> cat dog
    ghi jkl -> red sun
    Now decrypt: abc jkl
    """
    r = word_cipher_teacher(p, "cat sun")
    assert r["covered"] is True


def test_phase85_end_to_end_writes_reports(tmp_path):
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    out = tmp_path / "out"

    train.write_text(
        "id,prompt,answer\n"
        "r1,\"Numbers are secretly converted into a different numeral system. Now convert: 11\",XI\n"
        "r2,\"A secret unit conversion is applied. 10 m becomes 5.00. 20 m becomes 10.00. Now convert: 30 m\",15.00\n"
        "r3,\"The gravitational constant has been secretly changed. For t = 1.0s, distance = 5.00 m. For t = 2.0s, distance = 20.00 m. Now determine distance for t = 3.0s given d = 0.5*g*t^2.\",45.00\n"
        "r4,\"Secret encryption rules are used on text. abc def -> cat dog. Now decrypt: abc def\",cat dog\n"
        "r5,\"A secret bit manipulation rule transforms 8-bit binary numbers. Now transform: 10101010\",01010101\n"
        "r6,\"A secret set of transformation rules is applied to equations. 1+2=3. Now: 4+5\",9\n",
        encoding="utf-8",
    )
    test.write_text(
        "id,prompt\n"
        "t1,\"A secret bit manipulation rule transforms 8-bit binary numbers. Now transform: 00000000\"\n",
        encoding="utf-8",
    )

    final = run_phase85(train, test, out)
    assert (out / "public_family_audit_report.json").exists()
    assert (out / "public6_solver_coverage_report.json").exists()
    assert (out / "public6_training_mix_plan.json").exists()
    assert final["package_authorized"] is False
    assert final["submission_authorized"] is False
    assert final["leaderboard_claim"] is False
