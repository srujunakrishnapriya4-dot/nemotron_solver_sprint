import json

from kaggle_anti086.eval.build_day3_solver_smoke import build_rows, write_smoke


def test_day3_smoke_builder_writes_schema_compatible_rows(tmp_path) -> None:
    out = tmp_path / "day3_smoke.jsonl"
    rows = write_smoke(out)
    assert out.exists()
    assert len(rows) >= 60
    by_family = {}
    for row in rows:
        by_family[row["family"]] = by_family.get(row["family"], 0) + 1
    assert by_family["roman_numeral"] >= 12
    assert by_family["unit_conversion"] >= 12
    assert by_family["numeric_formula"] >= 12
    assert by_family["gravity_numeric"] >= 12
    assert by_family["word_cipher"] >= 12
    first = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert first["verification_status"] == "verified"


def test_day3_smoke_builder_is_deterministic() -> None:
    assert build_rows() == build_rows()
