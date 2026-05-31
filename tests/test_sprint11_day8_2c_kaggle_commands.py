from kaggle_anti086.training.day8_2c_kaggle_commands import main


def test_day8_2c_command_plan_no_placeholders_or_package(tmp_path):
    out = tmp_path / "plan.txt"
    assert main(["--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "<" not in text and ">" not in text
    executable = "\n".join(line for line in text.splitlines() if not line.startswith("#")).lower()
    assert "package" not in executable
    assert "submit" not in executable
    assert "day8_2c_smoke_preflight.py" in text
    assert "day8_2b_smoke_orchestrator.py" in text
    assert "day8_2c_smoke_readiness.py" in text
    assert "day8_3_full_v2a_train.py" in text
