from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.competition_sprint.solver_coverage_report import build_solver_coverage_report  # noqa: E402
from test_competition_prompt_adapter import BIT_PROMPT, CIPHER_PROMPT, ROMAN_PROMPT  # noqa: E402


def test_solver_coverage_report_handles_empty_csv(tmp_path: Path) -> None:
    path = tmp_path / "train.csv"
    path.write_text("id,prompt,answer\n", encoding="utf-8")
    report = build_solver_coverage_report(path, tmp_path)
    assert "verified_correct_by_family" in report
    assert (tmp_path / "solver_coverage_report.json").exists()


def test_solver_coverage_report_nonzero_on_real_like_fixture(tmp_path: Path) -> None:
    path = tmp_path / "train.csv"
    path.write_text(
        "id,prompt,answer\n"
        f"b,\"{BIT_PROMPT.replace(chr(34), chr(34)+chr(34))}\",11001011\n"
        f"r,\"{ROMAN_PROMPT.replace(chr(34), chr(34)+chr(34))}\",XXXVIII\n"
        f"c,\"{CIPHER_PROMPT.replace(chr(34), chr(34)+chr(34))}\",cat book\n",
        encoding="utf-8",
    )

    report = build_solver_coverage_report(path, tmp_path)

    assert report["parsed_count"] == 3
    assert sum(report["solver_verified_correct_by_family"].values()) >= 2
    assert report["top_answer_normalization_mismatch_reasons"] == {}


def test_solver_coverage_report_equation_fixture_and_taxonomy(tmp_path: Path) -> None:
    prompt = (
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:\n"
        "%|*\"| = %|\"|\n"
        "\\(*[^ = \\([^\n"
        "(%+[@ = (%[@\n"
        "|[*([ = |[([\n"
        "[^-[( = -^\n"
        "Now, determine the result for: \\(*[#"
    )
    path = tmp_path / "train.csv"
    path.write_text("id,prompt,answer\n" f"e,\"{prompt.replace(chr(34), chr(34)+chr(34))}\",\\([#\n", encoding="utf-8")

    report = build_solver_coverage_report(path, tmp_path)

    assert report["solver_verified_correct_by_family"]["equation_symbolic"] == 1
    assert (tmp_path / "equation_failure_taxonomy.json").exists()
    assert (tmp_path / "equation_failure_clusters.jsonl").exists()
