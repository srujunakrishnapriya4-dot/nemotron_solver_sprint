from kaggle_anti086.solvers.format_only_solver import FormatOnlySolver
from kaggle_anti086.solvers.solver_ensemble import SolverEnsemble


def _row(raw: str, answer: str, answer_type: str = "generic") -> dict:
    return {
        "id": "fmt_1",
        "family": "format_only",
        "subfamily": "boxed_answer",
        "prompt": f"Format-only cleanup task. Raw output: {raw}\nExtract final answer only.",
        "answer": answer,
        "metadata": {"answer_type": answer_type},
    }


def test_format_only_boxed_numeric() -> None:
    result = FormatOnlySolver().solve(_row(r"\boxed{154.620}", "154.62", "numeric"))
    assert not result.abstained
    assert result.candidates[0].answer == "154.62"


def test_format_only_think_roman() -> None:
    result = FormatOnlySolver().solve(_row("</think>\nXXXVIII", "XXXVIII", "roman"))
    assert not result.abstained
    assert result.candidates[0].answer == "XXXVIII"


def test_format_only_preserves_symbol_answer() -> None:
    result = FormatOnlySolver().solve(_row("Answer: @&.", "@&", "symbol"))
    assert not result.abstained
    assert result.candidates[0].answer == "@&"


def test_format_only_binary_first_line() -> None:
    result = FormatOnlySolver().solve(_row("00101010\nExplanation: scratch.", "00101010", "binary"))
    assert not result.abstained
    assert result.candidates[0].answer == "00101010"


def test_format_only_rejects_mismatch() -> None:
    result = FormatOnlySolver().solve(_row("Answer: 123", "456", "numeric"))
    assert result.abstained
    assert result.reason == "normalized_answer_mismatch"


def test_ensemble_routes_format_only() -> None:
    result = SolverEnsemble().run_all(_row("38 -> XXXVIII", "XXXVIII", "roman"))
    assert not result.abstained
    assert result.candidates[0].source == "format_only_solver"
