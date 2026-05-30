from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.parsing import canonicalize_prompt  # noqa: E402
from nemotron_engine.solvers.known_cipher_digit import solve  # noqa: E402


def test_exact_cipher_token_mapping_solved() -> None:
    result = solve(canonicalize_prompt("p", "@ -> 1\n# -> 2\n@ -> ?"))

    assert result.best_attempt is not None
    assert result.best_attempt.proof.target_execution.output_value == "1"


def test_non_bijective_unseen_and_whitespace_scope_rejected() -> None:
    non_bijective = solve(canonicalize_prompt("n", "@ -> 1\n# -> 1\n@ -> ?"))
    unseen = solve(canonicalize_prompt("u", "@ -> 1\n# -> ?"))
    multi = solve(canonicalize_prompt("m", "@ # -> 12\n% & -> ?"))

    assert non_bijective.best_attempt is None
    assert non_bijective.rejected_attempts[0].reason == "non_bijective_mapping"
    assert unseen.best_attempt is None
    assert unseen.rejected_attempts[0].reason == "unseen_target_token"
    assert multi.best_attempt is None
    assert multi.rejected_attempts[0].reason == "unsupported_multi_token_delimiter_preserving_output"


def test_internal_whitespace_is_not_collapsed_or_normalized() -> None:
    target_multi = solve(canonicalize_prompt("t", "@ -> 1\n# -> 2\n@ # -> ?"))

    assert target_multi.best_attempt is None
    assert target_multi.rejected_attempts[0].reason == "unsupported_multi_token_delimiter_preserving_output"
