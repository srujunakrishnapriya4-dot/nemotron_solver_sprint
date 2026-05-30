from __future__ import annotations

from pathlib import Path


def write_day8_submission_strategy(
    output_path: str | Path = "artifacts/anti086/DAY8_SUBMISSION_STRATEGY.md",
) -> str:
    text = """# Day-2 to Day-8 Submission Strategy

## Day 2
- Finish win-mode artifacts, validate token masks, and run micro only.
- Do not run v1/v2/v3 or package a custom adapter.

## Day 3
- Train v1 only if micro gate passes.
- Evaluate parent vs child on public_like_60 and family_hard_60.

## Day 4
- Train v2 only if v1 improves private-like hard splits without public-like collapse.
- Evaluate rule_holdout_60 and family_hard_60.

## Day 5
- Train v3 only if v2 improves hard families for real, not only synthetic holdout.
- Contrastive rows stay capped and are rejected on answer-format regression.

## Day 6
- Train one final selected recipe from the best gated stage.
- Package only if adapter rank, size, structure, and eval gates pass.

## Day 7
- Submit at most one or two strongest gated candidates if the user chooses.
- Record public scores manually and compare against private-like evidence.

## Day 8
- Freeze the best candidate.
- No last-minute blind training, no public-only chasing, and no submission of unknown behavior.

## Fallbacks
- If micro fails, stop custom training and package the known parent.
- Fallback to parent whenever no child has private-like evidence stronger than the known adapter.
- If public score is worse than parent, reject the child unless private-like evidence is overwhelming and risk is accepted explicitly.
- If private-like and public disagree, choose the model with lower private-risk, not the model with a tiny public-only bump.
"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text
