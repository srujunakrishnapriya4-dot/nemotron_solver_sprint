from __future__ import annotations

from pathlib import Path


def detect_critical_gaps(repo_root: str | Path = ".") -> list[str]:
    root = Path(repo_root)
    gaps: list[str] = []
    eval_path = root / "kaggle_anti086" / "kaggle_eval_anti086_vllm.py"
    if eval_path.exists():
        text = eval_path.read_text(encoding="utf-8", errors="ignore")
        if '"prompt_copy_rate": 0.0' in text or '"answer_format_pass_rate": 1.0' in text:
            gaps.append("eval_metrics_hardcoded")
        if "EVAL_SCRIPT_PLACEHOLDER" in text:
            gaps.append("eval_placeholder")
    train_path = root / "kaggle_anti086" / "kaggle_train_anti086_adapter.py"
    if train_path.exists() and "loss_weights" not in train_path.read_text(encoding="utf-8", errors="ignore"):
        gaps.append("training_not_using_loss_weights")
    if not (root / "src" / "nemotron_engine" / "training_contract").exists():
        gaps.append("missing_training_contract")
    if not (root / "src" / "nemotron_engine" / "competition_sprint" / "solver_coverage_report.py").exists():
        gaps.append("missing_solver_coverage_report")
    return sorted(set(gaps))


def readiness_from_gaps(gaps: list[str]) -> str:
    if any(gap in gaps for gap in ("eval_metrics_hardcoded", "eval_placeholder", "training_not_using_loss_weights")):
        return "NOT_READY"
    if gaps:
        return "MICRO_READY"
    return "MAIN_READY"
