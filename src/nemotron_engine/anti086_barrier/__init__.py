"""Anti-0.86 barrier tooling for private-like validation and hard curricula."""

from .distribution_gap_analyzer import analyze_distribution_gap
from .training_recipe_selector import select_training_recipe
from .win_mode_program import build_win_mode_program

__all__ = ["analyze_distribution_gap", "build_win_mode_program", "select_training_recipe"]
