from __future__ import annotations

from abc import ABC, abstractmethod

from kaggle_anti086.solvers.types import SolverResult


class BaseSolver(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def supported_families(self) -> tuple[str, ...]:
        raise NotImplementedError

    @abstractmethod
    def solve(self, row: dict) -> SolverResult:
        raise NotImplementedError

    def abstain(self, reason: str) -> SolverResult:
        family = self.supported_families[0] if self.supported_families else "unknown"
        return SolverResult(solver_name=self.name, family=family, candidates=[], abstained=True, reason=reason, metadata={})
