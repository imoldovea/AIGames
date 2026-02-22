# maze_solver.py
from __future__ import annotations

import cProfile
import io
import pstats
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from maze import Maze

Pos = Tuple[int, int]


@dataclass(slots=True)
class SolveResult:
    """
    Standardised result object returned by every solver.

    - success: True only if the solver actually reaches the exit.
    - path: the returned path (may be empty even if attempted).
    - steps: typically len(path) or number of moves executed.
    - visited: number of explored states (if the algorithm tracks this).
    - time_ms: filled by the experiment runner (recommended).
    - error: set if an exception occurred; success should be False then.
    - extra: free-form extra metrics (frontier size, backtracks, etc.).
    """
    success: bool
    path: List[Pos] = field(default_factory=list)

    steps: int = 0
    visited: int = 0
    time_ms: float = 0.0
    error: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def with_time_ms(self, time_ms: float) -> "SolveResult":
        self.time_ms = time_ms
        return self


class MazeSolver(ABC):
    """
    Base interface for all maze solvers.

    Design goals:
    - Every solver returns SolveResult (consistent success definition).
    - Experiment runner can benchmark uniformly.
    - Optional callback support for animation / debugging.
    """

    # Friendly name for reporting; default to class name if not overridden.
    name: str = ""

    def __init__(self, maze: Maze):
        self.maze = maze

    # ---------- Required API ----------

    @abstractmethod
    def solve(self) -> SolveResult:
        """
        Solve the maze and return a SolveResult.
        """
        raise NotImplementedError

    # ---------- Optional API (for animation / tracing) ----------

    def solve_with_callback(
        self,
        callback: Optional[Callable[..., None]] = None,
        *,
        callback_every: int = 1,
    ) -> SolveResult:
        """
        Optional: solvers can override this for step-by-step animation.

        Default implementation: just runs solve() and fires a single callback
        at the end (useful for a "final render" callback).
        """
        result = self.solve()
        if callback:
            callback(
                position=(result.path[-1] if result.path else getattr(self.maze, "current_position", None)),
                visited=None,
                path=result.path,
                result=result,
            )
        return result

    # ---------- Shared helpers ----------

    @property
    def algorithm_name(self) -> str:
        return self.name or self.__class__.__name__

    def is_valid_solution(self, path: List[Pos]) -> bool:
        """
        Definition of "solved" used for benchmarking consistency.
        """
        if not path:
            return False
        exit_pos = getattr(self.maze, "exit", None)
        if exit_pos is None:
            return False
        return path[-1] == exit_pos

    def make_result(
        self,
        path: Optional[List[Pos]],
        *,
        visited: int = 0,
        error: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> SolveResult:
        """
        Helper to create SolveResult with consistent success logic.
        """
        p = path or []
        success = False if error else self.is_valid_solution(p)

        return SolveResult(
            success=success,
            path=p,
            steps=len(p),
            visited=visited,
            error=error,
            extra=extra or {},
        )

    # ---------- Utility: profiler decorator ----------

    @staticmethod
    def profiled(func):
        """
        Decorator for quick profiling during development.
        (Use the experiment runner for benchmarking instead.)
        """
        def wrapper(*args, **kwargs):
            profiler = cProfile.Profile()
            profiler.enable()
            result = func(*args, **kwargs)
            profiler.disable()
            s = io.StringIO()
            stats = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
            stats.print_stats()
            print(s.getvalue())
            return result

        return wrapper

    # ---------- (Optional) legacy helper ----------

    def loss(self) -> float:
        """
        Kept from your previous design:
        Manhattan distance + penalty for steps taken so far.
        Useful for ML/agent experiments.
        """
        exit_pos = getattr(self.maze, "exit", None)
        if exit_pos is None:
            raise ValueError("Exit is not defined for the maze.")

        cur = getattr(self.maze, "current_position", None)
        if cur == exit_pos:
            return 0.0

        current_r, current_c = cur
        exit_r, exit_c = exit_pos
        manhattan_distance = abs(current_r - exit_r) + abs(current_c - exit_c)

        path = getattr(self.maze, "path", [])
        path_penalty = 0.1 * len(path)
        return manhattan_distance + path_penalty