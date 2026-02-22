# optimized_backtrack_maze_solver.py
from __future__ import annotations

import logging
import traceback
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from maze import Maze
from maze_solver import MazeSolver, SolveResult, Pos
from utils import setup_logging, load_mazes

logger = logging.getLogger(__name__)


class OptimizedBacktrackingMazeSolver(MazeSolver):
    """
    DFS/backtracking with:
      - NumPy visited grid for speed
      - cached neighbors per cell (as NumPy arrays)
      - neighbor ordering by Manhattan distance to the goal (heuristic)
    Returns SolveResult (benchmark-friendly).
    """

    name = "OptimizedBacktracking"

    def __init__(self, maze: Maze):
        super().__init__(maze)

        # Pre-compute and cache valid neighbors for each cell
        self._neighbors_cache: Dict[Pos, np.ndarray] = self._precompute_neighbors()

    def solve(self) -> SolveResult:
        if self.maze.exit is None:
            return self.make_result([], error="Maze exit is not set.")

        try:
            start = self.maze.start_position
            goal = self.maze.exit

            if start == goal:
                return self.make_result([start], visited=1)

            visited = np.zeros((self.maze.rows, self.maze.cols), dtype=bool)
            path: List[Pos] = []

            success = self._dfs(current=start, target=goal, visited=visited, path=path)

            visited_count = int(visited.sum())
            if success:
                try:
                    self.maze.path = path
                except Exception:
                    pass
                return self.make_result(path, visited=visited_count)

            return self.make_result([], visited=visited_count)

        except Exception as e:
            logger.error(f"Optimized backtracking solver error: {e}")
            logger.error(traceback.format_exc())
            return self.make_result([], error=str(e))

    def solve_with_callback(
        self,
        callback: Optional[Callable[..., None]] = None,
        *,
        callback_every: int = 1,
    ) -> SolveResult:
        """
        Same DFS as solve(), but emits callback periodically.

        callback(...) receives:
          - position: current cell
          - visited: either a bool grid (numpy) or a set-like (we pass counts + path)
          - path: current path
          - result: only on final callback
        """
        if self.maze.exit is None:
            return self.make_result([], error="Maze exit is not set.")

        try:
            start = self.maze.start_position
            goal = self.maze.exit

            if start == goal:
                result = self.make_result([start], visited=1)
                if callback:
                    callback(position=start, visited=1, path=[start], result=result)
                return result

            visited = np.zeros((self.maze.rows, self.maze.cols), dtype=bool)
            path: List[Pos] = []
            steps = 0

            # We'll do DFS but count callback steps on node-entry
            def dfs_cb(cur: Pos) -> None:
                nonlocal steps
                steps += 1
                if callback and (steps % max(1, callback_every) == 0):
                    callback(
                        position=cur,
                        visited=int(visited.sum()),
                        path=path.copy(),
                    )

            success = self._dfs(
                current=start,
                target=goal,
                visited=visited,
                path=path,
                on_enter=dfs_cb,
            )

            visited_count = int(visited.sum())
            if success:
                result = self.make_result(path, visited=visited_count)
            else:
                result = self.make_result([], visited=visited_count)

            if callback:
                callback(
                    position=(path[-1] if path else getattr(self.maze, "current_position", None)),
                    visited=visited_count,
                    path=(path.copy() if success else []),
                    result=result,
                )

            return result

        except Exception as e:
            logger.error(f"Optimized backtracking callback error: {e}")
            logger.error(traceback.format_exc())
            return self.make_result([], error=str(e))

    # ------------------------
    # Internals
    # ------------------------
    def _precompute_neighbors(self) -> Dict[Pos, np.ndarray]:
        """
        Precompute valid neighbors for each cell.
        Returns dict[pos] -> np.ndarray of neighbor positions (shape Nx2).
        """
        cache: Dict[Pos, np.ndarray] = {}
        directions = np.array([(0, 1), (1, 0), (0, -1), (-1, 0)], dtype=np.int16)

        for r in range(self.maze.rows):
            for c in range(self.maze.cols):
                if self.maze.is_wall((r, c)):
                    continue

                candidates = np.array([r, c], dtype=np.int16) + directions
                valid: List[Pos] = []

                for nr, nc in candidates:
                    if 0 <= nr < self.maze.rows and 0 <= nc < self.maze.cols:
                        if not self.maze.is_wall((int(nr), int(nc))):
                            valid.append((int(nr), int(nc)))

                cache[(r, c)] = np.array(valid, dtype=np.int16)

        return cache

    def _get_cached_neighbors(self, position: Pos) -> np.ndarray:
        return self._neighbors_cache.get(position, np.empty((0, 2), dtype=np.int16))

    def _dfs(
        self,
        current: Pos,
        target: Pos,
        visited: np.ndarray,
        path: List[Pos],
        *,
        on_enter: Optional[Callable[[Pos], None]] = None,
    ) -> bool:
        """
        Recursive DFS (kept from your original).
        Uses NumPy visited and cached neighbors; sorts neighbors by heuristic.
        """
        path.append(current)
        r, c = current
        visited[r, c] = True

        if on_enter:
            on_enter(current)

        if current == target:
            return True

        neighbors = self._get_cached_neighbors(current)

        if neighbors.size > 0:
            # Sort by Manhattan distance to target (heuristic ordering)
            if len(neighbors) > 1:
                tr, tc = target
                distances = np.abs(neighbors[:, 0] - tr) + np.abs(neighbors[:, 1] - tc)
                order = np.argsort(distances)
                neighbors = neighbors[order]

            for nr, nc in neighbors:
                nr_i, nc_i = int(nr), int(nc)
                if visited[nr_i, nc_i]:
                    continue

                if self._dfs((nr_i, nc_i), target, visited, path, on_enter=on_enter):
                    return True

        # Backtrack
        path.pop()
        return False


# Example test function (updated)
def solver() -> None:
    try:
        mazes = load_mazes("input/mazes.h5", samples=0)
        logging.info(f"Loaded {len(mazes)} mazes.")

        for i, maze in enumerate(mazes):
            s = OptimizedBacktrackingMazeSolver(maze)
            result = s.solve()

            if result.success:
                logging.debug(f"Maze {i + 1} solved. steps={result.steps} visited={result.visited}")
                maze.set_solution(result.path)
                maze.plot_maze(show_path=False, show_solution=True, show_position=False)
            else:
                logging.debug(f"Maze {i + 1} NOT solved. visited={result.visited} error={result.error or '-'}")
                maze.plot_maze(show_path=False, show_solution=False, show_position=False)

    except Exception as e:
        logging.error(f"An error occurred: {e}\n\nStack Trace:{traceback.format_exc()}")
        raise


if __name__ == "__main__":
    setup_logging()
    solver()