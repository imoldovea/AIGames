# backtrack_maze_solver.py
from __future__ import annotations

import logging
import traceback
from time import perf_counter
from typing import Callable, Dict, List, Optional, Set, Tuple

from maze import Maze
from maze_solver import MazeSolver, SolveResult, Pos
from utils import setup_logging, load_mazes


class BacktrackingMazeSolver(MazeSolver):
    """
    Stack-based DFS (backtracking-style) with neighbor caching.
    Returns SolveResult (success/path/visited/steps/etc).
    """

    name = "Backtracking"

    def __init__(self, maze: Maze):
        super().__init__(maze)
        # Cache for neighbor calculations
        self._neighbors_cache: Dict[Pos, List[Pos]] = {}

    def solve(self) -> SolveResult:
        """
        Attempts to solve the maze using DFS with caching.

        Returns:
            SolveResult(success, path, visited, steps, error, extra)
        """
        if self.maze.exit is None:
            return self.make_result([], error="Maze exit is not set.")

        try:
            # Pre-calculate valid neighbors once per solve call
            self._neighbors_cache.clear()
            self._precompute_neighbors()

            visited: Set[Pos] = set()
            path = self._dfs(self.maze.start_position, visited)

            # Optional: update maze.path only if solved
            if self.is_valid_solution(path):
                try:
                    self.maze.path = path
                except Exception:
                    pass

            return self.make_result(path, visited=len(visited))

        except Exception as e:
            logging.error(f"Backtracking solver error: {e}")
            logging.error(traceback.format_exc())
            return self.make_result([], error=str(e))

    def solve_with_callback(
        self,
        callback: Optional[Callable[..., None]] = None,
        *,
        callback_every: int = 1,
    ) -> SolveResult:
        """
        Same algorithm as solve(), but calls callback periodically.

        callback signature is flexible; we pass:
          - position: current cell
          - visited: copy of visited set (optional)
          - path: current reconstructed path from start to position
          - result: (optional) only on final call
        """
        if self.maze.exit is None:
            return self.make_result([], error="Maze exit is not set.")

        try:
            self._neighbors_cache.clear()
            self._precompute_neighbors()

            visited: Set[Pos] = set()
            parent: Dict[Pos, Optional[Pos]] = {self.maze.start_position: None}

            stack: List[Pos] = [self.maze.start_position]
            steps = 0

            while stack:
                position = stack.pop()
                if position in visited:
                    continue

                visited.add(position)
                steps += 1

                if callback and (steps % max(1, callback_every) == 0):
                    callback(
                        position=position,
                        visited=visited.copy(),
                        path=self._reconstruct_path(position, parent),
                    )

                if position == self.maze.exit:
                    path = self._reconstruct_path(position, parent)
                    result = self.make_result(path, visited=len(visited))
                    if callback:
                        callback(position=position, visited=visited.copy(), path=path, result=result)
                    return result

                # Push neighbors (reversed to preserve similar exploration order as your original)
                for nb in reversed(self._get_cached_neighbors(position)):
                    if nb not in visited:
                        # only set parent once (first time discovered)
                        if nb not in parent:
                            parent[nb] = position
                        stack.append(nb)

            # Not found
            result = self.make_result([], visited=len(visited))
            if callback:
                callback(
                    position=getattr(self.maze, "current_position", None),
                    visited=visited.copy(),
                    path=[],
                    result=result,
                )
            return result

        except Exception as e:
            logging.error(f"Backtracking solver callback error: {e}")
            logging.error(traceback.format_exc())
            return self.make_result([], error=str(e))

    # ------------------------
    # Internals
    # ------------------------
    def _precompute_neighbors(self) -> None:
        rows, cols = self.maze.rows, self.maze.cols
        for r in range(rows):
            for c in range(cols):
                pos = (r, c)
                if not self.maze.is_wall(pos):
                    self._neighbors_cache[pos] = list(self.maze.get_neighbors(pos))

    def _get_cached_neighbors(self, position: Pos) -> List[Pos]:
        if position not in self._neighbors_cache:
            self._neighbors_cache[position] = list(self.maze.get_neighbors(position))
        return self._neighbors_cache[position]

    def _dfs(self, start: Pos, visited: Set[Pos]) -> List[Pos]:
        """
        Stack-based DFS with parent pointers (fast, avoids recursion limit).
        Returns reconstructed path if found; else [].
        """
        parent: Dict[Pos, Optional[Pos]] = {start: None}
        stack: List[Pos] = [start]

        while stack:
            position = stack.pop()
            if position in visited:
                continue

            visited.add(position)

            if position == self.maze.exit:
                return self._reconstruct_path(position, parent)

            for nb in reversed(self._get_cached_neighbors(position)):
                if nb not in visited:
                    if nb not in parent:
                        parent[nb] = position
                    stack.append(nb)

        return []

    def _reconstruct_path(self, end_position: Pos, parent: Dict[Pos, Optional[Pos]]) -> List[Pos]:
        path: List[Pos] = []
        cur: Optional[Pos] = end_position
        while cur is not None:
            path.append(cur)
            cur = parent.get(cur)
        path.reverse()
        return path


# Example test function (kept, but now uses SolveResult)
def backtracking_solver() -> None:
    try:
        mazes = load_mazes("input/mazes.h5")
        for i, maze in enumerate(mazes):
            solver = BacktrackingMazeSolver(maze)
            result = solver.solve()

            if result.success:
                logging.debug(f"Maze {i + 1} solved. steps={result.steps} visited={result.visited}")
            else:
                logging.debug(f"Maze {i + 1} NOT solved. error={result.error or '-'} visited={result.visited}")

            # Visualize (only set_solution if solved)
            if result.success:
                maze.set_solution(result.path)
            maze.plot_maze(show_path=False, show_solution=result.success, show_position=False)

    except Exception as e:
        logging.error(f"An error occurred: {e}\n\nStack Trace:{traceback.format_exc()}")
        raise


if __name__ == "__main__":
    setup_logging()
    backtracking_solver()