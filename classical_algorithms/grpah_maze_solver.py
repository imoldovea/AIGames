# grpah_maze_solver.py
from __future__ import annotations

import heapq
import logging
import traceback
from typing import Callable, Dict, List, Optional, Set, Tuple

from maze import Maze
from maze_solver import MazeSolver, SolveResult, Pos
from utils import setup_logging, load_mazes


class AStarMazeSolver(MazeSolver):
    """A maze solver that uses the A* algorithm for efficient pathfinding."""

    name = "A*"

    def __init__(self, maze: Maze):
        super().__init__(maze)

        # Pre-compute and cache valid neighbors for each cell
        self._neighbors_cache: Dict[Pos, List[Pos]] = self._precompute_neighbors()

    def solve(self) -> SolveResult:
        """Solve the maze using A*.

        Returns:
            SolveResult with success/path/visited/steps/error/extra.
        """
        if self.maze.exit is None:
            return self.make_result([], error="Maze exit is not set.")

        try:
            start = self.maze.start_position
            goal = self.maze.exit

            if start == goal:
                return self.make_result([start], visited=1)

            # Priority queue: (f_score, tie_breaker, position)
            open_heap: List[Tuple[int, int, Pos]] = []
            counter = 0

            g_score: Dict[Pos, int] = {start: 0}
            came_from: Dict[Pos, Pos] = {}

            heapq.heappush(open_heap, (self._heuristic(start, goal), counter, start))
            counter += 1

            closed: Set[Pos] = set()
            expanded = 0
            max_open = 1

            while open_heap:
                _, _, current = heapq.heappop(open_heap)

                if current in closed:
                    continue

                closed.add(current)
                expanded += 1

                if current == goal:
                    path = self._reconstruct_path(came_from, current)
                    # Optional: update maze.path only if solved
                    try:
                        self.maze.path = path
                    except Exception:
                        pass
                    return self.make_result(
                        path,
                        visited=expanded,
                        extra={"max_open_set": max_open},
                    )

                for nb in self._get_cached_neighbors(current):
                    if nb in closed:
                        continue

                    tentative_g = g_score[current] + 1
                    if tentative_g >= g_score.get(nb, 10**12):
                        continue

                    came_from[nb] = current
                    g_score[nb] = tentative_g
                    f = tentative_g + self._heuristic(nb, goal)

                    heapq.heappush(open_heap, (f, counter, nb))
                    counter += 1
                    if len(open_heap) > max_open:
                        max_open = len(open_heap)

            # No path found
            return self.make_result([], visited=expanded, extra={"max_open_set": max_open})

        except Exception as e:
            logging.error(f"A* solver error: {e}")
            logging.error(traceback.format_exc())
            return self.make_result([], error=str(e))

    def solve_with_callback(
        self,
        callback: Optional[Callable[..., None]] = None,
        *,
        callback_every: int = 1,
    ) -> SolveResult:
        """A* with step callbacks.

        Calls callback(position=..., visited=..., path=...) every N expansions.
        """
        if self.maze.exit is None:
            return self.make_result([], error="Maze exit is not set.")

        try:
            start = self.maze.start_position
            goal = self.maze.exit

            if start == goal:
                result = self.make_result([start], visited=1)
                if callback:
                    callback(position=start, visited={start}, path=[start], result=result)
                return result

            open_heap: List[Tuple[int, int, Pos]] = []
            counter = 0

            g_score: Dict[Pos, int] = {start: 0}
            came_from: Dict[Pos, Pos] = {}

            heapq.heappush(open_heap, (self._heuristic(start, goal), counter, start))
            counter += 1

            closed: Set[Pos] = set()
            expanded = 0
            max_open = 1

            while open_heap:
                _, _, current = heapq.heappop(open_heap)

                if current in closed:
                    continue

                closed.add(current)
                expanded += 1

                if callback and (expanded % max(1, callback_every) == 0):
                    callback(
                        position=current,
                        visited=closed.copy(),
                        path=self._reconstruct_path(came_from, current),
                    )

                if current == goal:
                    path = self._reconstruct_path(came_from, current)
                    result = self.make_result(path, visited=expanded, extra={"max_open_set": max_open})
                    if callback:
                        callback(position=current, visited=closed.copy(), path=path, result=result)
                    return result

                for nb in self._get_cached_neighbors(current):
                    if nb in closed:
                        continue

                    tentative_g = g_score[current] + 1
                    if tentative_g >= g_score.get(nb, 10**12):
                        continue

                    came_from[nb] = current
                    g_score[nb] = tentative_g
                    f = tentative_g + self._heuristic(nb, goal)

                    heapq.heappush(open_heap, (f, counter, nb))
                    counter += 1
                    if len(open_heap) > max_open:
                        max_open = len(open_heap)

            result = self.make_result([], visited=expanded, extra={"max_open_set": max_open})
            if callback:
                callback(
                    position=getattr(self.maze, "current_position", None),
                    visited=closed.copy(),
                    path=[],
                    result=result,
                )
            return result

        except Exception as e:
            logging.error(f"A* solver callback error: {e}")
            logging.error(traceback.format_exc())
            return self.make_result([], error=str(e))

    # ------------------------
    # Internals
    # ------------------------
    def _precompute_neighbors(self) -> Dict[Pos, List[Pos]]:
        cache: Dict[Pos, List[Pos]] = {}
        for r in range(self.maze.rows):
            for c in range(self.maze.cols):
                pos = (r, c)
                if self.maze.is_wall(pos):
                    continue
                cache[pos] = list(self.maze.get_neighbors(pos))
        return cache

    def _get_cached_neighbors(self, position: Pos) -> List[Pos]:
        return self._neighbors_cache.get(position, [])

    @staticmethod
    def _heuristic(a: Pos, b: Pos) -> int:
        # Manhattan distance
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def _reconstruct_path(came_from: Dict[Pos, Pos], current: Pos) -> List[Pos]:
        path: List[Pos] = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path


# Example test function (kept, updated)
def solver() -> None:
    try:
        mazes = load_mazes("input/mazes.h5", samples=0)
        logging.info(f"Loaded {len(mazes)} mazes.")

        for i, maze in enumerate(mazes):
            logging.debug(f"Solving maze {i + 1}...")

            solver = AStarMazeSolver(maze)
            result = solver.solve()

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