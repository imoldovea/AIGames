# bfs_maze_solver.py
from __future__ import annotations

import logging
import traceback
from collections import deque
from typing import Callable, Dict, List, Optional, Set, Tuple

from maze import Maze
from maze_solver import MazeSolver, SolveResult, Pos
from utils import setup_logging, load_mazes, save_movie


class BFSMazeSolver(MazeSolver):
    """
    Breadth-First Search (BFS) maze solver.
    Returns SolveResult with consistent success definition.
    """

    name = "BFS"

    def __init__(self, maze: Maze):
        super().__init__(maze)
        # Avoid setting algorithm here; benchmark runner should do it.
        # But keeping it commented so it doesn't affect behaviour.
        # try:
        #     maze.set_algorithm(self.__class__.__name__)
        # except Exception:
        #     pass

    def solve(self) -> SolveResult:
        if self.maze.exit is None:
            return self.make_result([], error="Maze exit is not set.")

        try:
            start = self.maze.start_position
            goal = self.maze.exit

            queue: deque[Pos] = deque([start])
            visited: Set[Pos] = {start}
            parent: Dict[Pos, Optional[Pos]] = {start: None}

            while queue:
                current = queue.popleft()

                if current == goal:
                    path = self._reconstruct_path(parent, current)
                    return self.make_result(path, visited=len(visited))

                for neighbor in self.maze.get_neighbors(current):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        parent[neighbor] = current
                        queue.append(neighbor)

            # No solution found
            return self.make_result([], visited=len(visited))

        except Exception as e:
            logging.error(f"BFS solver error: {e}")
            logging.error(traceback.format_exc())
            return self.make_result([], error=str(e))

    def solve_with_callback(
        self,
        callback: Optional[Callable[..., None]] = None,
        *,
        callback_every: int = 1,
    ) -> SolveResult:
        """
        BFS with step callbacks for animation/debug.
        Calls callback(position=..., visited=..., path=...) every N expansions.
        """
        if self.maze.exit is None:
            return self.make_result([], error="Maze exit is not set.")

        try:
            start = self.maze.start_position
            goal = self.maze.exit

            queue: deque[Pos] = deque([start])
            visited: Set[Pos] = {start}
            parent: Dict[Pos, Optional[Pos]] = {start: None}

            expansions = 0

            while queue:
                current = queue.popleft()
                expansions += 1

                if callback and (expansions % max(1, callback_every) == 0):
                    callback(
                        position=current,
                        visited=visited.copy(),
                        path=self._reconstruct_path(parent, current),
                    )

                if current == goal:
                    path = self._reconstruct_path(parent, current)
                    result = self.make_result(path, visited=len(visited))
                    if callback:
                        callback(position=current, visited=visited.copy(), path=path, result=result)
                    return result

                for neighbor in self.maze.get_neighbors(current):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        parent[neighbor] = current
                        queue.append(neighbor)

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
            logging.error(f"BFS solver callback error: {e}")
            logging.error(traceback.format_exc())
            return self.make_result([], error=str(e))

    @staticmethod
    def _reconstruct_path(parent: Dict[Pos, Optional[Pos]], current: Pos) -> List[Pos]:
        path: List[Pos] = []
        cur: Optional[Pos] = current
        while cur is not None:
            path.append(cur)
            cur = parent.get(cur)
        path.reverse()
        return path


# Example test function for the BFS solver (kept, updated for SolveResult)
def bfs_solver():
    try:
        mazes = load_mazes("input/mazes.h5", samples=0)
        mazes = sorted(mazes, key=lambda m: m.complexity, reverse=False)

        for i, maze in enumerate(mazes):
            logging.debug(f"Solving maze {i + 1}...")

            maze.set_animate(False)
            maze.set_save_movie(True)

            solver = BFSMazeSolver(maze)
            result = solver.solve()

            if result.success:
                logging.debug(f"Maze {i + 1} solved. steps={result.steps} visited={result.visited}")
                maze.set_solution(result.path)
                maze.plot_maze(show_path=False, show_solution=True)
            else:
                logging.debug(f"No solution found for maze {i + 1}. visited={result.visited} error={result.error or '-'}")
                maze.plot_maze(show_path=False, show_solution=False)

            save_movie([maze], f"output/solved_maze_{maze.index}.mp4")

        logging.info("All mazes processed.")

    except Exception as e:
        logging.error(f"An error occurred: {e}\n\nStack Trace:{traceback.format_exc()}")
        raise


if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.debug("Logging is configured.")
    bfs_solver()