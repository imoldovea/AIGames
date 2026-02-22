# pladge_maze_solver.py
from __future__ import annotations

import logging
import traceback
from typing import Callable, Dict, List, Optional, Tuple

from maze import Maze
from maze_solver import MazeSolver, SolveResult, Pos
from utils import load_mazes, setup_logging


class PledgeMazeSolver(MazeSolver):
    """
    Pledge algorithm adapted to a Maze where the exit may not be known in advance
    via coordinates, but Maze.at_exit() tells us when we reach it.

    Requirements assumed on Maze:
      - start_position: (row, col)
      - current_position: updated after Maze.move(...)
      - is_valid_move(position) -> bool
      - move(position) -> updates current_position
      - at_exit() -> bool
      - exit (optional): (row,col) if known (used only for validation / benchmarking)
    """

    name = "Pledge"

    def __init__(self, maze: Maze, *, max_steps: int = 10_000, main_direction: int = 0):
        super().__init__(maze)
        self.max_steps = int(max_steps)

        # Directions: 0=North, 1=East, 2=South, 3=West
        self.directions: List[Pos] = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        # "Main" direction for pledge (default north=0); can be configured
        self.main_direction = int(main_direction) % 4

    def solve(self) -> SolveResult:
        if getattr(self.maze, "start_position", None) is None:
            return self.make_result([], error="Maze start_position is not set.")

        try:
            path, visited_count, steps, net_turn, reached_exit, max_steps_reached = self._run_pledge(
                callback=None,
                callback_every=1,
            )

            # IMPORTANT:
            # - reached_exit is authoritative for this algorithm (uses maze.at_exit())
            # - make_result() validates against maze.exit if present; we override success to match reached_exit.
            result = self.make_result(
                path if reached_exit else [],
                visited=visited_count,
                extra={
                    "steps": steps,
                    "net_turn": net_turn,
                    "max_steps": self.max_steps,
                    "max_steps_reached": max_steps_reached,
                },
            )
            result.success = bool(reached_exit)
            result.steps = steps

            # Only set maze.path if solved
            if result.success:
                try:
                    self.maze.path = path
                except Exception:
                    pass

            return result

        except Exception as e:
            logging.error(f"Pledge solver error: {e}")
            logging.error(traceback.format_exc())
            return self.make_result([], error=str(e))

    def solve_with_callback(
        self,
        callback: Optional[Callable[..., None]] = None,
        *,
        callback_every: int = 1,
    ) -> SolveResult:
        try:
            path, visited_count, steps, net_turn, reached_exit, max_steps_reached = self._run_pledge(
                callback=callback,
                callback_every=callback_every,
            )

            result = self.make_result(
                path if reached_exit else [],
                visited=visited_count,
                extra={
                    "steps": steps,
                    "net_turn": net_turn,
                    "max_steps": self.max_steps,
                    "max_steps_reached": max_steps_reached,
                },
            )
            result.success = bool(reached_exit)
            result.steps = steps

            if callback:
                callback(
                    position=(path[-1] if path else getattr(self.maze, "current_position", None)),
                    visited=visited_count,
                    path=(path.copy() if reached_exit else []),
                    result=result,
                )

            return result

        except Exception as e:
            logging.error(f"Pledge solver callback error: {e}")
            logging.error(traceback.format_exc())
            return self.make_result([], error=str(e))

    # ------------------------
    # Internals
    # ------------------------
    def _is_free(self, position: Pos) -> bool:
        return bool(self.maze.is_valid_move(position))

    def _move(self, position: Pos, direction_index: int) -> Pos:
        dr, dc = self.directions[direction_index]
        return (position[0] + dr, position[1] + dc)

    def _run_pledge(
        self,
        *,
        callback: Optional[Callable[..., None]],
        callback_every: int,
    ) -> Tuple[List[Pos], int, int, int, bool, bool]:
        """
        Runs the Pledge walk.
        Returns:
          path, visited_count, steps, net_turn, reached_exit, max_steps_reached
        """
        current_position: Pos = self.maze.start_position

        # Initialize Maze's current position
        self.maze.move(current_position)

        path: List[Pos] = [current_position]
        visited = {current_position}

        main_direction = self.main_direction
        current_direction = main_direction

        wall_following = False
        net_turn = 0  # net quarter turns (each +1 is 90° right turn, -1 left)

        steps = 0
        max_steps_reached = False

        while steps < self.max_steps:
            steps += 1

            # Periodic callback
            if callback and (steps % max(1, callback_every) == 0):
                callback(
                    position=current_position,
                    visited=visited.copy(),
                    path=path.copy(),
                )

            # Check solved condition (authoritative for Pledge here)
            if self.maze.at_exit():
                return path, len(visited), steps, net_turn, True, False

            if not wall_following:
                forward = self._move(current_position, current_direction)
                if self._is_free(forward):
                    self.maze.move(forward)
                    current_position = self.maze.current_position
                    path.append(current_position)
                    visited.add(current_position)
                    continue

                # hit a wall -> start wall following
                wall_following = True
                net_turn = 0

                # turn right to begin wall-following
                current_direction = (current_direction + 1) % 4
                net_turn += 1

                next_pos = self._move(current_position, current_direction)
                if self._is_free(next_pos):
                    self.maze.move(next_pos)
                    current_position = self.maze.current_position
                    path.append(current_position)
                    visited.add(current_position)
                # else: stay and loop again

            else:
                # wall-following:
                left_dir = (current_direction - 1) % 4
                front_dir = current_direction
                right_dir = (current_direction + 1) % 4

                # Prefer left, then forward, then right, else U-turn
                left_pos = self._move(current_position, left_dir)
                front_pos = self._move(current_position, front_dir)
                right_pos = self._move(current_position, right_dir)

                if self._is_free(left_pos):
                    current_direction = left_dir
                    net_turn -= 1
                    self.maze.move(left_pos)
                elif self._is_free(front_pos):
                    self.maze.move(front_pos)
                elif self._is_free(right_pos):
                    current_direction = right_dir
                    net_turn += 1
                    self.maze.move(right_pos)
                else:
                    # Dead end: 180° turn
                    current_direction = (current_direction + 2) % 4
                    net_turn += 2
                    back_pos = self._move(current_position, current_direction)
                    if self._is_free(back_pos):
                        self.maze.move(back_pos)
                    else:
                        # truly stuck
                        break

                current_position = self.maze.current_position
                path.append(current_position)
                visited.add(current_position)

                # Exit wall-following if net_turn reset and main direction is free
                if net_turn == 0 and self._is_free(self._move(current_position, main_direction)):
                    wall_following = False
                    current_direction = main_direction

        # Max steps exceeded / not solved
        max_steps_reached = steps >= self.max_steps
        if max_steps_reached:
            logging.error("Pledge: maximum steps exceeded without reaching exit.")
        return path, len(visited), steps, net_turn, False, max_steps_reached


def pledge_solver(file_path: str = "input/mazes.h5", samples: int = 10) -> None:
    mazes = load_mazes(file_path, samples=samples)
    mazes = sorted(mazes, key=lambda m: m.complexity, reverse=False)

    for i, maze in enumerate(mazes, start=1):
        logging.info(f"Solving maze {i}/{len(mazes)} (index={maze.index}) with Pledge...")

        solver = PledgeMazeSolver(maze)
        result = solver.solve()

        if result.success:
            maze.set_solution(result.path)
            maze.plot_maze(show_path=False, show_solution=True, show_position=False)
        else:
            # For failed pledge, don't mark solution; show unsolved
            maze.plot_maze(show_path=False, show_solution=False, show_position=False)

        logging.info(
            f"Pledge result: success={result.success} steps={result.steps} "
            f"visited={result.visited} extra={result.extra}"
        )


if __name__ == "__main__":
    setup_logging()
    pledge_solver(file_path="input/mazes.h5", samples=0)  # 0 = load all