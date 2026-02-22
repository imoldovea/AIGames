# Python
import logging
from collections import deque

from maze_solver import MazeSolver
from utils import load_mazes, setup_logging


class PledgeMazeSolver(MazeSolver):
    """
    Maze solver using the Pledge Algorithm adapted to a Maze object where the exit
    is not known in advance. The Maze object is assumed to provide:
      - start_position: the starting cell (row, col)
      - current_position: the current cell (row, col) that will be updated after every move
      - is_valid_move(position): returns True if the position is free to move into.
      - at_exit(): returns True if the current_position is the exit.
    """

    def __init__(self, maze):
        """
        Initialize the solver with a Maze instance.

        Args:
            maze: an instance of Maze which has methods:
                  - is_valid_move(position)
                  - at_exit()
                  and attributes:
                  - start_position (tuple)
                  - current_position (tuple)
        """
        self.maze = maze
        maze.set_algorithm(self.__class__.__name__)
        # Set starting position from maze
        self.start = maze.start_position
        # Directions: 0=North, 1=East, 2=South, 3=West
        self.directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    def _is_free(self, position):
        """
        Use Maze's is_valid_move method to check if a cell is free.
        """
        return self.maze.is_valid_move(position)

    def _move(self, position, direction_index):
        """
        Returns a new position by moving from the current cell in the given direction.
        """
        dr, dc = self.directions[direction_index]
        return (position[0] + dr, position[1] + dc)

    def solve(self):
        """
        Solve the maze using the Pledge algorithm. Each move is executed via
        Maze.move(), ensuring the Maze's current_position is updated.

        Returns:
            path (list of positions): The sequence of (row, col) tuples of the path.
        """
        current_position = self.start
        # Initialize Maze's current position
        self.maze.move(current_position)
        path = [current_position]

        # The initial main direction: for example, north (0)
        main_direction = 0
        current_direction = main_direction

        wall_following = False
        net_turn = 0  # net quarter turns (each +1 is a 90° right turn)

        max_steps = 10000
        steps = 0

        while steps < max_steps:
            steps += 1

            # Check if the maze is solved by asking Maze via at_exit().
            if self.maze.at_exit():
                logging.debug("Exit found at position: " + str(current_position))
                break

            if not wall_following:
                forward = self._move(current_position, current_direction)
                if self._is_free(forward):
                    self.maze.move(forward)
                    current_position = self.maze.current_position
                    path.append(current_position)
                    continue
                else:
                    # Switch to wall-following mode after hitting a wall.
                    wall_following = True
                    net_turn = 0
                    # Turn right to begin wall-following.
                    current_direction = (current_direction + 1) % 4
                    net_turn += 1
                    next_pos = self._move(current_position, current_direction)
                    if self._is_free(next_pos):
                        self.maze.move(next_pos)
                        current_position = self.maze.current_position
                        path.append(current_position)
                    else:
                        continue
            else:
                # Wall-following mode: try turning left first.
                left_dir = (current_direction - 1) % 4
                front_dir = current_direction
                right_dir = (current_direction + 1) % 4

                if self._is_free(self._move(current_position, left_dir)):
                    current_direction = left_dir
                    net_turn -= 1
                    new_pos = self._move(current_position, current_direction)
                    self.maze.move(new_pos)
                    current_position = self.maze.current_position
                    path.append(current_position)
                elif self._is_free(self._move(current_position, front_dir)):
                    new_pos = self._move(current_position, current_direction)
                    self.maze.move(new_pos)
                    current_position = self.maze.current_position
                    path.append(current_position)
                elif self._is_free(self._move(current_position, right_dir)):
                    current_direction = right_dir
                    net_turn += 1
                    new_pos = self._move(current_position, current_direction)
                    self.maze.move(new_pos)
                    current_position = self.maze.current_position
                    path.append(current_position)
                else:
                    # Dead end: perform a 180° turn.
                    current_direction = (current_direction + 2) % 4
                    net_turn += 2
                    new_pos = self._move(current_position, current_direction)
                    if self._is_free(new_pos):
                        self.maze.move(new_pos)
                        current_position = self.maze.current_position
                        path.append(current_position)
                    else:
                        logging.error("No available moves during wall-following.")
                        break

                # Check if conditions allow exit from wall-following mode.
                if net_turn == 0 and self._is_free(self._move(current_position, main_direction)):
                    wall_following = False
                    current_direction = main_direction

        if steps >= max_steps:
            logging.error("Maximum steps exceeded without reaching the exit.")
        return path

    def solve_with_callback(self, callback=None):
        queue = deque([self.maze.start_position])
        visited = {self.maze.start_position}
        parent = {self.maze.start_position: None}

        while queue:
            current = queue.popleft()

            # invoke callback with current position, path so far, etc.
            if callback:
                callback(position=current, visited=visited.copy(), path=self.reconstruct_path(parent, current))

            if current == self.maze.exit:
                return self.reconstruct_path(parent, current)

            for neighbor in self.maze.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current
                    queue.append(neighbor)

        return []

    def reconstruct_path(self, parent, current):
        path = []
        while current:
            path.append(current)
            current = parent[current]
        path.reverse()
        return path


def pledge_solver(file_path: str = "input/mazes.h5", samples: int = 10) -> None:
    mazes = load_mazes(file_path, samples=samples)
    mazes = sorted(mazes, key=lambda m: m.complexity, reverse=False)

    for i, maze in enumerate(mazes, start=1):
        logging.info(f"Solving maze {i}/{len(mazes)} (index={maze.index}) with Pledge...")

        solver = PledgeMazeSolver(maze)
        path = solver.solve()

        maze.set_solution(path)
        maze.plot_maze(show_path=False, show_solution=True, show_position=False)


if __name__ == "__main__":
    setup_logging()
    pledge_solver(file_path="input/mazes.h5", samples=0)  # 0 = load all
