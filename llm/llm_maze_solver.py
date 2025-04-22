# llm_maze_solver.py
# LLMMazesSolver
import logging

import numpy as np

from maze_solver import MazeSolver


class LLMMazesSolver(MazeSolver):
    """
    Maze solver using the an LLM adapted to a Maze object where the exit
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
        # Directions: 0=North, 1=East, 2=South, 3=West
        self.directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        super().__init__(maze)
        self.maze = maze
        maze.set_algorithm(self.__class__.__name__)

    def solve(self):
        """
        Solve the maze using LLMs. Each move is executed via
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

        if steps >= max_steps:
            logging.error("Maximum steps exceeded without reaching the exit.")
        return path


# Example usage:
if __name__ == "__main__":
    # Create a simple 5x5 maze where 1 (or Maze.WALL) indicates a wall and 0 indicates a free cell.
    # For demonstration purposes, we assume the Maze class uses 1 for walls.
    maze_grid = np.array([
        [1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 3, 0, 1],
        [1, 1  1, 1, 1]
    ])
