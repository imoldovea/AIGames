from abc import ABC, abstractmethod
from maze import Maze
import cProfile, pstats, io


class MazeSolver(ABC):
    def __init__(self, maze: Maze):
        """
        Initializes the solver with a Maze instance.
        """
        self.maze = maze

    @abstractmethod
    def solve(self):
        """
        Abstract method for solving the maze.

        Concrete implementations should return a list of (row, col) coordinates representing the path from
        the starting position to the exit.
        """
        pass

    def profile(func):
        def wrapper(*args, **kwargs):
            profiler = cProfile.Profile()
            profiler.enable()
            result = func(*args, **kwargs)
            profiler.disable()
            s = io.StringIO()
            stats = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            stats.print_stats()
            print(s.getvalue())
            return result

        return wrapper

    def loss(self):
        """
        A sample loss function that computes a score based on:
          - The Manhattan distance from the current position to the exit
          - A small penalty proportional to the number of steps taken

        This can be useful for guiding search or reinforcement learning strategies.
        """
        if self.maze.exit is None:
            raise ValueError("Exit is not defined for the maze.")
        current_r, current_c = self.maze.current_position
        exit_r, exit_c = self.maze.exit
        manhattan_distance = abs(current_r - exit_r) + abs(current_c - exit_c)
        # Adding a penalty for the number of steps taken (0.1 per step)
        path_penalty = 0.1 * len(self.maze.path)
        return manhattan_distance + path_penalty