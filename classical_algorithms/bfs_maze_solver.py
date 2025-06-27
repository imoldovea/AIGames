import logging
import traceback
from collections import deque

from numpy.f2py.auxfuncs import throw_error

from maze import Maze
from maze_solver import MazeSolver
from utils import setup_logging, load_mazes, save_movie


class BFSMazeSolver(MazeSolver):
    def __init__(self, maze):
        """
        Initializes the BFSMazeSolver with a Maze object.
        Args:
            maze (Maze): The maze to solve.
        """
        super().__init__(maze)
        self.maze = maze
        maze.set_algorithm(self.__class__.__name__)

    def solve(self):
        """
        Solves the maze using the Breadth-First Search (BFS) algorithm.

        Returns:
            A list of (row, col) coordinates representing the path from the start to the exit,
            or None if no solution is found.
        """
        if self.maze.exit is None:
            raise ValueError("Maze exit is not set.")

        start = self.maze.start_position
        maze_exit = self.maze.exit

        # Queue for BFS and dictionary to store the parent of each visited position.
        queue = deque([start])
        visited = {start}
        parent = {start: None}

        # Perform BFS until the queue is empty or the exit is found.
        while queue:
            current = queue.popleft()

            if current == maze_exit:
                return self.reconstruct_path(parent, current)

            # Explore the neighbors.
            for neighbor in self.maze.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current
                    queue.append(neighbor)
                    # self.maze.move(current)

        # No solution found
        return None

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


# Example test function for the BFS solver
def bfs_solver():
    """
    Test function that loads an array of mazes from 'input/mazes.npy',
    creates a Maze object using the first maze in the array, sets an exit,
    solves the maze using the BFSMazeSolver, and displays the solution.
    """
    try:
        # Load mazes
        mazes = load_mazes("input/mazes.h5", 10)
        mazes = sorted(mazes, key=lambda maze: maze.complexity, reverse=False)
        mazes = mazes[9:]
        # Iterate through each maze in the array
        for i, maze in enumerate(mazes):

            logging.debug(f"Solving maze {i + 1}...")

            # Create a Maze object from the maze matrix
            maze.set_animate(False)
            maze.set_save_movie(True)

            # Instantiate the BFS maze solver
            solver = BFSMazeSolver(maze)
            solution = solver.solve()

            if solution:
                logging.debug(f"Maze {i + 1} solution found:")
                logging.debug(solution)
            else:
                logging.debug(f"No solution found for maze {i + 1}.")

            # Visualize the solved maze (with the solution path highlighted)
            maze.set_solution(solution)
            maze.plot_maze(show_path=False, show_solution=True)
            save_movie([maze], f"output/solved_maze_{maze.index}.mp4")
        logging.info("All mazes solved.")
    except Exception as e:
        logging.error(f"An error occurred: {e}\n\nStack Trace:{traceback.format_exc()}")
        throw_error(e)


if __name__ == '__main__':
    # setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.debug("Logging is configured.")

    bfs_solver()
