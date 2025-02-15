from numpy.f2py.auxfuncs import throw_error

from maze_solver import MazeSolver
from maze import Maze
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

class BacktrackingMazeSolver(MazeSolver):
    def __init__(self, maze):
        """
        Initializes the BacktrackingMazeSolver with a Maze object.
        Args:
            maze (Maze): The maze to solve.
        """
        self.maze = maze
        maze.set_algorithm(self.__class__.__name__)

    def solve(self):
        """
        Attempts to solve the maze using recursive backtracking.

        Returns:
            A list of (row, col) coordinates representing the path from the start to the exit,
            or None if no solution is found.
        """
        if self.maze.exit is None:
            raise ValueError("Maze exit is not set.")
        visited = set()
        path = []
        solution = self._dfs(self.maze.start_position, visited, path)
        if solution is not None:
            # Optionally update the maze's path to the solution found
            self.maze.path = solution
        return solution

    def _dfs(self, current, visited, path):
        """
        Helper method that performs depth-first search from the current position.
        """
        visited.add(current)
        path.append(current)

        # Animate the current position
        self.maze.move(current)

        # Check if we've reached the exit
        if current == self.maze.exit:
            return path.copy()

        # Explore valid neighbors not yet visited
        for neighbor in self.maze.get_neighbors(current):
            if neighbor not in visited:
                sol = self._dfs(neighbor, visited, path)
                if sol is not None:
                    return sol

        # Animate the current position
        self.maze.move(current,backtrack=True)
        # Backtrack if no path found from the current position
        path.pop()

        return None

# Example test function
def test_backtracking_solver():
    """
    Test function that loads an array of mazes from 'input/mazes.npy',
    creates a Maze object using the first maze in the array, sets an exit,
    solves the maze using the BacktrackingMazeSolver, and displays the solution.
    """
    try:
        # Load the numpy file containing an array of mazes
        maze_array = np.load("input/mazes.npy", allow_pickle=True)

        # Iterate through each maze in the array
        for i, maze_matrix in enumerate(maze_array):
            logging.debug(f"Solving maze {i + 1}...")

            # Create a Maze object from the maze matrix
            maze_obj = Maze(maze_matrix)
            maze_obj.set_animate(True)
            maze_obj.set_save_movie(True)

            # Instantiate the backtracking maze solver
            solver = BacktrackingMazeSolver(maze_obj)
            solution = solver.solve()

            if solution:
                logging.debug(f"Maze {i + 1} solution found:")
                logging.debug(solution)
            else:
                logging.debug(f"No solution found for maze {i + 1}.")

            # Visualize the solved maze (with the solution path highlighted)
            maze_obj.set_solution(solution)
            maze_obj.plot_maze(show_path=False, show_solution=True)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        throw_error(e)



if __name__ == '__main__':
    test_backtracking_solver()