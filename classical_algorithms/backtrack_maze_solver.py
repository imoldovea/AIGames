# backtrack_maze_solver.py

import logging
import traceback

from maze import Maze
from maze_solver import MazeSolver
from utils import setup_logging, load_mazes


class BacktrackingMazeSolver(MazeSolver):
    def __init__(self, maze):
        """
        Initializes the BacktrackingMazeSolver with a Maze object.
        Args:
            maze (Maze): The maze to solve.
        """
        super().__init__(maze)
        self.maze = maze
        maze.set_algorithm(self.__class__.__name__)
        # Cache for neighbor calculations
        self._neighbors_cache = {}

    def solve(self):
        """
        Attempts to solve the maze using recursive backtracking.

        Returns:
            A list of (row, col) coordinates representing the path from the start to the exit,
            or None if no solution is found.
        """
        if self.maze.exit is None:
            raise ValueError("Maze exit is not set.")

        # Pre-calculate the valid neighbors for each position to avoid redundant calculations
        self._precompute_neighbors()

        visited = set()
        path = []
        solution = self._dfs(self.maze.start_position, visited, path)

        if solution is not None:
            # Optionally update the maze's path to the solution found
            self.maze.path = solution

        return solution

    def _precompute_neighbors(self):
        """
        Pre-compute valid neighbors for each position in the maze.
        This reduces redundant calculations during the DFS traversal.
        """
        rows, cols = self.maze.rows, self.maze.cols
        for r in range(rows):
            for c in range(cols):
                if not self.maze.is_wall((r, c)):
                    self._neighbors_cache[(r, c)] = list(self.maze.get_neighbors((r, c)))

    def _get_cached_neighbors(self, position):
        """
        Return cached neighbors if available, otherwise compute them.
        """
        if position not in self._neighbors_cache:
            self._neighbors_cache[position] = list(self.maze.get_neighbors(position))
        return self._neighbors_cache[position]

    def _dfs(self, current, visited, path):
        """
        Helper method that performs depth-first search from the current position.
        Optimized with caching and early exit.
        """
        # Stack-based DFS implementation to avoid deep recursion
        stack = [(current, None)]  # (position, parent_index)
        parent_indices = {}

        while stack:
            position, parent_idx = stack.pop()

            if position in visited:
                continue

            # Add position to visited set and determine path index
            visited.add(position)

            # Update parent indices
            if parent_idx is not None:
                parent_indices[position] = parent_idx

            # Animate only when necessary
            # self.maze.move(position)

            # Check if we've reached the exit
            if position == self.maze.exit:
                # Reconstruct path
                solution = self._reconstruct_path(position, parent_indices)
                return solution

            # Add unvisited neighbors to stack
            for neighbor in reversed(self._get_cached_neighbors(position)):
                if neighbor not in visited:
                    stack.append((neighbor, position))

        return None

    def _reconstruct_path(self, end_position, parent_indices):
        """
        Reconstruct the path from start to end using parent indices.
        """
        path = [end_position]
        current = end_position

        while current in parent_indices:
            current = parent_indices[current]
            path.append(current)

        return list(reversed(path))

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


# Example test function
def backtracking_solver() -> None:
    """
    Test function that loads an array of mazes from 'input/mazes.npy',
    creates a Maze object using the first maze in the array, sets an exit,
    solves the maze using the BacktrackingMazeSolver, and displays the solution.
    """
    try:
        # Load the numpy file containing an array of mazes
        mazes = load_mazes("input/mazes.h5")

        # Iterate through each maze in the array
        for i, maze in enumerate(mazes):
            # logging.debug(f"Solving maze {i + 1}...")

            # Instantiate the backtracking maze solver
            solver = BacktrackingMazeSolver(maze)
            solution = solver.solve()

            if solution:
                logging.debug(f"Maze {i + 1} solution found:")
                logging.debug(solution)
            else:
                logging.debug(f"No solution found for maze {i + 1}.")

            # Visualize the solved maze (with the solution path highlighted)
            maze.set_solution(solution)
            maze.plot_maze(show_path=False, show_solution=True, show_position=False)

    except Exception as e:
        logging.error(f"An error occurred: {e}\n\nStack Trace:{traceback.format_exc()}")
        raise e


if __name__ == '__main__':
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    # logger.debug("Logging is configured.")

    backtracking_solver()
