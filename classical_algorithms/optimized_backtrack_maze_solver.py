import logging
import pickle
import traceback
from typing import List, Tuple, Dict, Optional

import numpy as np

from maze import Maze
from utils import setup_logging

logger = logging.getLogger(__name__)


class OptimizedBacktrackingMazeSolver:
    """A maze solver that uses a vectorized backtracking algorithm to find a path."""

    def __init__(self, maze):
        """Initialize the solver with a maze.

        Args:
            maze: The maze to solve.
        """
        self.maze = maze
        maze.set_algorithm(self.__class__.__name__)

        # Pre-compute and cache valid neighbors for each cell
        self._neighbors_cache = self._precompute_neighbors()

        # Create numpy grid representation for vectorized operations
        self._grid = np.zeros((maze.rows, maze.cols), dtype=np.int8)
        for r in range(maze.rows):
            for c in range(maze.cols):
                if maze.is_wall((r, c)):
                    self._grid[r, c] = 1

    def solve(self) -> Optional[List[Tuple[int, int]]]:
        """Solve the maze using vectorized backtracking.

        Returns:
            A list of positions from start to exit, or None if no path exists.
        """
        if self.maze.exit is None:
            logger.warning("Maze exit is not set.")
            return None

        start_position = self.maze.start_position
        exit_position = self.maze.exit

        if start_position == exit_position:
            logger.info("Start and exit positions are the same.")
            return [start_position]

        # Use numpy array for visited cells
        visited = np.zeros((self.maze.rows, self.maze.cols), dtype=bool)

        # Track the path
        path = []

        # Start DFS with vectorized operations

        # logger.debug(f"Starting backtracking from {start_position}")
        success = self._dfs(start_position, exit_position, visited, path)

        if success:
            self.maze.path = path
            #logger.debug(f"Found solution path with {len(path)} steps")
            return path

        logger.info("No path found")
        return None

    def _precompute_neighbors(self) -> Dict[Tuple[int, int], np.ndarray]:
        """Precompute valid neighbors for each cell for faster lookups.

        Returns:
            A dictionary mapping positions to arrays of valid neighbor positions.
        """
        cache = {}
        directions = np.array([(0, 1), (1, 0), (0, -1), (-1, 0)])  # right, down, left, up

        for r in range(self.maze.rows):
            for c in range(self.maze.cols):
                if self.maze.is_wall((r, c)):
                    continue

                # Use vectorized operations to calculate all neighbors at once
                neighbors = np.array([r, c]) + directions

                # Filter valid neighbors
                valid_neighbors = []
                for nr, nc in neighbors:
                    if (0 <= nr < self.maze.rows and
                            0 <= nc < self.maze.cols and
                            not self.maze.is_wall((nr, nc))):
                        valid_neighbors.append((nr, nc))

                cache[(r, c)] = np.array(valid_neighbors)

        #logger.debug(f"Precomputed neighbors for {len(cache)} positions")
        return cache

    def _get_cached_neighbors(self, position: Tuple[int, int]) -> np.ndarray:
        """Get cached valid neighbors for a position.

        Args:
            position: The position to get neighbors for.

        Returns:
            An array of valid neighbor positions.
        """
        # Use cached neighbors if available, otherwise return empty array
        return self._neighbors_cache.get(position, np.array([]))

    def _dfs(self, current: Tuple[int, int], target: Tuple[int, int],
             visited: np.ndarray, path: List[Tuple[int, int]]) -> bool:
        """Depth-first search with vectorized operations.

        Args:
            current: Current position.
            target: Target position (exit).
            visited: NumPy array of visited positions.
            path: Current path being built.

        Returns:
            True if a path to the target was found, False otherwise.
        """
        # Add current position to path
        path.append(current)

        # Mark as visited
        r, c = current
        visited[r, c] = True

        #logger.debug(f"Visiting {current}, path length: {len(path)}")

        # Check if we reached the target
        if current == target:
            #logger.debug(f"Found target at {current}")
            return True

        # Update the maze's current position for visualization
        self.maze.move(current)

        # Get cached neighbors using vectorized arrays
        neighbors = self._get_cached_neighbors(current)

        # Evaluate all neighbors at once when possible
        if len(neighbors) > 0:
            # Sort neighbors by distance to target for potential optimization
            # This uses the Manhattan distance heuristic
            if len(neighbors) > 1:
                tr, tc = target
                distances = np.abs(neighbors[:, 0] - tr) + np.abs(neighbors[:, 1] - tc)
                sorted_indices = np.argsort(distances)
                neighbors = neighbors[sorted_indices]

            # Check each neighbor
            for neighbor in neighbors:
                nr, nc = neighbor

                # Skip if already visited
                if visited[nr, nc]:
                    continue

                # Recursive DFS
                if self._dfs((nr, nc), target, visited, path):
                    return True

        # Backtrack if no solution found via this path
        path.pop()
        #logger.debug(f"Backtracking from {current}, path length: {len(path)}")
        return False


# Example test function
def solver() -> None:
    """
    Test function that loads an array of mazes from 'input/mazes.npy',
    creates a Maze object using the first maze in the array, sets an exit,
    solves the maze using the BacktrackingMazeSolver, and displays the solution.
    """
    try:
        # Load the numpy file containing an array of mazes
        with open('../input/mazes.pkl', 'rb') as f:
            mazes = pickle.load(f)
        logging.info(f"Loaded {len(mazes)} mazes.")

        # Iterate through each maze in the array
        for i, maze_matrix in enumerate(mazes):
            maze = Maze(maze_matrix)
            #logging.debug(f"Solving maze {i + 1}...")

            # Instantiate the backtracking maze solver
            solver = OptimizedBacktrackingMazeSolver(maze)
            solution = solver.solve()

            if solution:
                logging.debug(f"Maze {i + 1} solution")
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
    #logger.debug("Logging is configured.")

    solver()
