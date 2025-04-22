# grpah_maze_solver.py

import heapq
import logging
import pickle
import traceback
from typing import List, Tuple, Dict, Optional

import numpy as np

from maze import Maze
from maze_solver import MazeSolver
from utils import setup_logging


class AStarMazeSolver(MazeSolver):
    """A maze solver that uses the A* algorithm for efficient pathfinding."""

    def __init__(self, maze):
        """Initialize the solver with a maze.

        Args:
            maze: The maze to solve.
        """
        self.maze = maze
        maze.set_algorithm(self.__class__.__name__)

        # Pre-compute and cache valid neighbors for each cell
        self._neighbors_cache = self._precompute_neighbors()

        # Create a numpy grid representation
        self._grid = np.zeros((maze.rows, maze.cols), dtype=np.int8)
        for r in range(maze.rows):
            for c in range(maze.cols):
                if maze.is_wall((r, c)):
                    self._grid[r, c] = 1

    def solve(self) -> Optional[List[Tuple[int, int]]]:
        """Solve the maze using the A* algorithm.

        Returns:
            A list of positions from start to exit, or None if no path exists.
        """
        if self.maze.exit is None:
            logging.warning("Maze exit is not set.")
            return None

        start_position = self.maze.start_position
        exit_position = self.maze.exit

        if start_position == exit_position:
            logging.info("Start and exit positions are the same.")
            return [start_position]

        # Initialize the open set with start position
        open_set = []

        # Use counter to break ties in priority queue consistently
        counter = 0

        # f_score = g_score + heuristic
        # g_score = cost from start to current
        # Push (f_score, counter, position) to priority queue
        heapq.heappush(open_set, (0, counter, start_position))
        counter += 1

        # Dictionary to track where a node came from for path reconstruction
        came_from = {}

        # Cost from start to each node
        g_score = {start_position: 0}

        # Use a numpy array for closed set (visited nodes)
        closed_set = np.zeros((self.maze.rows, self.maze.cols), dtype=bool)

        logging.debug(f"Starting A* search from {start_position}")

        while open_set:
            # Get node with lowest f_score
            _, _, current = heapq.heappop(open_set)
            # Update current position in the visualization
            self.maze.move(current)

            # Check if we've reached the exit
            if current == exit_position:
                logging.debug(f"Found path to exit at {current}")
                path = self._reconstruct_path(came_from, current)
                self.maze.path = path
                return path

            # Add to closed set
            r, c = current
            closed_set[r, c] = True

            # Get cached neighbors
            neighbors = self._get_cached_neighbors(current)

            # Process each neighbor
            for neighbor in neighbors:
                nr, nc = neighbor

                # Skip if already in closed set
                if closed_set[nr, nc]:
                    continue

                # Calculate g_score for this neighbor
                tentative_g_score = g_score[current] + 1

                # If this node is already in open set with better g_score, skip it
                if neighbor in g_score and tentative_g_score >= g_score[neighbor]:
                    continue

                # This is a better path to this neighbor
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score

                # Calculate f_score with Manhattan distance heuristic
                f_score = tentative_g_score + self._heuristic(neighbor, exit_position)

                # Add to open set
                heapq.heappush(open_set, (f_score, counter, neighbor))
                counter += 1

        logging.info("No path found")
        return None

    def _precompute_neighbors(self) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        """Precompute valid neighbors for each cell for faster lookups.

        Returns:
            A dictionary mapping positions to lists of valid neighbor positions.
        """
        cache = {}
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up

        for r in range(self.maze.rows):
            for c in range(self.maze.cols):
                if self.maze.is_wall((r, c)):
                    continue

                # Calculate neighbors
                valid_neighbors = []
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < self.maze.rows and
                            0 <= nc < self.maze.cols and
                            not self.maze.is_wall((nr, nc))):
                        valid_neighbors.append((nr, nc))

                cache[(r, c)] = valid_neighbors

        logging.debug(f"Precomputed neighbors for {len(cache)} positions")
        return cache

    def _get_cached_neighbors(self, position: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get cached valid neighbors for a position.

        Args:
            position: The position to get neighbors for.

        Returns:
            A list of valid neighbor positions as tuples.
        """
        # Use cached neighbors if available, otherwise return empty list
        return self._neighbors_cache.get(position, [])

    def _heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance heuristic between two positions.

        Args:
            pos1: First position.
            pos2: Second position.

        Returns:
            The Manhattan distance between the positions.
        """
        # Calculate Manhattan distance directly
        r1, c1 = pos1
        r2, c2 = pos2
        return abs(r1 - r2) + abs(c1 - c2)

    def _reconstruct_path(self, came_from: Dict[Tuple[int, int], Tuple[int, int]],
                          current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct the path from start to current position.

        Args:
            came_from: Dictionary mapping positions to their predecessor.
            current: The current (end) position.

        Returns:
            List of positions from start to end.
        """
        total_path = [current]

        while current in came_from:
            current = came_from[current]
            total_path.append(current)

        # Reverse to get path from start to end
        total_path.reverse()

        logging.debug(f"Reconstructed path with {len(total_path)} steps")
        return total_path


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
            logging.debug(f"Solving maze {i + 1}...")

            # Instantiate the backtracking maze solver
            solver = AStarMazeSolver(maze)
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

    solver()
