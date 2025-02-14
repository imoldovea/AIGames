from maze_solver import MazeSolver
from maze import Maze
from collections import deque
import logging
import cProfile

logging.basicConfig(level=logging.INFO)


class BFSMazeSolver(MazeSolver):
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
        exit = self.maze.exit

        # Queue for BFS and dictionary to store the parent of each visited position.
        queue = deque([start])
        visited = {start}
        parent = {start: None}

        # Perform BFS until the queue is empty or the exit is found.
        while queue:
            current = queue.popleft()

            if current == exit:
                # Exit found, reconstruct the path from start to exit.
                path = []
                while current is not None:
                    path.append(current)
                    current = parent[current]
                path.reverse()
                # Optionally update the maze's path with the found solution.
                self.maze.path = path
                return path

            # Explore the neighbors.
            for neighbor in self.maze.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current
                    queue.append(neighbor)

        # No solution found
        return None


# Example test function for the BFS solver
def test_bfs_solver():
    """
    Test function that loads an array of mazes from 'input/mazes.npy',
    creates a Maze object using the first maze in the array, sets an exit,
    solves the maze using the BFSMazeSolver, and displays the solution.
    """
    try:
        import numpy as np
        # Load mazes from a NumPy file
        maze_array = np.load("input/mazes.npy", allow_pickle=True)

        # Iterate through each maze in the array
        for i, maze_matrix in enumerate(maze_array):
            logging.debug(f"Solving maze {i + 1} with BFS...")
            maze_obj = Maze(maze_matrix)
            solver = BFSMazeSolver(maze_obj)
            solution = solver.solve()

            if solution:
                logging.debug(f"Maze {i + 1} solution found:")
                logging.debug(solution)
            else:
                logging.debug(f"No solution found for maze {i + 1}.")

            # Visualize the solved maze (with the solution path highlighted)
            maze_obj.set_solution(solution)
            maze_obj.plot_maze(show_path=True, show_solution=False)
    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == '__main__':
    test_bfs_solver()
