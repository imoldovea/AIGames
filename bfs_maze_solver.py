from numpy.f2py.auxfuncs import throw_error
from maze_solver import MazeSolver
from maze import Maze
from collections import deque
import traceback
import logging
import pickle

logging.basicConfig(level=logging.INFO)


class BFSMazeSolver(MazeSolver):
    def __init__(self, maze):
        """
        Initializes the BFSMazeSolver with a Maze object.
        Args:
            maze (Maze): The maze to solve.
        """
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
                    #self.maze.move(current)
                path.reverse()
                #self.maze.move(current, backtrack=True)
                # Optionally update the maze's path with the found solution.
                self.maze.path = path
                return path

            # Explore the neighbors.
            for neighbor in self.maze.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current
                    queue.append(neighbor)
                    self.maze.move(current)

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
        # Load mazes
        with open('input/mazes.pkl', 'rb') as f:
            mazes = pickle.load(f)
        logging.info(f"Loaded {len(mazes)} mazes.")

        # Iterate through each maze in the array
        for i, maze_matrix in enumerate(mazes):
            maze = Maze(maze_matrix)

            logging.debug(f"Solving maze {i + 1}...")

            # Create a Maze object from the maze matrix
            maze.set_animate(False)
            maze.set_save_movie(False)

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
        logging.info("All mazes solved.")
    except Exception as e:
        logging.error(f"An error occurred: {e}\n\nStack Trace:{traceback.format_exc()}")
        throw_error(e)


if __name__ == '__main__':
    test_bfs_solver()
