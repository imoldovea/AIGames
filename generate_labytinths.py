# geberate_labytibths.py
import gc
import logging
import os
import pickle
import random
from configparser import ConfigParser

import matplotlib.pyplot as plt
import numpy as np
import tqdm

from classical_algorithms.backtrack_maze_solver import BacktrackingMazeSolver
from classical_algorithms.bfs_maze_solver import BFSMazeSolver
from classical_algorithms.grpah_maze_solver import AStarMazeSolver
from classical_algorithms.optimized_backtrack_maze_solver import OptimizedBacktrackingMazeSolver
from classical_algorithms.pladge_maze_solver import PledgeMazeSolver
from maze import Maze
from utils import setup_logging, profile_method

PATH = 0
WALL = 1
START = 3

OUTPUT_FOLDER = 'input'
PARAMETERS_FILE = "config.properties"
config = ConfigParser()
config.read(PARAMETERS_FILE)
OUTPUT = config.get("FILES", "OUTPUT", fallback="output/")
INPUT = config.get("FILES", "INPUT", fallback="input/")

# Mapping of available solver names to their classes
solver_mapping = {
    'BacktrackingMazeSolver': BacktrackingMazeSolver,
    'OptimizedBacktrackingMazeSolver': OptimizedBacktrackingMazeSolver,
    'PledgeMazeSolver': PledgeMazeSolver,
    'BFSMazeSolver': BFSMazeSolver,
    'AStarMazeSolver': AStarMazeSolver,
}

num_mazes = config.getint("MAZE", "num_mazes")


def create_maze(width: int, height: int) -> np.ndarray:
    """
    Generate a random rectangular maze with walls and paths.

    Args:
        width (int): Width of the maze in cells.
        height (int): Height of the maze in cells.

    Returns:
        numpy.ndarray: A 2D array representing the maze, where:
        1 is a wall, 0 is a path, and 2 marks the starting position.
    """
    if width < 3 or height < 3 or width % 2 == 0 or height % 2 == 0:
        raise ValueError("Maze dimensions must be odd and at least 3x3.")

    # Create a grid of cells with walls (1 = wall, 0 = path)
    maze = np.ones((height, width), dtype=np.int8)  # 1 = wall, 0 = path

    def carve(x, y):
        """
        Recursively carve paths through the maze by removing walls.

        Args:
            x (int): Current x-coordinate in the maze.
            y (int): Current y-coordinate in the maze.
        """
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
        random.shuffle(directions)  # Shuffle the order of movements (Right, Down, Left, Up)
        # to ensure that paths appear random and not biased.
        for dx, dy in directions:
            nx, ny = x + 2 * dx, y + 2 * dy  # Move two cells to carve
            if 0 < nx < width - 1 and 0 < ny < height - 1 and maze[ny, nx] == 1:
                maze[ny - dy, nx - dx] = 0  # Break wall
                maze[ny, nx] = PATH  # Carve path
                carve(nx, ny)

    # Carve maze starting from (1, 1)
    maze[1, 1] = PATH
    carve(1, 1)

    # Ensure all paths are interconnected
    ensure_all_paths_connected(maze)

    #set starting point
    maze = find_start(maze, width, height)

    # find exit
    maze = find_exit(maze, width, height)

    maze = add_loops(maze)

    return maze


def find_start(maze: np.ndarray, width: int, height: int) -> np.ndarray:
    # Set the starting point randomly so it is always on a valid path (0)
    path_cells = [(y, x) for y in range(height) for x in range(width) if maze[y, x] == 0]
    if path_cells:
        start_y, start_x = random.choice(path_cells)
        maze[start_y, start_x] = START  # Use `3` to mark the starting position

    return maze

def find_exit(maze, width, height):
    # Add a single exit (randomly choose any valid cell on the perimeter)
    perimeter_cells = []

    # Collect potential perimeter cells for the exit
    for x in range(width):
        if maze[1, x] == 0:  # Top edge
            perimeter_cells.append((0, x))
        if maze[height - 2, x] == 0:  # Bottom edge
            perimeter_cells.append((height - 1, x))
    for y in range(height):
        if maze[y, 1] == 0:  # Left edge
            perimeter_cells.append((y, 0))
        if maze[y, width - 2] == 0:  # Right edge
            perimeter_cells.append((y, width - 1))

    # Randomly select one perimeter cell as the exit
    if perimeter_cells:
        exit_y, exit_x = random.choice(perimeter_cells)
        maze[exit_y, exit_x] = 0

    return maze


def ensure_all_paths_connected(maze: np.ndarray) -> None:
    """
    Ensure all passable cells (0s) in the maze are part of the same connected component.
    This function modifies the maze in-place to guarantee connectivity.

    Args:
        maze (numpy.ndarray): 2D array representing the maze. Modified in-place.
    """
    height, width = maze.shape
    visited = np.zeros_like(maze, dtype=bool)
    maze_labels = np.zeros_like(maze, dtype=int)

    def flood_fill(x, y, label):
        stack = [(x, y)]  # Initialize a stack for iterative flood-fill traversal.
        # The stack stores coordinates of cells to visit.
        while stack:
            cx, cy = stack.pop()
            if 0 <= cx < width and 0 <= cy < height and not visited[cy, cx] and maze[cy, cx] == 0:
                visited[cy, cx] = True
                maze_labels[cy, cx] = label
                stack.extend([(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)])

    # Label connected components in the maze.
    # Each passable area is assigned a unique label using flood-fill.
    label = 1
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if maze[y, x] == 0 and not visited[y, x]:
                flood_fill(x, y, label)
                label += 1

    # Connect components.
    # This step ensures that previously disconnected areas are merged into a single connected component.
    # It modifies walls to break them and create paths between these areas.
    if label > 2:
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if maze_labels[y, x] > 1:  # If part of a disconnected component
                    maze[y, x] = PATH  # Connect to the main component


def add_loops(maze):
    """
    Introduce loops in a perfect maze by removing additional walls.

    Args:
        maze (numpy.ndarray): 2D array representing the maze structure.

    Returns:
        numpy.ndarray: Maze with additional loops.
    """
    height, width = maze.shape
    loop_probability = config.getfloat("MAZE", "loop_probability")
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            # Check if current cell is a wall that might separate two different paths.
            if maze[y, x] == 1:
                # Check horizontal wall candidates
                if (maze[y, x - 1] == 0 and maze[y, x + 1] == 0) and random.random() < loop_probability:
                    maze[y, x] = 0
                # Check vertical wall candidates
                elif (maze[y - 1, x] == 0 and maze[y + 1, x] == 0) and random.random() < loop_probability:
                    maze[y, x] = 0

    return maze


def save_mazes(folder, filename, mazes):
    if os.path.exists(os.path.join(OUTPUT_FOLDER, filename)):
        os.remove(os.path.join(OUTPUT_FOLDER, filename))
        logging.info(f"Deleted: {os.path.join(OUTPUT_FOLDER, filename)}")

    with open(folder + '/' + filename, 'wb') as f:
        pickle.dump(mazes, f)

def display_maze(maze):
    """
    Display the maze in ASCII format using characters.

    Args:
        maze (numpy.ndarray): 2D array representing the maze structure.
    """
    os.system('cls' if os.name == 'nt' else 'clear')
    ascii_maze = "\n".join("".join('█' if cell == 1 else ('S' if cell == 3 else ' ') for cell in row) for row in maze)
    print(ascii_maze)


def plot_maze(maze):
    """
    Plot the maze visually using Matplotlib.

    Args:
        maze (numpy.ndarray): 2D array representing the maze structure.
    """
    plt.imshow(maze, cmap='binary', origin='upper')
    plt.axis('off')  # Hides axes for better visualization
    plt.show()


@profile_method(output_file=f"generate_maze")
def generate(filename, number, solve=False):
    """
    Generate a specified number of mazes and optionally solve them.

    This function generates a set of maze instances with random dimensions within a specified range.
    Each maze can be solved using a solver defined in the configuration, and if solving is successful,
    the solution is tested before adding the maze to the output list. The generated mazes are then saved
    to an output folder.

    :param filename: str
        The name of the file where the mazes will be saved.

    :param number: int
        The number of mazes to generate.

    :param solve: bool, optional (default=False)
        A flag indicating whether to attempt solving each maze. If True,
        the function uses a solver specified in the configuration to find and
        test solutions for the generated mazes before saving them.

    :return: None

    """
    logging.info(f"Generating {filename}, {number} solved {solve} mazes...")

    min_size = config.getint("MAZE", "min_size")
    max_size = config.getint("MAZE", "max_size")
    mazes = []

    for i in tqdm.trange(number, desc="Generating mazes"):
        if i > 0 and i % 1000 == 0:
            gc.collect()
        maze = generate_single_maze(min_size, max_size, solve, i)
        if maze is not None:
            mazes.append(maze)

    save_mazes(OUTPUT_FOLDER, filename, mazes)
    logging.info(f"Saved {len(mazes)} mazes to {OUTPUT_FOLDER}/{filename}")


def generate_single_maze(min_size, max_size, solve, index):
    # Select random dimensions
    width = random.choice(range(min_size, max_size, 2))
    height = random.choice(range(min_size, max_size, 2))
    maze_array = create_maze(width, height)
    maze = Maze(maze_array, index)
    if not maze.self_test():
        logging.warning(f"Maze {index} is invalid")
        return None
    if solve:
        maze.animate = False
        maze.save_movie = False
        solver_obj = config.get("DEFAULT", "solver", fallback="BacktrackingMazeSolver")
        solver_class = solver_mapping.get(solver_obj)
        solver = solver_class(maze)
        try:
            solution = solver.solve()
            maze.set_solution(solution)
        except Exception as e:
            logging.error(f"An error occurred: {e}")
        if maze.test_solution():
            return maze
        else:
            # You may decide to discard mazes that fail the test.
            logging.warning(f"Maze {index} is invalid")
            return None
    else:
        return maze

def main():
    mazes = config.get("FILES", "MAZES", fallback="mazes.pkl")
    training_mazes = config.get("FILES", "TRAINING_MAZES", fallback="mazes.pkl")
    validation_mazes = config.get("FILES", "VALIDATION_MAZES", fallback="mazes.pkl")

    generate(filename=training_mazes, number=num_mazes, solve=True)
    generate(filename=validation_mazes, number=num_mazes // 10, solve=True)
    generate(filename=mazes, number=10, solve=False)

if __name__ == "__main__":
    #setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.debug("Logging is configured.")

    main()
