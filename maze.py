import numpy as np
import json
import matplotlib.pyplot as plt
import logging


class Maze:
    def __init__(self, grid, start_marker=3):
        """
        Initializes the maze from a provided NumPy matrix.

        Parameters:
          - grid: a NumPy array where walls are marked as 1, corridors as 0,
                  and the starting position is marked with start_marker (default 3).
          - start_marker: the marker used to denote the starting position in the maze.

        The constructor finds the starting marker, records its coordinates, replaces it with 0,
        and initializes the path with the starting position.
        """
        self.grid = np.array(grid, copy=True)
        self.rows, self.cols = self.grid.shape
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.info("Maze initialized with dimensions (%d, %d)", self.rows, self.cols)

        # Locate the starting position using the provided start_marker
        start_positions = np.argwhere(self.grid == start_marker)
        if start_positions.size == 0:
            self.logger.error("Starting marker %d not found in maze matrix.", start_marker)
            raise ValueError(f"Starting marker {start_marker} not found in maze matrix.")
        self.start_position = tuple(start_positions[0])
        self.logger.info("Starting position located at %s", self.start_position)
        self.current_position = self.start_position

        # Replace the starting marker with a corridor (0)
        self.grid[self.start_position] = 0
        self.logger.info("Starting marker replaced with corridor at %s", self.start_position)

        # Initialize the path with the starting position
        self.path = [self.start_position]

        # Exit can be defined later using set_exit()
        self.set_exit()

    def set_exit(self):
        """
        Automatically sets the exit position as the first '0'
        encountered along the maze borders.
        """
        for r in range(self.rows):
            for c in range(self.cols):
                if (r == 0 or r == self.rows - 1 or c == 0 or c == self.cols - 1) and self.grid[r, c] == 0:
                    self.exit = (r, c)
                    self.logger.info("Exit automatically set at position %s", self.exit)
                    return
        self.logger.error("No valid exit found on the maze border.")
        raise ValueError("No valid exit found on the maze border.")

    def is_within_bounds(self, position):
        r, c = position
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_wall(self, position):
        if not self.is_within_bounds(position):
            return True  # Out-of-bounds is treated as a wall.
        r, c = position
        return self.grid[r, c] == 1

    def is_corridor(self, position):
        if not self.is_within_bounds(position):
            return False
        r, c = position
        return self.grid[r, c] == 0

    def is_valid_move(self, position):
        """
        Checks if moving to the given position is valid (within bounds and not hitting a wall).
        """
        return self.is_within_bounds(position) and not self.is_wall(position)

    def move(self, position):
        """
        Moves to a new position if the move is valid.

        Updates the current position and records the move in the path.
        Returns True if the move was successful, False otherwise.
        """
        if self.is_valid_move(position):
            self.current_position = position
            self.path.append(position)
            self.logger.info("Moved to position %s", position)
            return True
        self.logger.warning("Invalid move attempted to position %s", position)
        return False

    def get_neighbors(self, position=None):
        """
        Returns a list of valid neighboring positions (up, down, left, right) from the given position.
        If no position is provided, the current position is used.
        """
        if position is None:
            position = self.current_position
        r, c = position
        possible_moves = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        return [pos for pos in possible_moves if self.is_valid_move(pos)]


    def at_exit(self):
        """
        Checks if the current position is the exit.
        """
        if self.exit is None:
            return False
        return self.current_position == self.exit

    def get_path(self):
        """
        Returns the list of recorded path coordinates.

        The path includes all visited positions, starting from the starting position.
        """
        return self.path

    def save_maze_as_json(self, filename):
        """
        Saves the current maze configuration to a JSON file.
    
        The JSON file will include:
          - grid: the maze grid as a list of lists
          - path: the ordered sequence of path coordinates
          - start_position: the starting position of the maze
          - exit: the exit position (if set)
        """
        data = {
            "grid": self.grid.tolist(),
            "path": self.path,
            "start_position": self.start_position,
            "exit": self.exit
        }
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

    def save_maze_as_png(self, filename, show_path=True):
        """
        Saves the current maze configuration as a PNG image.

        Parameters:
          - filename: the name of the output PNG file.
          - show_path: if True, the path taken is overlaid on the maze.

        Color scheme:
          - Walls: black
          - Corridors: white
          - Path: red (if show_path is True)
          - Start: green
          - Exit: blue (if defined)
        """
        # Create an RGB image based on the maze grid
        img = np.zeros((self.rows, self.cols, 3), dtype=np.uint8)
        corridors = (self.grid == 0)
        walls = (self.grid == 1)
        img[corridors] = [255, 255, 255]  # White for corridors
        img[walls] = [0, 0, 0]  # Black for walls

        # Optionally overlay the path in red
        if show_path:
            for (r, c) in self.path:
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    img[r, c] = [255, 0, 0]

        # Mark the start in green and the exit in blue (if defined)
        start_r, start_c = self.start_position
        img[start_r, start_c] = [0, 255, 0]  # Start marker in green
        if self.exit is not None:
            exit_r, exit_c = self.exit
            img[exit_r, exit_c] = [0, 0, 255]  # Exit marker in blue

        plt.imsave(filename, img)

    def plot_maze(self, show_path=True):
        """
        Plots the current maze configuration on the screen.

        Parameters:
          - show_path: if True, the path taken is overlaid on the maze.

        Color scheme is the same as in save_maze_as_png.
        """
        # Create an RGB image based on the maze grid
        img = np.zeros((self.rows, self.cols, 3), dtype=np.uint8)
        corridors = (self.grid == 0)
        walls = (self.grid == 1)
        img[corridors] = [255, 255, 255]  # White for corridors
        img[walls] = [0, 0, 0]  # Black for walls

        # Optionally overlay the path in red
        if show_path:
            for (r, c) in self.path:
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    img[r, c] = [255, 0, 0]

        # Mark the start in green and the exit in blue (if defined)
        start_r, start_c = self.start_position
        img[start_r, start_c] = [0, 255, 0]
        if self.exit is not None:
            exit_r, exit_c = self.exit
            img[exit_r, exit_c] = [0, 0, 255]

        plt.imshow(img)
        plt.title("Maze Visualization")
        plt.axis("off")
        plt.show()


def test_maze():
    """
    Test function that loads an array of mazes from 'input/mazes.npy',
    creates a Maze object using the first maze in the array, and displays it.
    """
    try:
        # Load the numpy file containing an array of mazes
        maze_array = np.load("input/mazes.npy", allow_pickle=True)

        # Check if maze_array is a multi-dimensional array of mazes
        if maze_array.ndim == 3:
            maze_matrix = maze_array[0]
        else:
            # Otherwise, assume it's a single maze or a list-like of mazes
            maze_matrix = maze_array[0] if isinstance(maze_array, (list, np.ndarray)) else maze_array

        # Iterate through all the maze matrices and print each one
        for idx, maze_matrix in enumerate(maze_array):
            print(f"Maze {idx}:\n{maze_matrix}")
            # Create a Maze object from the first maze matrix
            maze_obj = Maze(maze_matrix)

            # Optionally, set an exit if you know where it should be (e.g., bottom right corner)
            #maze_obj.set_exit((maze_obj.rows - 2, maze_obj.cols - 2))
            # Display the first maze with the path overlay
            maze_obj.plot_maze(show_path=True)

            # Optionally, save the first maze as a PNG image
            #maze_obj.save_maze_as_png("output/test_maze.png", show_path=True)

    except Exception as e:
        logging.error("An error occurred: %s", e)


if __name__ == '__main__':
    test_maze()
