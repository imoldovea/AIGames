import json
import logging
import traceback
from configparser import ConfigParser

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from utils import (load_mazes)

PARAMETERS_FILE = "config.properties"
config = ConfigParser()
config.read(PARAMETERS_FILE)
OUTPUT = config.get("FILES", "OUTPUT", fallback="output/")
INPUT = config.get("FILES", "INPUT", fallback="input/")

class Maze:
    WALL = 1
    CORRIDOR = 0
    START = 3
    IMG_SIZE = 26

    def __init__(self, grid: np.ndarray) -> None:

        """
            Initializes the maze from a provided NumPy matrix.

            Parameters:
              - grid: a NumPy array where walls are marked as 1, corridors as 0,
                      and the starting position is marked with start_marker (default 3).
              - start_marker: the marker used to denote the starting position in the maze.

            The constructor finds the starting marker, records its coordinates, replaces it with 0,
            and initializes the path with the starting position.
            """
        self._solution = [] # To hold the maze solution as a list of coordinates
        self.grid = np.array(grid, copy=True)
        self.rows, self.cols = self.grid.shape
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.visited_cells = []
        self.animate = False
        self.save_movie = False
        self.raw_movie = []
        self.algorithm = None
        self.valid_solution = False

        # Locate the starting position using the provided start_marker
        start_positions = np.argwhere(self.grid == self.START)
        if start_positions.size == 0:
            self.logger.error("Starting marker %d not found in maze matrix.", self.START)
            raise ValueError(f"Starting marker not found in maze matrix.")
        self.start_position = tuple(start_positions[0])
        self.current_position = self.start_position
        self.current_position = self.start_position

        # Replace the starting marker with a corridor (0)
        self.grid[self.start_position] = 0

        # Initialize the path with the starting position
        self.path = [self.start_position]
        #self.move(self.start_position)

        # Exit can be defined later using set_exit()
        self.set_exit()
        self.self_test()

    def set_exit(self):
        """
        Automatically sets the exit position as the first '0'
        encountered along the maze borders.
        """
        for r in range(self.rows):
            for c in range(self.cols):
                if (r == 0 or r == self.rows - 1 or c == 0 or c == self.cols - 1) and self.grid[r, c] == 0:
                    self.exit = (r, c)
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
        return self.grid[r, c] == self.WALL

    def is_corridor(self, position):
        if not self.is_within_bounds(position):
            return False
        r, c = position
        return self.grid[r, c] == self.CORRIDOR

    def is_valid_move(self, position):
        """
        Checks if moving to the given position is valid (within bounds and not hitting a wall).
        """
        return self.is_within_bounds(position) and not self.is_wall(position)

    def move(self, position: tuple[int, int], backtrack: bool = False) -> bool:

        """
            Moves to a new position if the move is valid.

            Updates the current position and records the move in the path.
            Returns True if the move was successful, False otherwise.
            """
        self.valid_solution = False #Assume that solving maze is still in progress
        if self.is_valid_move(position):
            self.current_position = position
            self.path.append(position)

            if backtrack:
                self.path.pop()
                if position not in self.visited_cells:
                    self.visited_cells.append(position)

            if self.animate:
                self.plot_maze()

            if self.save_movie:
                self.raw_movie.append(self.get_maze_as_png(show_path=True, show_solution=False, show_position=False))

            return True
        self.logger.warning("Invalid move attempted to position %s", position)
        return False

    def set_animate(self, value):
        """
        Setter for the 'animate' attribute.
    
        Args:
            value (bool): The new value for 'animate'.
    
        Raises:
            ValueError: If the value is not a boolean.
        """
        if not isinstance(value, bool):
            raise ValueError("The 'animate' attribute must be a boolean.")
        self.animate = value


    def get_raw_movie(self):
        """
        Getter for the 'raw_movie' attribute.
        Returns:
            list: The value of the 'raw_movie' attribute.
        """
        return self.raw_movie


    def set_save_movie(self, value):
        """
        Setter for the 'save_movie' attribute.
    
        Args:
            value (bool): The new value for 'save_movie'.
    
        Raises:
            ValueError: If the value is not a boolean.
        """
        if not isinstance(value, bool):
            raise ValueError("The 'save_movie' attribute must be a boolean.")
        self.save_movie = value

    def set_algorithm(self, algorithm):
        """
        Setter for the 'algorithm' attribute.
    
        Args:
            algorithm (str): The name of the algorithm to set.
    
        Raises:
            ValueError: If the algorithm name is not a string.
        """
        if not isinstance(algorithm, str):
            raise ValueError("Algorithm must be a string.")
        self.algorithm = algorithm

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

    def get_solution(self):
        """
        Getter method to retrieve the solution as a list of coordinates.
        """
        return self._solution

    def set_solution(self, solution):
        """
        Setter method to define the solution as a list of coordinates.
        """
        if not isinstance(solution, list):
            raise ValueError("Solution must be a list of coordinates.")
        # if not self.test_solution():
        #     logging.error("Solution is invalid. No solution set.")
        else:
            self._solution = solution
            self.valid_solution = self.test_solution()

    def self_test(self) -> bool:
        """
        Validates the maze configuration:
            - There is one and only one starting position.
            - There is one and only one exit.
            - The exit is on the perimeter of the maze and is accessible.

        Returns:
            bool: True if the maze configuration is valid, False otherwise.
        """
        # Validate minimum size of 5 for both width (cols) and height (rows)
        if self.rows < 5 or self.cols < 5:
            raise ValueError("Maze dimensions must be at least 5x5.")

        # Validate the start position
        if not self.is_within_bounds(self.start_position):
            self.logger.error("The start position is not within the bounds of the maze.")
            return False

        # Validate a single exit position
        exit_positions = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if self.grid[r, c] == self.CORRIDOR and (r == 0 or c == 0 or r == self.rows - 1 or c == self.cols - 1)
        ]
        if len(exit_positions) != 1:
            self.logger.error("Maze must have exactly one exit on the perimeter.")
            return False

        # Validate that the exit is connected to a border
        if self.exit is None or not self.is_within_bounds(self.exit):
            self.logger.error("Exit is not within the bounds of the maze.")
            return False

        r, c = self.exit
        if r != 0 and r != self.rows - 1 and c != 0 and c != self.cols - 1:
            self.logger.error("Exit is not connected to the maze border.")
            return False


        # Explicit log for successful validation
        return True

    def test_solution(self) -> bool:

        """
            Validates if the provided solution navigates through valid positions, starts at
            the start position, ends at the exit, and avoids walls.
            Returns:
                bool: True if the solution is valid, False otherwise.
            """
        if self.valid_solution:
            return self.valid_solution

        self.valid_solution = False
        if self._solution is None:
            self.logger.error("No solution provided.")
            self.valid_solution = False
            return False

        # Check if the solution length is greater than 1
        if len(self._solution) <= 1:
            self.logger.debug("Solution length is less than 2.")
            self.valid_solution = False
            return False

        if self._solution[0] != self.start_position:
            self.logger.debug("Solution does not start at the start position.")
            self.valid_solution = False
            return False

        if self.exit is None or self._solution[-1] != self.exit:
            self.logger.debug("Solution does not end at the exit position.")
            return False

        # Validate that the path is contiguous and avoids walls.
        for i in range(1, len(self._solution)):
            current, next_pos = self._solution[i - 1], self._solution[i]
            if next_pos not in self.get_neighbors(current):
                self.logger.error("Solution contains invalid moves between %s and %s.", current, next_pos)
                self.valid_solution = False
                return False
            if self.is_wall(next_pos):
                self.logger.error("Solution tries to move through a wall at %s.", next_pos)
                self.valid_solution = False
                return False

        self.valid_solution = True
        return True

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

    def get_maze_as_json(self) -> json:
        """
        Get the current maze configuration to a JSON file.

        The JSON file will include:
          - grid: the maze grid as a list of lists
          - path: the ordered sequence of path coordinates
          - start_position: the starting position of the maze
          - exit: the exit position (if set)
        """
        data = json.dumps({
            "grid": self.grid.tolist(),
            "solution": self._solution,
            "path": self.path,
            "start_position": self.start_position,
            "exit": self.exit
        })
        return data

    import numpy as np

    def create_padded_image(self, image_data, width=25, height=25):
        """
        Adds padding to the provided image data and overlays the solution validity.
    
        Args:
            image_data (np.ndarray): The maze image data as a NumPy array.
            width (int): The target width of the padded image.
            height (int): The target height of the padded image.
    
        Returns:
            np.ndarray: A new padded image with solution validity overlaid as text.
        """
        # Define the desired padding color (white in this case)
        padding_color = (255, 255, 255)

        # Initialize the resized image with the padding color
        resized_image = np.full((height, width, 3), padding_color, dtype=np.uint8)

        # Calculate starting indices to center the image_data
        start_row = (height - self.rows) // 2
        start_col = (width - self.cols) // 2

        # Place the image_data into the resized_image
        resized_image[start_row:start_row + self.rows, start_col:start_col + self.cols] = image_data

        # Convert the numpy array to a PIL Image for easier text drawing
        pil_image = Image.fromarray(resized_image)

        # Convert back to a numpy array
        resized_image = np.array(pil_image)

        return resized_image

    def get_maze_as_png(self, show_path=False, show_solution=True, show_position=False) -> np.ndarray:
        """
        Returns the current maze configuration as an RGB NumPy image.

        Parameters:
          - show_path: if True, the path taken is highlighted in red.

        Color scheme:
          - Walls: black
          - Corridors: white
          - Path: red (if show_path is True)
          - Start: green
          - Visited cells: gray
          - Exit: blue (if defined)

        Returns:
          An RGB image (as a NumPy array) that represents the maze.
        """
        # Create an RGB image based on the maze grid
        image_data = np.zeros((self.rows, self.cols, 3), dtype=np.uint8)
        corridors = (self.grid == 0)
        walls = (self.grid == 1)
        image_data[corridors] = [255, 255, 255]  # White for corridors
        image_data[walls] = [0, 0, 0]  # Black for walls

        # Optionally overlay the solution in red or green depending on validity
        if show_solution:
            solution_color = [0, 255, 0] if self.test_solution() else [255, 0, 0]  # Green if valid, Red if not
            for (r, c) in self._solution:
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    image_data[r, c] = solution_color

        if show_position:
            image_data[self.current_position] = [255, 192, 203] #Pink

        # Mark the start (green) and exit (blue, if defined)
        start_r, start_c = self.start_position
        image_data[start_r, start_c] = [255, 255, 0]  # Start in yellow
        if self.exit is not None:
            exit_r, exit_c = self.exit
            image_data[exit_r, exit_c] = [0, 255, 255]  # Start of gradient (green to cyan).
            if show_path:
                path_length = len(self.path)
                for idx, (r, c) in enumerate(self.path): #show active track
                    if 0 <= r < self.rows and 0 <= c < self.cols:
                        # Calculate the gradient color from green ([0, 255, 0]) to cyan ([0, 255, 255])
                        t = idx / (path_length - 1) if path_length > 1 else 0
                        color = [0, 255, int(255 * t)]
                        image_data[r, c] = color
                for (r, c) in self.visited_cells:  # show visited cells.
                    if 0 <= r < self.rows and 0 <= c < self.cols:
                        image_data[r, c] = [128, 128, 128]

        resized_image = self.create_padded_image(image_data, self.IMG_SIZE, self.IMG_SIZE)
        return resized_image

    def get_maze_as_text(self) -> str:
        """
        Returns the maze in an ASCII art-like format, with indicators for walls,
        corridors, the start position, and the exit.
    
        Representations:
          - "1": Wall
          - "0": Corridor
          - "X": Starting position
          - "O": Exit position
    
        Returns:
            str: The maze displayed as an ASCII matrix.
        """
        # Initialize a list for ASCII rows
        ascii_maze = []

        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                if (r, c) == self.start_position:
                    row.append('X')
                elif self.exit is not None and (r, c) == self.exit:
                    row.append('O')
                else:
                    row.append(str(self.grid[r, c]))
                row.append(' ')
            ascii_maze.append(''.join(row))

        # Join rows with newline to create final ASCII maze
        return '\n'.join(ascii_maze)

    def plot_maze(self, show_path=True, show_solution=True, show_position=False):
        """
        Plots the current maze configuration as a visual representation.
    
        Args:
            show_path (bool): If True, overlays the current path as a gradient on the maze.
            show_solution (bool): If True, highlights the solution path on the maze.
            show_position (bool): If True, marks the current position in the maze.
    
        Returns:
            None
        """
        image_data = self.get_maze_as_png(show_path=show_path, show_solution=show_solution,
                                           show_position=show_position)
        plt.imshow(image_data, interpolation='none')

        # Get the result of the solution test
        valid_solution = self.test_solution()

        # Determine the text color based on the test_solution result

        if valid_solution is True:
            text_color = "green"  # green
        elif valid_solution is False:
            text_color = "red"  # red
        else:
            text_color = "black"  # black

        # Prepare the overlay text
        text = f"Valid Solution = {valid_solution}"

        plt.title(f"{self.algorithm} - Maze Visualization\n{text}\nSolution steps: {len(self.path) - 1}",
                  color=text_color, pad=-60)
        plt.axis("off")
        plt.show()

def run_maze():
    """
    Test function that loads an array of mazes from 'input/mazes.npy',
    creates a Maze object using the first maze in the array, and displays it.
    """
    try:
        # Load mazes
        load_path = f"{INPUT}/mazes.pkl"
        mazes = load_mazes(load_path)
        # with open('input/mazes.pkl', 'rb') as f:
        #     mazes = pickle.load(f)
        # logging.info(f"Loaded {len(mazes)} mazes.")

        # Iterate through all the maze matrices and print each one
        for idx, maze in enumerate(mazes):
            # Create a Maze object from the first maze matrix
            maze_obj = Maze(maze)
            print(f"Maze {idx}:\n{maze_obj.get_maze_as_text()}")

            # Optionally, set an exit if you know where it should be (e.g., bottom right corner)
            #maze_obj.set_exit((maze_obj.rows - 2, maze_obj.cols - 2))
            # Display the first maze with the path overlay
            maze_obj.plot_maze(show_path=True)

            # Optionally, save the first maze as a PNG image
            #maze_obj.save_maze_as_png("output/test_maze.png", show_path=True)

    except Exception as e:
        logging.error(f"An error occurred: {e}\n\nStack Trace:{traceback.format_exc()}")


if __name__ == '__main__':
    run_maze()
