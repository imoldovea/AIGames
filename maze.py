import numpy as np
import json
import matplotlib.pyplot as plt
import logging

from h5py.h5o import visit


class Maze:
    WALL = 1
    CORRIDOR = 0

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
        self._solution = []  # To hold the maze solution as a list of coordinates
        self.grid = np.array(grid, copy=True)
        self.rows, self.cols = self.grid.shape
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.visited_cells = []
        self.animate = False
        self.save_movie = False
        self.raw_movie = []
        self.algorithm = None

        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.debug("Maze initialized with dimensions (%d, %d)", self.rows, self.cols)

        # Locate the starting position using the provided start_marker
        start_positions = np.argwhere(self.grid == start_marker)
        if start_positions.size == 0:
            self.logger.error("Starting marker %d not found in maze matrix.", start_marker)
            raise ValueError(f"Starting marker {start_marker} not found in maze matrix.")
        self.start_position = tuple(start_positions[0])
        self.current_position = self.start_position
        self.logger.debug("Starting position located at %s", self.start_position)
        self.current_position = self.start_position

        # Replace the starting marker with a corridor (0)
        self.grid[self.start_position] = 0
        self.logger.debug("Starting marker replaced with corridor at %s", self.start_position)

        # Initialize the path with the starting position
        self.path = [self.start_position]
        #self.move(self.start_position)

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
                    self.logger.debug("Exit automatically set at position %s", self.exit)
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


    def move(self, position, backtrack=False):
        """
        Moves to a new position if the move is valid.

        Updates the current position and records the move in the path.
        Returns True if the move was successful, False otherwise.
        """
        if self.is_valid_move(position):
            self.current_position = position
            self.path.append(position)
            self.logger.debug("Moved to position %s", position)

            if backtrack:
                logging.debug("Backtracking to position %s", position)
                self.path.pop()
                logging.debug("Backtrack complete. Current position is %s", self.current_position, " ", len(self.path))
                if position not in self.visited_cells:
                    self.visited_cells.append(position)

            if self.animate:
                self.plot_maze()

            if self.save_movie:
                self.raw_movie.append(self.get_maze_as_png(show_path=True, show_solution=False, show_position=False))

            return True
        self.logger.warning("Invalid move attempted to position %s", position)
        return False

    def get_animate(self) -> bool:
        """
        Getter for the 'animate' attribute.
        Returns:
            bool: The value of the 'animate' attribute.
        """
        return self.animate

    def get_save_movie(self)->bool:
        """
        Getter for the 'save_movie' attribute.
        Returns:
            bool: The value of the 'save_movie' attribute.
        """
        return self.save_movie

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

    def get_position(self):
        """
        Returns the current navigation position in the maze.
        """
        return self.current_position

    def get_algorithm(self):
        """
        Getter for the 'algorithm' attribute.
        Returns:
            str: The name of the algorithm in use or None if not set.
        """
        return self.algorithm

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
        self._solution = solution

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

    def get_maze_as_png(self, show_path=True, show_solution=True, show_position=False) -> np.ndarray:
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

        # Optionally overlay the solution in red
        if show_solution:
            for (r, c) in self._solution:
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    image_data[r, c] = [255, 0, 0]

        if show_position:
            image_data[self.current_position] = [255, 0, 0]

        # Mark the start (green) and exit (blue, if defined)
        start_r, start_c = self.start_position
        image_data[start_r, start_c] = [0, 255, 0]  # Start in green
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


        return image_data

    def get_maze_as_text(self) -> str:
        """
        Return the maze as an ASCII string, showing the maze as a matrix with:
        "1" - Walls
        "0" - Path
        "X" - Starting point
        "O" - Exit point
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
        Plots the current maze configuration on the screen.

        Parameters:
          - show_path: if True, the path taken is overlaid on the maze.
        """
        imgage_data = self.get_maze_as_png(show_path=show_path, show_solution=show_solution, show_position=show_position)
        plt.imshow(imgage_data)
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
            # Create a Maze object from the first maze matrix
            maze_obj = Maze(maze_matrix)
            print(f"Maze {idx}:\n{maze_obj.get_maze_as_text()}")

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
