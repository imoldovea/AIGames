import json
import logging
from configparser import ConfigParser

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

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

    def __init__(self, grid: np.ndarray, index) -> None:

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
        self.rows, self.cols = self.grid.shape  # Must be here before using self.rows

        self._context_map = None

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.visited_cells = set()
        self.animate = False
        self.save_movie = False
        self.raw_movie = []
        self.algorithm = None
        self.valid_solution = False
        self.complexity = self._compute_complexity()
        self.index = int(index)

        # cash
        self._bounds_cache = {}
        self._wall_cache = {}
        self._corridor_cache = {}

        self.grid = np.array(self.grid)  # Ensure grid is a NumPy array
        self._wall_table = (self.grid == self.WALL)  # Precompute wall positions as boolean array
        self._wall_table = self._compute_wall_table()
        self._valid_moves_cache = self.precompute_all_valid_moves()
        # Locate the starting position using the provided start_marker
        start_indices = np.where(self.grid == self.START)
        if len(start_indices[0]) == 0:
            self.logger.error("Starting marker %d not found in maze matrix.", self.START)
            raise ValueError(f"Starting marker not found in maze matrix.")
        self.start_position = (int(start_indices[0][0]), int(start_indices[1][0]))
        self.current_position = self.start_position

        # Replace the starting marker with a corridor (0)
        self.grid[self.start_position] = 0

        # Initialize the path with the starting position
        self.path = [self.start_position]
        self.move(self.start_position)

        # Exit can be defined later using set_exit()
        self.set_exit()
        self.self_test()


    def reset_solution(self):
        """
        Resets the solution state of an instance to its initial state.

        This method reinitializes or clears any relevant attributes or
        dependencies to return the solution or object to its default
        configuration. It ensures a clean state that enables further
        reuse or preparation for a new process.

        :raises AttributeError: If any required attribute is missing from
            the object during the reset process.
        :return: None
        """
        self.valid_solution = False
        self._solution = []  # To hold the maze solution as a list of coordinates

        # Replace the starting marker with a corridor (0)
        self.grid[self.start_position] = 0

        self.visited_cells = set()
        self.raw_movie = []

        self.path = [self.start_position]
        self.move(self.start_position)

    @property
    def context_map(self):
        if self._context_map is None:
            self._context_map = {
                (r, c): self._compute_local_context((r, c))
                for r in range(self.rows)
                for c in range(self.cols)
                if self.grid[r, c] == self.CORRIDOR
            }
        return self._context_map

    @property
    def height(self):
        return self.rows

    @property
    def width(self):
        return self.cols

    def clear_context_map(self):
        self._context_map = None

    def _compute_complexity(self):
        """
        Computes the maze complexity score based on:
          - Normalized solution path length
          - Maze area
          - Estimated number of loops

        Returns:
            float: Complexity score in the range [0, ~3+] (unbounded upper)
        """
        complexity_score = 1000
        # A. Path length (if available)
        if self._solution and isinstance(self._solution, list):
            path_length = len(self._solution)
        else:
            path_length = 0  # fallback for unsolved mazes

        # B. Maze area
        area = self.rows * self.cols

        # C. Loops = empty cells not on the path and not visited
        empty_cells = np.count_nonzero(self.grid == self.CORRIDOR)
        loop_estimate = max(0, empty_cells - path_length)

        # Normalize components to bring them into similar scale
        norm_path = path_length / (self.rows + self.cols)  # relative to perimeter
        norm_area = area / (18 * 18)  # relative to max maze size from config
        norm_loops = loop_estimate / 10  # empirical scaling

        complexity_score = round(norm_path + norm_area + norm_loops, 2)
        return complexity_score

    def _compute_local_context(self, position):
        """
        Optimized local context computation for a given position.
        Returns a list of 4 values (N, S, W, E) indicating wall/corridor.
        """
        r, c = position
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        context = []
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                context.append(self.grid[nr, nc])
            else:
                context.append(self.WALL)  # Treat out of bounds as wall
        return context

    def is_within_bounds(self, position):
        """
        Checks if a given position is within the defined bounds of the grid. The method
        utilizes a cache to store the result of previously checked positions for
        efficiency. The bounds are determined based on the grid's dimensions
        (rows and columns).

        :param position: The position to check, represented as an iterable of two elements
                         (row, column).
        :type position: Iterable[int]
        :return: True if the position is within bounds, False otherwise.
        :rtype: bool
        """
        if position is None:
            logging.debug("Position is None.")
            return False
        # Convert the position to a tuple to ensure it is hashable.
        key = tuple(position)
        if key not in self._bounds_cache:
            r, c = key
            self._bounds_cache[key] = 0 <= r < self.rows and 0 <= c < self.cols
        return self._bounds_cache[key]

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

    def is_wall(self, position):
        """Return True if the position is a wall."""
        if position is None or not hasattr(position, "__iter__") or len(position) != 2:
            return False

        x, y = position
        # Fast lookup in _wall_table
        if 0 <= x < self._wall_table.shape[0] and 0 <= y < self._wall_table.shape[1]:
            return self._wall_table[x, y]
        return False  # Out-of-bounds treated as not wall (optional: you could return True)

    def is_corridor(self, position):
        if not self.is_within_bounds(position):
            return False
        r, c = position
        return self.grid[r, c] == self.CORRIDOR

    def _compute_is_valid_move(self, position):
        """
        Compute the validity of a move without using the cache.

        Args:
            position (tuple): (x, y) coordinates of the position.

        Returns:
            bool: True if the move is valid, False otherwise.
        """
        # Inline check for None or invalid iterable
        if not position or len(position) != 2:
            return False

        x, y = position

        # Check within precomputed bounds
        if not (0 <= x < self.rows and 0 <= y < self.cols):
            return False

        # Use precomputed wall lookup
        return not self._wall_table[x, y]

    def _compute_wall_table(self):
        """
        Precompute a table representing wall positions for fast lookups.

        Returns:
            numpy.ndarray: A binary table of wall positions (1 if wall, 0 otherwise).
        """
        # NumPy operation to create a boolean table: True for walls, False otherwise
        return self.grid == self.WALL

    def reset_cache(self):
        """
        Clear the cache of valid moves.
        Use this method if the maze changes dynamically.
        """
        self._valid_moves_cache = {}

    def precompute_all_valid_moves(self):
        """
        Precompute valid moves for all positions in the maze and store them in the cache.
        """
        valid_moves_cache = {}
        for r in range(self.rows):
            for c in range(self.cols):
                position = (r, c)
                valid_moves_cache[position] = self._compute_is_valid_move(position)
        return valid_moves_cache

    def is_valid_move(self, position):
        """
        Check if a move is valid, using the cache if available.

        Args:
            position (tuple): (x, y) coordinates of the position.

        Returns:
            bool: True if the move is valid, False otherwise.
        """
        if position in self._valid_moves_cache:
            # Return cached result if it exists
            return self._valid_moves_cache[position]

        # Compute validity if not in cache
        is_valid = self._compute_is_valid_move(position)

        # Store the result in the cache
        self._valid_moves_cache[position] = is_valid

        return is_valid

    def can_move(self, current_position, move):
        """
        Checks if you can move in the given direction from current_position.
        """
        r, c = current_position
        dr, dc = move
        new_r, new_c = r + dr, c + dc
        new_position = (new_r, new_c)

        # Check both:
        if not self.is_within_bounds(new_position):
            return False
        if self.is_wall(new_position):
            return False
        if self.is_wall(current_position):
            return False
        return True

    def print_local_context(self, position):
        """
        Prints the local surroundings (north, south, east, west) around the given position
        to help debug why a move might be invalid.
        """
        r, c = position
        directions = {
            'north': (r - 1, c),
            'south': (r + 1, c),
            'east': (r, c + 1),
            'west': (r, c - 1),
        }

        info = []
        for dir_name, (nr, nc) in directions.items():
            if self.is_within_bounds((nr, nc)):
                cell = self.grid[nr, nc]
                meaning = "corridor" if cell == self.CORRIDOR else "wall"
                info.append(f"{dir_name.upper()}: {meaning} (value={cell})")
            else:
                info.append(f"{dir_name.upper()}: out of bounds")
        print(f"\nLocal context around {position}:")
        print("\n".join(info))

    def print_mini_map(self, position, size=1):
        """
        Prints a 3x3 (or larger) mini-map centered around the given position.

        Args:
            position (tuple): (row, column) of the agent.
            size (int): how far to extend the view from the center. size=1 means 3x3, size=2 means 5x5, etc.
        """
        r, c = position
        mini_map = []

        for dr in range(-size, size + 1):
            row = []
            for dc in range(-size, size + 1):
                nr, nc = r + dr, c + dc
                if (nr, nc) == position:
                    row.append('X')  # Current agent position
                elif not self.is_within_bounds((nr, nc)):
                    row.append('#')  # Out of bounds
                else:
                    if self.grid[nr, nc] == self.WALL:
                        row.append('1')  # Wall
                    else:
                        row.append('0')  # Corridor
            mini_map.append(' '.join(row))

        print("\nMini Map around position {}:".format(position))
        print('\n'.join(mini_map))

    def reset(self):
        """Reset maze to starting position for each new maze"""
        self.current_position = self.start_position
        self.path = [self.start_position]
        self.visited_cells = set()
        self.valid_solution = False
        self.raw_movie = []

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
            self.visited_cells.add(position)

            if backtrack:
                self.path.pop()

            if self.animate:
                self.plot_maze()

            if self.save_movie:
                # Create a snapshot that truly represents the current state.
                frame = self.get_maze_as_png(show_path=True, show_solution=False, show_position=False)
                self.raw_movie.append(frame)

            return True
        self.logger.debug("Invalid move attempted to position %s", position)
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

    def get_solution_summary(self):
        """
        Return a JSON representation of the maze with the current solution path.
        Only one frame showing the complete solution state.

        :return: JSON object containing the maze and solution information
        """

        # Store original state
        original_position = self.current_position
        original_path = self.path.copy()

        # Set current position to the end of solution (or start if no solution)
        if self._solution:
            self.current_position = self._solution[-1]  # End position
            self.path = self._solution  # Complete solution path
        else:
            self.current_position = self.start_position
            self.path = [self.start_position]

        solution_data = {
            "maze_index": self.index,
            "algorithm": self.algorithm,
            "grid": self.grid.tolist(),
            "solution": [list(pos) for pos in self._solution] if self._solution else [],
            "path": [list(pos) for pos in self.path],
            "current_position": list(self.current_position),
            "start_position": list(self.start_position),
            "exit": list(self.exit) if self.exit else None,
            "has_solution": bool(self._solution),
            "is_valid_solution": self.test_solution(),
            "solution_length": len(self._solution) if self._solution else 0,
            "maze_dimensions": {
                "width": self.width,
                "height": self.height
            },
            "complexity": self.complexity
        }

        # Restore original state
        self.current_position = original_position
        self.path = original_path

        return solution_data

    def get_solution_animation_data(self):
        """
        Calculate the step by step solution path through the maze
        :return: an array of json objects containing the maze and the solution steps.
        """

        steps = []

        # Store original state
        original_position = self.current_position
        original_path = self.path.copy()

        # Reset to initial state
        self.current_position = self.start_position
        self.path = [self.start_position]

        # If there's no solution, return only the initial frame
        if not self._solution:
            initial_step = {
                "step": 0,
                "position": list(self.current_position),
                "grid": self.grid.tolist(),
                "path": [list(pos) for pos in self.path],
                "solution": [],
                "start_position": list(self.start_position),
                "exit": list(self.exit) if self.exit else None,
                "is_complete": False,
                "algorithm": self.algorithm,
                "maze_index": self.index,
                "has_solution": False
            }
            steps.append(initial_step)

            # Restore original state
            self.current_position = original_position
            self.path = original_path

            return steps

        # Step 0: Initial maze state (just start position)
        initial_step = {
            "step": 0,
            "position": list(self.current_position),
            "grid": self.grid.tolist(),
            "path": [list(pos) for pos in self.path],
            "solution": [list(pos) for pos in self._solution],
            "start_position": list(self.start_position),
            "exit": list(self.exit) if self.exit else None,
            "is_complete": False,
            "algorithm": self.algorithm,
            "maze_index": self.index,
            "has_solution": True,
            "total_steps": len(self._solution)
        }
        steps.append(initial_step)

        # Generate steps for each position in the solution
        for i, position in enumerate(self._solution):
            self.current_position = position
            self.path = self._solution[:i + 1]

            step_data = {
                "step": i + 1,
                "position": list(self.current_position),
                "grid": self.grid.tolist(),
                "path": [list(pos) for pos in self.path],
                "solution": [list(pos) for pos in self._solution],
                "start_position": list(self.start_position),
                "exit": list(self.exit) if self.exit else None,
                "is_complete": self.current_position == self.exit,
                "algorithm": self.algorithm,
                "maze_index": self.index,
                "has_solution": True,
                "total_steps": len(self._solution)
            }
            steps.append(step_data)

        # Restore original state
        self.current_position = original_position
        self.path = original_path

        return steps

    def get_frames(self):
        """
        Instead of saving one frame at a time with self.raw_movie.append(self.get_maze_as_png(....))
        This method does use iterate solution one step at a time to create a list of frames. It will use the same
        self.get_maze_as_png(....) to transform a snapshot of the maze, and the current iteration in the solutions
        into a frame.

        Each frame will highlight the current position and the path taken. That means that the following frame will
        update the previous frame removing the current position highlight in the previous frame and adding it to the
         current frame with the current position highlight.
        :return: list of frames as png array. The first frame is the initial maze.
        """

        row_frames = []

        # Add the initial maze frame (showing the maze without solution or path)
        row_frames.append(self.get_maze_as_png(show_path=True, show_solution=False, show_position=False))

        # Store the original current position
        original_position = self.current_position

        # Iterate through the solution steps
        for i in range(len(self.path)):
            # Set the current position to the position at this step in the solution
            self.current_position = self.path[i]

            # Create a partial path up to the current position
            temp_path = self.path[:i + 1]
            original_path = self.path
            self.path = temp_path

            # Create a frame showing the path taken so far and highlighting the current position
            frame = self.get_maze_as_png(show_path=True, show_solution=False, show_position=True)

            # Add the frame to the list
            row_frames.append(frame)

            # Restore the original path
            self.path = original_path

        frame = self.get_maze_as_png(show_path=False, show_solution=True, show_position=True)
        # Restore the original current position
        self.current_position = original_position

        return row_frames

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

        self._solution = solution
        self.valid_solution = self.test_solution()
        if self.valid_solution:
            # recompute complexity score
            self.path = solution  # <- add this line to reuse the same path for visualization
            self._compute_complexity()
        else:
            self._solution = []
            self.valid_solution = False

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

        # Validate that the exit is connected to at least one corridor
        neighbors = self.get_neighbors(self.exit)
        if not any(self.is_corridor(neighbor) for neighbor in neighbors):
            self.logger.error("Exit is not connected to any corridor.")
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
                path_length = len(self.visited_cells)
                for (r, c) in self.visited_cells:
                    if 0 <= r < self.rows and 0 <= c < self.cols:
                        image_data[r, c] = [128, 128, 128]

                for idx, (r, c) in enumerate(self.path): #show active track
                    if 0 <= r < self.rows and 0 <= c < self.cols:
                        # Calculate the gradient color from green ([0, 255, 0]) to cyan ([0, 255, 255])
                        t = idx / (path_length - 1) if path_length > 1 else 0
                        color = [0, 255, int(255 * t)]
                        image_data[r, c] = np.clip(color, 0, 255)


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

    def print_ascii(self) -> None:
        """
        Prints the maze in ASCII format using 0, 1, and 3, where:
        - 0: Corridor
        - 1: Wall 
        - 3: Start position
        - X: current postion
        """

        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                if (r, c) == self.current_position:
                    row.append(' X ')
                elif (r, c) == self.start_position:
                    row.append(' 3 ')
                else:
                    row.append(f" {str(self.grid[r, c])} ")
                row.append(' ')
            print(''.join(row))

        # Print legend
        print("\nLegend:")
        print("0: corridor")
        print("1: wall")
        print("3: start point")
        print("X: current position")

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

        plt.title(f"{self.algorithm} - Maze Visualization: {self.index}\n{text}\nSolution steps: {len(self.path) - 1}",
                  color=text_color, pad=-60)
        plt.axis("off")
        plt.show()