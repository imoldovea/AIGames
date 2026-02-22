import json
import logging
from configparser import ConfigParser
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

PARAMETERS_FILE = "config.properties"
config = ConfigParser()
config.read(PARAMETERS_FILE)
OUTPUT = config.get("FILES", "OUTPUT", fallback="output/")
INPUT = config.get("FILES", "INPUT", fallback="input/")


Pos = Tuple[int, int]


class Maze:
    WALL = 1
    CORRIDOR = 0
    START = 3
    IMG_SIZE = 26

    def __init__(self, grid: np.ndarray, index) -> None:
        """
        Initializes the maze from a provided NumPy matrix.

        grid: NumPy array with:
          - 1 = wall
          - 0 = corridor
          - 3 = start
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        self._solution: List[Pos] = []
        self.valid_solution: bool = False

        self.grid = np.array(grid, copy=True)
        self.rows, self.cols = self.grid.shape
        self.index = int(index)

        # Find start
        start_indices = np.where(self.grid == self.START)
        if len(start_indices[0]) == 0:
            self.logger.error("Starting marker %d not found in maze matrix.", self.START)
            raise ValueError("Starting marker not found in maze matrix.")

        self.start_position: Pos = (int(start_indices[0][0]), int(start_indices[1][0]))

        # Replace START marker with corridor
        self.grid[self.start_position] = self.CORRIDOR

        # Precompute wall table (fast lookups everywhere)
        self._wall_table: np.ndarray = (self.grid == self.WALL)

        # Precompute neighbors table (BIG speedup for all solvers)
        self._neighbors_table: Dict[Pos, List[Pos]] = self._precompute_neighbors_table()

        # Lazy local-context cache (useful for ML agent experiments)
        self._local_context_cache: Dict[Pos, List[int]] = {}
        self._context_map = None  # kept for compatibility; can be built from cache

        # Runtime / episode state
        self.current_position: Pos = self.start_position
        self.path: List[Pos] = [self.start_position]
        self.visited_cells = {self.start_position}

        self.animate: bool = False
        self.save_movie: bool = False
        self.raw_movie: List[np.ndarray] = []

        self.algorithm: Optional[str] = None

        # Exit can be defined later; default: first corridor on border
        self.exit: Optional[Pos] = None
        self.set_exit()

        # Complexity based on current _solution (empty at init)
        self.complexity: float = self._compute_complexity()

        self.self_test()

    # -------------------------
    # Reset / state management
    # -------------------------
    def reset_solution(self) -> None:
        """Reset solution + traversal state to initial position without duplicating start."""
        self.valid_solution = False
        self._solution = []

        self.current_position = self.start_position
        self.path = [self.start_position]
        self.visited_cells = {self.start_position}

        self.raw_movie = []

    def reset(self) -> None:
        """Alias for resetting traversal state (used by some experiments)."""
        self.reset_solution()

    # -------------------------
    # Neighbors / movement
    # -------------------------
    def _precompute_neighbors_table(self) -> Dict[Pos, List[Pos]]:
        table: Dict[Pos, List[Pos]] = {}
        # Cardinal directions only
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for r in range(self.rows):
            for c in range(self.cols):
                if self._wall_table[r, c]:
                    continue
                pos = (r, c)
                nbs: List[Pos] = []
                for dr, dc in dirs:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols and not self._wall_table[nr, nc]:
                        nbs.append((nr, nc))
                table[pos] = nbs
        return table

    def get_neighbors(self, position: Optional[Pos] = None) -> List[Pos]:
        """Fast neighbor lookup (precomputed)."""
        if position is None:
            position = self.current_position
        return self._neighbors_table.get(position, [])

    def is_wall(self, position: Optional[Pos]) -> bool:
        """Out-of-bounds is treated as wall (simplifies algorithms)."""
        if position is None:
            return True
        r, c = position
        if r < 0 or c < 0 or r >= self.rows or c >= self.cols:
            return True
        return bool(self._wall_table[r, c])

    def is_corridor(self, position: Optional[Pos]) -> bool:
        if position is None:
            return False
        r, c = position
        if r < 0 or c < 0 or r >= self.rows or c >= self.cols:
            return False
        return not bool(self._wall_table[r, c])

    def is_valid_move(self, position: Optional[Pos]) -> bool:
        """Valid move = inside bounds + not wall."""
        return not self.is_wall(position)

    def can_move(self, current_position: Pos, move: Pos) -> bool:
        """Checks if you can move in the given direction from current_position."""
        r, c = current_position
        dr, dc = move
        return self.is_valid_move((r + dr, c + dc)) and not self.is_wall(current_position)

    def move(self, position: Pos, backtrack: bool = False) -> bool:
        """
        Move to a new position if valid.

        backtrack=True keeps current_position update, but undoes the last path step
        (useful for some visualisations / algorithms).
        """
        self.valid_solution = False

        if not self.is_valid_move(position):
            self.logger.debug("Invalid move attempted to position %s", position)
            return False

        self.current_position = position
        self.visited_cells.add(position)

        # Maintain the path
        self.path.append(position)
        if backtrack and self.path:
            self.path.pop()

        # Rendering / movie capture (keep OFF during benchmarking)
        if self.animate:
            self.plot_maze()

        if self.save_movie:
            frame = self.get_maze_as_png(show_path=True, show_solution=False, show_position=False)
            self.raw_movie.append(frame)

        return True

    # -------------------------
    # Exit + validation
    # -------------------------
    def set_exit(self) -> None:
        """Automatically sets exit as the first corridor encountered on the border."""
        for r in range(self.rows):
            for c in range(self.cols):
                if (r == 0 or r == self.rows - 1 or c == 0 or c == self.cols - 1) and self.grid[r, c] == self.CORRIDOR:
                    self.exit = (r, c)
                    return
        self.logger.error("No valid exit found on the maze border.")
        raise ValueError("No valid exit found on the maze border.")

    def at_exit(self) -> bool:
        return self.exit is not None and self.current_position == self.exit

    def self_test(self) -> bool:
        # Minimum size
        if self.rows < 5 or self.cols < 5:
            raise ValueError("Maze dimensions must be at least 5x5.")

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

        # Exit should have at least one corridor neighbor
        if self.exit is None:
            self.logger.error("Exit is not set.")
            return False

        neighbors = self.get_neighbors(self.exit)
        if not any(self.is_corridor(nb) for nb in neighbors):
            self.logger.error("Exit is not connected to any corridor.")
            return False

        return True

    # -------------------------
    # Solution handling
    # -------------------------
    def get_solution(self) -> List[Pos]:
        return self._solution

    def set_solution(self, solution: List[Pos]) -> None:
        if not isinstance(solution, list):
            raise ValueError("Solution must be a list of coordinates.")

        self._solution = solution
        self.valid_solution = self.test_solution()

        if self.valid_solution:
            # reuse the same path for visualisation
            self.path = solution
            # FIX: actually store the recomputed value
            self.complexity = self._compute_complexity()
        else:
            self._solution = []
            self.valid_solution = False
            self.complexity = self._compute_complexity()

    def test_solution(self) -> bool:
        """Validate that solution starts at start, ends at exit, moves contiguously, avoids walls."""
        if self.valid_solution:
            return True

        if self._solution is None:
            self.valid_solution = False
            return False

        if len(self._solution) <= 1:
            self.valid_solution = False
            return False

        if self._solution[0] != self.start_position:
            self.valid_solution = False
            return False

        if self.exit is None or self._solution[-1] != self.exit:
            self.valid_solution = False
            return False

        for i in range(1, len(self._solution)):
            cur = self._solution[i - 1]
            nxt = self._solution[i]
            # neighbor check uses precomputed table
            if nxt not in self.get_neighbors(cur):
                self.valid_solution = False
                return False
            if self.is_wall(nxt):
                self.valid_solution = False
                return False

        self.valid_solution = True
        return True

    def _compute_complexity(self) -> float:
        """
        Complexity score based on:
          - Normalized solution path length
          - Maze area
          - Estimated number of loops (empty cells - path length)
        """
        if self._solution and isinstance(self._solution, list):
            path_length = len(self._solution)
        else:
            path_length = 0

        area = self.rows * self.cols
        empty_cells = int(np.count_nonzero(self.grid == self.CORRIDOR))
        loop_estimate = max(0, empty_cells - path_length)

        norm_path = path_length / (self.rows + self.cols)
        norm_area = area / (18 * 18)
        norm_loops = loop_estimate / 10

        return round(norm_path + norm_area + norm_loops, 2)

    # -------------------------
    # Local context (fast + lazy)
    # -------------------------
    def _compute_local_context(self, position: Pos) -> List[int]:
        """
        Local context for (r,c): [N,S,W,E] as WALL/CORRIDOR values.
        Out-of-bounds treated as WALL.
        """
        r, c = position
        ctx: List[int] = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                ctx.append(int(self.grid[nr, nc]))
            else:
                ctx.append(self.WALL)
        return ctx

    def get_local_context(self, position: Pos) -> List[int]:
        """Lazy cached local context (recommended for ML experiments)."""
        if position not in self._local_context_cache:
            self._local_context_cache[position] = self._compute_local_context(position)
        return self._local_context_cache[position]

    @property
    def context_map(self) -> Dict[Pos, List[int]]:
        """
        Compatibility property:
        Builds full map only when requested.
        Prefer get_local_context(pos) for speed.
        """
        if self._context_map is None:
            # compute only for corridors (including start which is corridor now)
            self._context_map = {
                (r, c): self.get_local_context((r, c))
                for r in range(self.rows)
                for c in range(self.cols)
                if self.grid[r, c] == self.CORRIDOR
            }
        return self._context_map

    def clear_context_map(self) -> None:
        self._context_map = None
        self._local_context_cache.clear()

    # -------------------------
    # Misc getters / setters
    # -------------------------
    @property
    def height(self) -> int:
        return self.rows

    @property
    def width(self) -> int:
        return self.cols

    def set_animate(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise ValueError("The 'animate' attribute must be a boolean.")
        self.animate = value

    def set_save_movie(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise ValueError("The 'save_movie' attribute must be a boolean.")
        self.save_movie = value

    def set_algorithm(self, algorithm: str) -> None:
        if not isinstance(algorithm, str):
            raise ValueError("Algorithm must be a string.")
        self.algorithm = algorithm

    def get_path(self) -> List[Pos]:
        return self.path

    def get_raw_movie(self) -> List[np.ndarray]:
        return self.raw_movie

    # -------------------------
    # Serialization / frames
    # -------------------------
    def get_maze_as_json(self) -> json:
        data = json.dumps({
            "grid": self.grid.tolist(),
            "solution": self._solution,
            "path": self.path,
            "start_position": self.start_position,
            "exit": self.exit
        })
        return data

    def create_padded_image(self, image_data: np.ndarray, width: int = 25, height: int = 25) -> np.ndarray:
        padding_color = (255, 255, 255)
        resized_image = np.full((height, width, 3), padding_color, dtype=np.uint8)

        start_row = (height - self.rows) // 2
        start_col = (width - self.cols) // 2
        resized_image[start_row:start_row + self.rows, start_col:start_col + self.cols] = image_data

        pil_image = Image.fromarray(resized_image)
        return np.array(pil_image)

    def get_maze_as_png(self, show_path: bool = True, show_solution: bool = True, show_position: bool = False) -> np.ndarray:
        image_data = np.zeros((self.rows, self.cols, 3), dtype=np.uint8)

        corridors = (self.grid == self.CORRIDOR)
        walls = (self.grid == self.WALL)
        image_data[corridors] = [255, 255, 255]
        image_data[walls] = [0, 0, 0]

        if show_solution:
            solution_color = [0, 255, 0] if self.test_solution() else [255, 0, 0]
            for (r, c) in self._solution:
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    image_data[r, c] = solution_color

        if show_position:
            r, c = self.current_position
            if 0 <= r < self.rows and 0 <= c < self.cols:
                image_data[r, c] = [255, 192, 203]  # pink

        if self.exit is not None:
            er, ec = self.exit
            image_data[er, ec] = [0, 255, 255]

            if show_path:
                for (r, c) in self.visited_cells:
                    if 0 <= r < self.rows and 0 <= c < self.cols:
                        image_data[r, c] = [128, 128, 128]

                path_length = max(1, len(self.path))
                for idx, (r, c) in enumerate(self.path):
                    if 0 <= r < self.rows and 0 <= c < self.cols:
                        t = idx / (path_length - 1) if path_length > 1 else 0
                        color = [0, 255, int(255 * t)]
                        image_data[r, c] = np.clip(color, 0, 255)

        # Force first cell in path to red (start)
        if self.path:
            sr, sc = self.path[0]
            image_data[sr, sc] = [255, 0, 0]

        return self.create_padded_image(image_data, self.IMG_SIZE, self.IMG_SIZE)

    def get_frames(self) -> List[np.ndarray]:
        frames: List[np.ndarray] = []
        frames.append(self.get_maze_as_png(show_path=True, show_solution=False, show_position=False))

        original_position = self.current_position
        original_path = self.path

        for i in range(len(self.path)):
            self.current_position = self.path[i]
            self.path = original_path[:i + 1]
            frames.append(self.get_maze_as_png(show_path=True, show_solution=False, show_position=True))

        # final frame with solution
        frames.append(self.get_maze_as_png(show_path=False, show_solution=True, show_position=True))

        self.current_position = original_position
        self.path = original_path
        return frames

    # -------------------------
    # Debug helpers (kept)
    # -------------------------
    def print_local_context(self, position: Pos) -> None:
        r, c = position
        directions = {
            "north": (r - 1, c),
            "south": (r + 1, c),
            "east": (r, c + 1),
            "west": (r, c - 1),
        }

        info = []
        for name, (nr, nc) in directions.items():
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                cell = self.grid[nr, nc]
                meaning = "corridor" if cell == self.CORRIDOR else "wall"
                info.append(f"{name.upper()}: {meaning} (value={cell})")
            else:
                info.append(f"{name.upper()}: out of bounds")
        print(f"\nLocal context around {position}:")
        print("\n".join(info))

    def print_mini_map(self, position: Pos, size: int = 1) -> None:
        r, c = position
        mini_map = []

        for dr in range(-size, size + 1):
            row = []
            for dc in range(-size, size + 1):
                nr, nc = r + dr, c + dc
                if (nr, nc) == position:
                    row.append("X")
                elif nr < 0 or nc < 0 or nr >= self.rows or nc >= self.cols:
                    row.append("#")
                else:
                    row.append("1" if self.grid[nr, nc] == self.WALL else "0")
            mini_map.append(" ".join(row))

        print(f"\nMini Map around position {position}:")
        print("\n".join(mini_map))

    # NOTE: plot_maze() is referenced by move() when animate=True.
    # Your original class had it elsewhere; keep your existing plot_maze implementation.
    def plot_maze(self, *args, **kwargs):
        raise NotImplementedError("plot_maze() should remain as in your original implementation.")


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
