#  Python
# llm_maze_solve.py
import configparser
import logging

import numpy as np

from llm.gpt_factory import GPTFactory
from maze_solver import MazeSolver
from utils import setup_logging, load_mazes, save_mazes_as_pdf, clean_output_folder, save_movie_v2, save_movie

# --- Centralized Direction Definitions ---
# Coordinate tuples (row_delta, col_delta)
NORTH = (-1, 0)
SOUTH = (1, 0)
EAST = (0, 1)
WEST = (0, -1)

# Consistent order for iteration if needed (e.g., N, S, E, W)
DIRECTIONS_ORDERED = [NORTH, SOUTH, EAST, WEST]

# Map coordinate tuples to string names
DIRECTION_NAMES = {
    NORTH: "north",
    SOUTH: "south",
    EAST: "east",
    WEST: "west"
}

# Map string names back to coordinate tuples (for processing LLM response)
ACTION_TO_DIRECTION = {
    "north": NORTH,
    "south": SOUTH,
    "east": EAST,
    "west": WEST
}
# --- End of Centralized Definitions ---

# Mental Map Cell States
WALL = '1'  # Wall cell
PATH = '2'  # Previously visited path
START = '3'
CURRENT = 'X'  # Current position
CORRIDOR = '0'  # Unexplored corridor
UNKNOWN = '-'  # Yet unknown cells

MAX_RETRIES_PER_STEP = 2

PARAMETERS_FILE = "config.properties"
SECRETS_FILE = "secrets.properties"

config = configparser.ConfigParser()
config.read(PARAMETERS_FILE)
OUTPUT = config.get("FILES", "OUTPUT", fallback="output/")
INPUT = config.get("FILES", "INPUT", fallback="input/")

SYSTEM_PROMPT = ("""
You are expected to find the exit form a maze using Backtracking algorithm. You are inside the maze and see only your immediate surroundings (north, south, east, west). 

Each step you'll receive:
- What's around you ("open", "wall", or "outside").
- Valid moves (directions that lead to unexplored or open spaces).
- History of moves taken (oldest to newest).
- A memory map of positions visited.

Rules:
- NEVER immediately reverse your previous move unless there is no other valid choice.
- Always prioritize unexplored paths first.
- Avoid looping between two positions (e.g., north then south repeatedly).
- Do not move in a direction that leads to a cell you've already marked as visited on the memory map (value 2), unless there are no unexplored (0 or -) directions left.

Backtracking is a general problem-solving strategy used to find solutions by exploring all potential candidates and abandoning paths that violate constraints. The non-recursive version implements this idea using an explicit stack, rather than relying on recursive function calls.

Backtracking means:
- You explore paths one step at a time.
- If you reach a dead end, you backtrack (reverse the last step) and try a different path.
- You avoid revisiting the same positions repeatedly unless no other options exist.
- NEVER go back and forth between directions like north-south-north ot east-south-east
- NEVER go back and forth between two positions (e.g., north then south repeatedly)

Step-by-step logic:

1. Initialize the Search
- Start with an empty solution and push it onto a stack.
- The stack keeps track of all the partial solutions that need to be explored.

2. Iterate Until Stack is Empty
- Repeatedly pop the top partial solution from the stack — this is the current state to explore.
- If this solution is complete (i.e., reaches the required length or satisfies the goal), it is returned as a valid result.
- If it's incomplete, continue to the next step.

3. Generate Candidates
- For the current position in the solution, generate all possible candidates (for example, numbers, choices, moves).
- For each candidate:
  - Check constraints: determine whether adding this candidate would violate any rule (e.g., no repeated elements, illegal moves, revisited positions).
  - If the candidate is valid, create a new solution by extending the current one with the candidate.
  - Push this new partial solution onto the stack for future exploration.

4. Backtracking Happens Naturally
- If all candidates from a state are invalid or already explored, that state is simply discarded (popped and not extended).
- This is the essence of backtracking — moving backward in the decision tree to explore unexplored branches.

5. Repeat
- The algorithm continues exploring new paths by popping from the stack, trying valid extensions, and pushing those back onto the stack.
- This process continues until a complete solution is found, or all paths have been exhausted.

Core principles:
- Explicit state tracking: Instead of recursive call stacks, the algorithm uses a manual stack to hold decisions.
- Constraint enforcement: Invalid partial solutions are discarded early.
- Exhaustive exploration: All possibilities are explored unless a valid solution is found early (if only one is needed).
- Memory-efficient control: The stack-based approach prevents stack overflow and gives more control over search order.

Respond with ONLY the next direction to move: north, south, east, or west. No extra text.
""")

class LLMMazeSolver(MazeSolver):
    """
    LLMMazeSolver using Large Language Models.
    """

    def __init__(self, maze):
        """
        Initialize the solver with a Maze instance.
        """
        # No separate directions list needed here, uses global constants
        super().__init__(maze)
        self.maze = maze
        maze.set_algorithm(self.__class__.__name__)
        self.memory_maze = np.full((3, 3), UNKNOWN)  # Initialize with 3x3 grid
        self.memory_center = (1, 1)  # This keeps track of the maze start position in memory_maze

    def solve(self):
        provider = config.get('LLM', 'provider')
        max_steps = config.getint('SOLVER', 'max_steps', fallback=50)
        if config.getboolean('DEFAULT', 'development_mode'):
            max_steps = 10
            logging.warning(f"Development mode is enabled. Setting max_steps to {max_steps}.")

        current_position = self.maze.start_position
        self.maze.move(current_position)
        path = [current_position]
        self.steps = 0
        #history of decissions 
        history = []  # stores 'north', 'south', etc.
        #

        llm_model = GPTFactory.create_gpt_model(provider, SYSTEM_PROMPT)

        while self.steps < max_steps:
            self.steps += 1

            # Get local context directly as a dictionary {name: status}
            local_context_dict = self._compute_local_context(self.maze, self.maze.current_position)

            # Build the payload for the prompt
            payload = {
                "local_context": local_context_dict,  # Use the dictionary directly
                "exit_reached": self.maze.at_exit()
            }

            valid_moves = [
                name for name, vec in ACTION_TO_DIRECTION.items()
                if self.maze.can_move(self.maze.current_position, vec)
            ]
            logging.info(f"Step {self.steps}, Valid moves: {valid_moves}")
            prompt = self._convert_json_to_prompt(payload, history, valid_moves)
            #print(prompt)

            retries = 0
            move_successful = False

            while retries < MAX_RETRIES_PER_STEP and not move_successful:
                logging.debug(f" Step {self.steps}, Position: {self.maze.current_position}, Prompt: {prompt}")
                response = llm_model.generate_response(prompt)
                logging.info(f"LLM response: {response}")
                direction_name = self._process_llm_response(response)  # Expects "north", "south", etc.

                if direction_name in ACTION_TO_DIRECTION:
                    move_vector = ACTION_TO_DIRECTION[direction_name]  # Get the (-1, 0) style vector

                    # Compute new position
                    r, c = self.maze.current_position
                    dr, dc = move_vector
                    new_position = (r + dr, c + dc)

                    # Validate move using the current position and the direction vector
                    if self.maze.can_move(self.maze.current_position, move_vector):
                        self.maze.move(new_position)
                        path.append(new_position)
                        history.append(direction_name)
                        logging.info(f"Moved {direction_name} to {new_position}")
                        # Update mental map with new position and surroundings
                        self._update_memory_maze(new_position, local_context_dict)

                        move_successful = True  # Exit retry loop
                    else:
                        retries += 1
                        logging.warning(
                            f"Invalid move attempted by LLM at step {self.steps}, from {self.maze.current_position}, towards {direction_name}. Retrying ({retries}/{MAX_RETRIES_PER_STEP})...")
                        # Optional: Print actual context from maze for debugging mismatch
                        # self.maze.print_local_context(self.maze.current_position)
                else:
                    retries += 1
                    logging.warning(
                        f"Invalid/unrecognized direction in LLM response at step {self.steps}: '{response}'. Retrying ({retries}/{MAX_RETRIES_PER_STEP})...")

            if not move_successful:
                logging.error(
                    f"Failed to make a valid move at step {self.steps} after {MAX_RETRIES_PER_STEP} retries. Position: {self.maze.current_position}. Context given to LLM: {local_context_dict}")
                break  # Stop simulation if stuck

            if self.maze.at_exit():
                logging.info(f"Exit found at position: {self.maze.current_position}.")
                break

        if self.steps >= max_steps and not self.maze.at_exit():
            logging.warning("Maximum steps exceeded without reaching the exit.")
            # Consider if you need the LLM response here or just log the error

        return path

    def _memory_maze_to_string(self):
        """
        Converts the agent's memory maze into a readable string for better communication with the LLM.
        """
        # Adding row and column headers for better referencing
        header = "    " + " ".join([f"{col:2}" for col in range(self.memory_maze.shape[1])])
        border = "   " + "-" * (3 * self.memory_maze.shape[1])

        # Build grid with row numbers for easy referencing
        grid = "\n".join(
            f"{row:2} | " + " ".join(self._map_symbol(value) for value in self.memory_maze[row])
            for row in range(self.memory_maze.shape[0])
        )

        # Legend for the symbols
        legend = (
            "\nLegend:\n"
            "  - : Unknown\n"
            "  0 : Corridor\n"
            "  1 : Wall\n"
            "  2 : Visited Path\n"
            "  3: Starting Point"
            "  X : Current Position\n"
        )

        return f"{legend}\n{header}\n{border}\n{grid}\n{border}"

    def _map_symbol(self, value):
        """
        Maps memory maze values to their corresponding symbols.
        """
        return {
            UNKNOWN: "-",
            CORRIDOR: "0",
            WALL: "1",
            PATH: "2",
            START: "3",
            CURRENT: "X"
        }.get(value, "?")  # Fallback for unexpected values

    def _compute_local_context(self, maze, position) -> dict:
        """
        Computes the local context around the position.

        Args:
            maze: The Maze object.
            position: The (row, col) tuple of the current position.

        Returns:
            A dictionary mapping direction names ('north', 'south', 'east', 'west')
            to their status ('open' or 'wall').
        """
        context = {}
        for vec, direction_name in DIRECTION_NAMES.items():
            neighbor = (position[0] + vec[0], position[1] + vec[1])

            if maze.is_within_bounds(neighbor):
                context[direction_name] = "open" if maze.can_move(position, vec) else "wall"
            else:
                context[direction_name] = "outside" if maze.at_exit() else "wall"
        return context

    def _convert_json_to_prompt(self, js: dict, history: list[str], valid_moves: list[str]) -> str:
        """
        Converts the payload dictionary (containing the context dict) to a prompt string.
        Now expects js['local_context'] to be a dictionary.

        :param js: The payload dictionary. Example:
                   {'local_context': {"north": "wall", ...}, 'exit_reached': False}
        :type js: dict
        :return: A formatted string prompt.
        :rtype: str
        """
        context_dict = js["local_context"]  # This should now be the dictionary
        exit_reached = js["exit_reached"]
        history_str = ", ".join(history) if history else "none"
        available = ", ".join(valid_moves) if valid_moves else "none"
        memory_maze_str = self._memory_maze_to_string()

        # Basic check to ensure context_dict is actually a dictionary
        if not isinstance(context_dict, dict):
            logging.error(
                f"[_convert_json_to_prompt] FATAL: Expected local_context to be a dict, but got {type(context_dict)}. Payload: {js}")
            # Return a default error prompt or raise an exception
            raise TypeError(f"[_convert_json_to_prompt] Expected local_context to be a dict, got {type(context_dict)}")

        # Use .get() for safer access in case a key is unexpectedly missing
        prompt = (
            f"You are navigating through a maze. Your current surroundings are:\n"
            f"- To the north: {context_dict.get('north', 'ERROR')}\n"
            f"- To the south: {context_dict.get('south', 'ERROR')}\n"
            f"- To the east: {context_dict.get('east', 'ERROR')}\n"
            f"- To the west: {context_dict.get('west', 'ERROR')}\n\n"
            f"Have you reached the exit? {exit_reached}\n"
            f"Available directions: {available}\n"
            f"History of previous steps (oldest to newest): {history_str}.\n"
            f"Memory map of the maze visited so far:\n{memory_maze_str}\n\n"
            f"Avoid looping unless there is no option\n"
            f"You are solving this maze using Backtracking:\n"
            f"- Avoid undoing your last step unless no other path is available.\n"
            f"- Do not alternate back and forth between two directions.\n"
            f"- Prefer new unexplored directions.\n"
            f"Your decision should be based on surroundings and previous moves.\n\n"
            f"Respond with ONLY one word: north, south, east, or west. Do not include any other text."
        )
        return prompt

    def _expand_memory_maze(self, direction):
        """Expand the memory maze in the specified direction."""
        rows, cols = self.memory_maze.shape
        if direction == 'vertical':
            new_maze = np.full((rows + 2, cols), UNKNOWN)
            new_maze[1:-1, :] = self.memory_maze
            self.memory_maze = new_maze
            self.memory_center = (self.memory_center[0] + 1, self.memory_center[1])
        else:  # horizontal
            new_maze = np.full((rows, cols + 2), UNKNOWN)
            new_maze[:, 1:-1] = self.memory_maze
            self.memory_maze = new_maze
            self.memory_center = (self.memory_center[0], self.memory_center[1] + 1)

    def _update_memory_maze(self, position, context):
        rel_row = position[0] - self.maze.start_position[0] + self.memory_center[0]
        rel_col = position[1] - self.maze.start_position[1] + self.memory_center[1]

        # Expand memory maze vertically if needed
        while rel_row <= 0 or rel_row >= self.memory_maze.shape[0] - 1:
            self._expand_memory_maze('vertical')
            rel_row = position[0] - self.maze.start_position[0] + self.memory_center[0]

        # Expand memory maze horizontally if needed
        while rel_col <= 0 or rel_col >= self.memory_maze.shape[1] - 1:
            self._expand_memory_maze('horizontal')
            rel_col = position[1] - self.maze.start_position[1] + self.memory_center[1]

        # Update previous position
        self.memory_maze[self.memory_maze == CURRENT] = START

        # Mark current position
        self.memory_maze[rel_row, rel_col] = CURRENT

        # Update walls or corridors around current position
        directions = {'north': (-1, 0), 'south': (1, 0), 'east': (0, 1), 'west': (0, -1)}
        for direction, (dr, dc) in directions.items():
            nr, nc = rel_row + dr, rel_col + dc
            if 0 <= nr < self.memory_maze.shape[0] and 0 <= nc < self.memory_maze.shape[1]:
                if context[direction] == "wall":
                    self.memory_maze[nr, nc] = WALL
                elif context[direction] == "open":
                    self.memory_maze[nr, nc] = CORRIDOR

    def _expand_memory_maze(self, direction):
        """Expand the memory maze in the specified direction."""
        rows, cols = self.memory_maze.shape
        if direction == 'vertical':
            new_maze = np.full((rows + 2, cols), CORRIDOR)
            new_maze[1:-1, :] = self.memory_maze
            self.memory_maze = new_maze
            self.memory_center = (self.memory_center[0] + 1, self.memory_center[1])
        else:  # horizontal
            new_maze = np.full((rows, cols + 2), CORRIDOR)
            new_maze[:, 1:-1] = self.memory_maze
            self.memory_maze = new_maze
            self.memory_center = (self.memory_center[0], self.memory_center[1] + 1)

    def _process_llm_response(self, response: str) -> str | None:
        """
        Process the LLM response to extract a valid direction name.
        Returns 'north', 'south', 'east', 'west', or None.
        """
        if not response:
            logging.warning("Received empty response from LLM.")
            return None

        # Clean up common variations
        cleaned_response = response.lower().strip().replace("'", "").replace("\"", "")

        if cleaned_response in ACTION_TO_DIRECTION:
            return cleaned_response
        else:
            # Optional: More complex parsing could be added here if needed
            logging.warning(f"LLM response '{response}' is not a valid direction ({list(ACTION_TO_DIRECTION.keys())}).")
            return None


# (Example usage __main__ block remains largely the same, ensure it handles potential errors)
if __name__ == "__main__":
    clean_output_folder()
    setup_logging()
    try:
        mazes_file = f"{INPUT}mazes.h5"
        # Load fewer mazes for faster testing
        mazes = load_mazes(mazes_file, 10)
        logging.info(f"Loaded {len(mazes)} mazes from {mazes_file}")
        if not mazes:
            raise FileNotFoundError("No mazes loaded.")  # Or handle appropriately

        # Sort mazes by complexity ascending

        mazes.sort(key=lambda maze: maze.complexity, reverse=False)

        test_maze = mazes[1]
        print(f"Testing maze {test_maze.index}...")
        test_maze.print_ascii()
        test_maze.animate = True  # Disable animation for faster debugging if needed
        test_maze.save_movie = False

        solver = LLMMazeSolver(test_maze)
        solution = solver.solve()

        if solution:
            test_maze.set_solution(solution)
            solved_mazes = [test_maze]
            pdf_filename = f"{OUTPUT}solved_maze_llm_{test_maze.index}.pdf"
            save_mazes_as_pdf(solved_mazes, pdf_filename)
            logging.info(f"Saved solved maze PDF to {pdf_filename}")
            # Add movie saving if enabled
            if config.getboolean("MONITORING", "save_solution_movie", fallback=True):
                movie_filename = f"{OUTPUT}solved_maze_llm_{test_maze.index}.gif"
                save_movie_v2(solved_mazes, movie_filename)
                save_movie(solved_mazes,f"v1_{movie_filename}")
                logging.info(f"Saved solution animation to {movie_filename}")
        else:
            logging.error(f"Failed to find a solution for maze {test_maze.index}.")

    except FileNotFoundError as e:
        logging.error(f"Error loading mazes file '{mazes_file}': {e}. Check INPUT path in config.properties.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)  # Log traceback