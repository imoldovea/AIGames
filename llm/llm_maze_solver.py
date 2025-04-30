#  Python
import configparser
import logging

from llm.gpt_factory import GPTFactory
from maze_solver import MazeSolver
from utils import setup_logging, load_mazes, save_mazes_as_pdf

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

MAX_RETRIES_PER_STEP = 2

PARAMETERS_FILE = "config.properties"
SECRETS_FILE = "secrets.properties"

config = configparser.ConfigParser()
config.read(PARAMETERS_FILE)
OUTPUT = config.get("FILES", "OUTPUT", fallback="output/")
INPUT = config.get("FILES", "INPUT", fallback="input/")

# (SYSTEM_PROMPT remains the same)
SYSTEM_PROMPT = (
    """You will help me solve a maze following the Backtracking strategy.

    Backtracking is a step-by-step search method where you explore possible paths through the maze by moving one step at a time.
    If you reach a dead end (no valid moves), you backtrack (undo the last move) and try a different direction.

    You act as someone **inside** the maze, without a full view of it.
    You can only see the immediate surroundings: north, south, east, and west.

    You will be provided with the local context at each position in this JSON format:
    '{"local_context": {"north": "wall", "south": "open", "east": "open", "west": "wall"}, "exit_reached": false}'.

    Meanings:
    - "wall" means a wall blocks that direction and moving there is not allowed.
    - "open" means a corridor you can move into.
    - "outside" means a neighbor position outside the maze, signaling you are at the exit.

    **Rules to follow:**
    - Never move toward a direction labeled as "wall".
    - Always prefer moving into an "open" corridor.
    - If a neighboring direction is "outside", immediately move there to exit the maze.
    - If multiple "open" directions exist, pick any open one.

    **Important:**
    - This is an API call.
    - You must respond with ONLY one word: 'north', 'south', 'east', or 'west'.
    - No explanations, no extra text.
    - Limit your output to a single token.
    """
)


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

    def solve(self):
        provider = config.get('LLM', 'provider')
        max_steps = config.getint('SOLVER', 'max_steps', fallback=20)
        if config.getboolean('DEFAULT', 'development_mode'):
            max_steps = 3
            logging.warning("Development mode is enabled. Setting max_steps to 3.")

        current_position = self.maze.start_position
        self.maze.move(current_position)
        path = [current_position]
        self.steps = 0

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
            prompt = self._convert_json_to_prompt(payload)  # Pass the dict payload

            retries = 0
            move_successful = False

            while retries < MAX_RETRIES_PER_STEP and not move_successful:
                logging.info(f" Step {self.steps}, Position: {self.maze.current_position}, Prompt: {prompt}")
                response = llm_model.generate_response(prompt)
                logging.info(f"LLM response: {response}")
                direction_name = self._process_llm_response(response)  # Expects "north", "south", etc.

                if direction_name and direction_name in ACTION_TO_DIRECTION:
                    move_vector = ACTION_TO_DIRECTION[direction_name]  # Get the (-1, 0) style vector

                    # Compute new position
                    r, c = self.maze.current_position
                    dr, dc = move_vector
                    new_position = (r + dr, c + dc)

                    # Validate move using the current position and the direction vector
                    if self.maze.can_move(self.maze.current_position, move_vector):
                        self.maze.move(new_position)
                        path.append(new_position)
                        logging.info(f"Moved {direction_name} to {new_position}")
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
            logging.error("Maximum steps exceeded without reaching the exit.")
            # Consider if you need the LLM response here or just log the error

        return path

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
        # Iterate through the defined directions using the global constants
        for direction_vector in DIRECTIONS_ORDERED:  # e.g., (-1, 0)
            direction_name = DIRECTION_NAMES[direction_vector]  # e.g., "north"

            if maze.can_move(position, direction_vector):
                context[direction_name] = "open"
            else:
                # TODO: Potentially check for "outside" if maze has a method like maze.is_outside(neighbor_pos)
                context[direction_name] = "wall"
        return context  # Returns dict like {"north": "wall", "south": "open", ...}

    def _convert_json_to_prompt(self, js: dict) -> str:
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
            "What direction should you move next to reach the exit? "
            "Respond with ONLY one word - either: north, south, east, or west. Do not include any other text."
        )
        return prompt

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

        test_maze = mazes[0]
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
        else:
            logging.error(f"Failed to find a solution for maze {test_maze.index}.")

    except FileNotFoundError as e:
        logging.error(f"Error loading mazes file '{mazes_file}': {e}. Check INPUT path in config.properties.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)  # Log traceback
