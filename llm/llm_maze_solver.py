#  Python
import configparser
import json
import logging
import os

import numpy as np

from llm.gpt_factory import GPTFactory
from maze_solver import MazeSolver
from utils import setup_logging, load_mazes, save_mazes_as_pdf, save_movie

OUTPUT = config.get("FILES", "OUTPUT", fallback="output/")

WALL = 1
CORRIDOR = 0
START = 3
OUTSIDE = 5
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRECTION_TO_ACTION = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}
ACTION_TO_DIRECTION = {'north': (-1, 0), 'south': (1, 0), 'east': (0, 1), 'west': (0, -1)}

PARAMETERS_FILE = "config.properties"
SECRETS_FILE = "secrets.properties"


class LLMMazeSolver(MazeSolver):
    """
    LLMMazeSolver is an AI-based maze-solving class utilizing Large Language Models (LLMs).

    This class is designed to solve a maze using intelligent decision-making driven by
    LLMs, specifically for navigating a grid-based maze represented by the provided Maze
    instance. It integrates with the Maze class to determine valid moves and to track
    the solver's progress towards reaching the exit. The class uses configurable LLM
    settings and limits the number of steps to prevent infinite loops in cases where
    the maze may not be solvable.

    :ivar maze: The instance of the Maze class being solved. The maze provides the
                current position, starting position, and methods to check valid moves
                and detect when the exit is reached.
    :type maze: Maze
    :ivar directions: A list representing possible movement directions in a 2D grid.
                      North is defined as (-1, 0), East as (0, 1), South as (1, 0),
                      and West as (0, -1).
    :type directions: list of tuple
    """

    def __init__(self, maze):
        """
        Initialize the solver with a Maze instance.

        Args:
            maze: an instance of Maze which has methods:
                  - is_valid_move(position)
                  - at_exit()
                  and attributes:
                  - start_position (tuple)
                  - current_position (tuple)
        """
        # Directions: 0=North, 1=East, 2=South, 3=West
        self.directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        super().__init__(maze)
        self.maze = maze
        maze.set_algorithm(self.__class__.__name__)

    def solve(self):
        """
        Solve the maze using LLMs. Each move is executed via
        Maze.move(), ensuring the Maze's current_position is updated.

        Returns:
            path (list of positions): The sequence of (row, col) tuples of the path.
        """

        config = configparser.ConfigParser()
        config.read(PARAMETERS_FILE)

        provider = config.get('LLM', 'provider')
        max_steps = config.getint('DEFAULT', 'max_steps')
        self.maze.algorithm = self.__class__.__name__
        current_position = self.maze.start_position
        # Initialize Maze's current position
        self.maze.move(current_position)
        path = [current_position]
        self.steps = 0

        llm_model = GPTFactory.create_gpt_model(provider)

        while self.steps < max_steps:
            self.steps += 1

            local_context = self._compute_loca_context_json(self.maze, current_position, self.directions)
            prompt = self._convert_json_to_prompt(local_context)
            logging.debug("Prompt: " + prompt)
            response = llm_model.generate_response(prompt)
            direction = self._process_llm_response(response)

            if direction:
                move = ACTION_TO_DIRECTION[direction]
                new_position = (current_position[0] + move[0], current_position[1] + move[1])
                if self.maze.is_valid_move(new_position):
                    self.maze.move(new_position)
                    current_position = new_position
                    path.append(current_position)
                    logging.info(f"Moved {direction} to {current_position}")

            # Check if the maze is solved by asking Maze via at_exit().
            if self.maze.at_exit():
                logging.info(
                    f"Exit found at position: {str(current_position)} Thank you! \n {llm_model.generate_response(thank_you)}")
                break

        if self.steps >= max_steps:
            thank_you = "Sorry, I can't find the exit. We will try again."
            logging.info(f"{thank_you} Response {llm_model.generate_response(thank_you)}")
            logging.error("Maximum steps exceeded without reaching the exit.")
        return path

    def _convert_json_to_prompt(self, js: json) -> str:
        """
        Converts a JSON object to a prompt string.

        The function processes the input JSON object and generates a string representation in a
        structured format suitable to be used as a prompt. The specific format of the output string
        is determined by the processing logic defined in the function.

        :param js: The JSON object to be converted to a prompt string.
        :type js: json

        :return: A string representation of the JSON structured as a prompt.
        :rtype: str
        """
        context = js["local_context"]
        exit_reached = js["exit_reached"]

        prompt = (
            f"You are navigating through a maze. Your current surroundings are:\n"
            f"- To the north: {context['north']}\n"
            f"- To the south: {context['south']}\n"
            f"- To the east: {context['east']}\n"
            f"- To the west: {context['west']}\n\n"
            f"Have you reached the exit? {exit_reached}\n"
            "What direction should you move next to reach the exit? "
            "Respond with ONLY one word - either: north, south, east, or west. Do not include any other text."
        )

        return prompt

    def _compute_loca_context_json(self, maze, position, directions) -> json:
        """
        Cnvert local context to JSON
        Args:
            maze (Maze): Maze object with .grid attribute (2D numpy array)
            position (tuple): (row, col) position in the maze
            directions (list): List of (dr, dc) direction offsets
        Returns:
            local context in a JSON format like: prompt = '{"local_context": {"north": "wall", "south": "open", "east": "open", "west": "wall"}, "exit_reached": false}'
        """

        local_context = self._compute_local_context(maze, position, directions)
        # verify if exit has been reached
        exit_reached = self.maze.at_exit()
        # convert contex        # convert context to JSON
        prompt = {
            "local_context": {
                "north": local_context[0],
                "south": local_context[1],
                "east": local_context[2],
                "west": local_context[3]
            },
            "exit_reached": exit_reached
        }
        logging.debug(f"Local context: {prompt}")

        return prompt

    def _process_llm_response(self, response: str) -> str:
        """
        Process the LLM response to extract a valid direction.
        
        Args:
            response (str): The raw response from the LLM
            
        Returns:
            str: One of 'north', 'south', 'east', 'west' or None if invalid
        """
        logging.debug(f"LLM response: {response}")
        response = response.lower().strip()
        valid_directions = ['north', 'south', 'east', 'west']
        for direction in valid_directions:
            if direction in response:
                return direction
        return None

    def _compute_local_context(self, maze, position, directions):
        """
        Vectorized version of local context computation using NumPy.

        Args:
            maze (Maze): Maze object with .grid attribute (2D numpy array)
            position (tuple): (row, col) position in the maze
            directions (list): List of (dr, dc) direction offsets

        Returns:
            list: Values of the 4 surrounding cells, defaulting to OUTSIDE if out of bounds
        """
        r, c = position
        rows, cols = maze.grid.shape
        dr_dc = np.array(directions)
        positions = dr_dc + np.array([r, c])  # shape: (4, 2)

        # Check which are in bounds
        in_bounds = (
                (positions[:, 0] >= 0) & (positions[:, 0] < rows) &
                (positions[:, 1] >= 0) & (positions[:, 1] < cols)
        )

        # Clip indices to valid range, so we can use them safely
        clamped = np.clip(positions, [0, 0], [rows - 1, cols - 1])

        # Use fancy indexing to get neighbor values
        neighbor_vals = maze.grid[clamped[:, 0], clamped[:, 1]]

        # Apply OUTSIDE where out of bounds
        context = np.where(in_bounds, neighbor_vals, OUTSIDE)

        return context.tolist()


# Example usage:
if __name__ == "__main__":
    # setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.debug("Logging is configured.")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ← Up one level
    input_mazes = os.path.join(project_root, "input",
                               "mazes.pkl")  # → C:/Users/imold/PycharmProjects/AIGames/input/mazes.pkl

    mazes = load_mazes(input_mazes)

    solved_mazes = []

    # sort mazes by complexity
    mazes.sort(key=lambda x: x.complexity)
    easy_maze = mazes[0]
    easy_maze.set_algorithm("LLMMazeSolver")
    easy_maze.easy_maze = False
    easy_maze.animate = False

    solver = LLMMazeSolver(easy_maze)
    solution = solver.solve()
    easy_maze.set_solution(solution)
    solved_mazes.append(easy_maze)

    save_mazes_as_pdf(solved_mazes, f"{OUTPUT}solved_mazes_rnn.pdf")
    save_movie(solved_mazes, f"{OUTPUT}solved_mazes_LLM.mp4")
