#  Python
import configparser
import json
import logging

import numpy as np

from llm.gpt_factory import GPTFactory
from maze import Maze
from maze_solver import MazeSolver
from utils import setup_logging, load_mazes, save_mazes_as_pdf, save_movie

WALL = 1
CORRIDOR = 0
START = 3
OUTSIDE = 5
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRECTION_TO_ACTION = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}
ACTION_TO_DIRECTION = {'north': (-1, 0), 'south': (1, 0), 'east': (0, 1), 'west': (0, -1)}

PARAMETERS_FILE = "config.properties"
SECRETS_FILE = "secrets.properties"

config = configparser.ConfigParser()
config.read(PARAMETERS_FILE)
OUTPUT = config.get("FILES", "OUTPUT", fallback="output/")
INPUT = config.get("FILES", "INPUT", fallback="input/")

SYSTEM_PROMPT = (
    """You will help me solve a maze following the {algorithm}. You will do this from the view point of 
    someone in the maze, without the full view of the maze but only the cells immediately around you. 
    You will be given the local context of the cells around the current position in this format 
    `prompt = '{"local_context": {"north": "wall", "south": "open", "east": "open", "west": "wall"}, "exit_reached": false}'`.  
    "wall" is a maze wall and moving in that direction is not permitted. "open" is a maze corridor, and it is a 
    direction that can be explored.  "outside" marks a neighbour position outside the maze, when the current position is at the exit.  
    Important note, this is a call from an API. You will respond with only the direction to travel next 
    like: north, south, east, or west. Maximum output tokens are 1."""
)

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
        provider = config.get('LLM', 'provider')
        max_steps = 10  # for debug only
        current_position = self.maze.start_position
        self.maze.move(current_position)
        path = [current_position]
        self.steps = 0

        llm_model = GPTFactory.create_gpt_model(provider, SYSTEM_PROMPT)

        while self.steps < max_steps:
            self.steps += 1

            # Compute the local context as a list...
            local_context_list = self._compute_local_context(self.maze, current_position, self.directions)
            payload = {
                "local_context": {
                    "north": local_context_list[0],
                    "south": local_context_list[1],
                    "east": local_context_list[2],
                    "west": local_context_list[3]
                },
                "exit_reached": self.maze.at_exit()
            }
            prompt = self._convert_json_to_prompt(payload)
            logging.info(f"Prompt: {prompt}")
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
                else:
                    # logging.warning("Invalid move")
                    resp = llm_model.generate_response(
                        "Not a valid move . You can only move in a direction marked as 'open' by the local context and not in a direction marked with 'wall'."
                    )
                    logging.warning(f"Invalid response: {resp}")
            else:
                resp = llm_model.generate_response(
                    "Not a valid response. Respond with ONLY one word - either: north, south, east, or west. Do not include any other text. Wait for the next prompt."
                )
                logging.warning(f"Invalid response: {resp}")

            if self.maze.at_exit():
                logging.info(f"Exit found at position: {current_position}.")
                break

        if self.steps >= max_steps:
            msg = "Sorry, I can't find the exit. We will try again."
            logging.info(f"{msg} Response: {llm_model.generate_response(msg)}")
            logging.error("Maximum steps exceeded without reaching the exit.")

        return path

    def _compute_local_context(self, maze, position, directions):
        """
        Computes the local context for the given position in the maze.

        Args:
            maze (Maze): Maze object with .grid attribute (2D numpy array)
            position (tuple): (row, col) position in the maze.
            directions (list): List of (dr, dc) direction offsets.

        Returns:
            list: A list of 4 string labels for the surrounding cells,
                  mapping WALL (1) to "wall", CORRIDOR (0) to "open", and
                  OUTSIDE (5) to "outside".
        """
        r, c = position
        rows, cols = maze.grid.shape
        dr_dc = np.array(directions)
        positions = dr_dc + np.array([r, c])  # shape: (4, 2)

        in_bounds = ((positions[:, 0] >= 0) & (positions[:, 0] < rows) &
                     (positions[:, 1] >= 0) & (positions[:, 1] < cols))

        clamped = np.clip(positions, [0, 0], [rows - 1, cols - 1])
        neighbor_vals = maze.grid[clamped[:, 0], clamped[:, 1]]
        context = np.where(in_bounds, neighbor_vals, OUTSIDE).tolist()

        mapping = {
            WALL: "wall",
            CORRIDOR: "open",
            OUTSIDE: "outside"
        }
        return [mapping.get(value, "unknown") for value in context]

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

        logging.warning(f"no direction returned. response: {response}")
        return None


# Example usage:
if __name__ == "__main__":
    # setup logging
    setup_logging()

    mazes_file = f"{INPUT}mazes.pkl"
    mazes = load_mazes(mazes_file)

    solved_mazes = []

    # sort mazes by complexity
    mazes = list(mazes)
    mazes.sort(key=lambda x: x.complexity)
    easy_maze = mazes[0]
    easy_maze.animate = True
    easy_maze.save_movie = True

    solver = LLMMazeSolver(easy_maze)
    solution = solver.solve()
    easy_maze.set_solution(solution)
    solved_mazes.append(easy_maze)

    save_mazes_as_pdf(solved_mazes, f"{OUTPUT}solved_mazes_rnn.pdf")
    save_movie(solved_mazes, f"{OUTPUT}solved_mazes_LLM.mp4")
