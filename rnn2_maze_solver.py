# rnn2_maze_solver.py
import logging
from maze_solver import MazeSolver
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pyparsing import Empty
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import utils
from maze import Maze  # Adjust the import path if necessary
from utils import load_mazes
from backtrack_maze_solver import BacktrackingMazeSolver

# -------------------------------
# Hyperparameters and Configurations
# -------------------------------
EPOCHS = 20  # Number of training epochs

# Maze encoding constants
PATH = 0
WALL = 1
START = 3

# Action mapping for local context
ACTION_MAPPING = {
    (0, -1): 2,  # Left
    (0, 1): 3,   # Right
    (-1, 0): 0,  # Up
    (1, 0): 1    # Down
}

logging.getLogger().setLevel(logging.INFO)

# -------------------------------
# RNN Model Definition
# -------------------------------
class MazeRNN2Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MazeRNN2Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def predict(self, maze: torch.Tensor) -> torch.Tensor:
        # Placeholder for prediction logic
        return Empty

class RNN2MazeTrainer:
    def __init__(self, training_file_path="input/training_mazes.pkl"):
        """
        Initializes the trainer with a file path for training mazes.
        Loads and processes the training mazes upon instantiation.
        """
        self.training_file_path = training_file_path
        self.training_mazes = self._load_and_process_training_mazes()

    def _load_and_process_training_mazes(self):
        """
        Loads training mazes from the specified file and processes each maze.
        Returns:
            list: A list of Maze objects with computed solutions.
        """
        training_mazes = self._load_mazes_safe(self.training_file_path)
        solved_training_mazes = []
        for i, maze_data in enumerate(training_mazes):
            try:
                solved_training_mazes.append(self._process_maze(maze_data, i))
            except Exception as e:
                logging.error(f"Failed to process maze {i + 1}: {str(e)}")
                raise RuntimeError(f"Processing maze {i + 1} failed.") from e
        return solved_training_mazes

    def _load_mazes_safe(self, file_path):
        try:
            return utils.load_mazes(file_path)
        except FileNotFoundError as e:
            logging.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"Could not find the specified file: {file_path}") from e
        except Exception as e:
            logging.error(f"Error loading mazes: {str(e)}")
            raise RuntimeError("Maze loading failed.") from e

    def _process_maze(self, data, index):
        maze = Maze(data)
        if not maze.self_test():
            logging.warning(f"Maze {index + 1} failed validation.")
            raise ValueError(f"Maze {index + 1} failed self-test.")
        solver = BacktrackingMazeSolver(maze)
        maze.set_solution(solver.solve())
        return maze

    def create_dataset(self):
        """
        Constructs a dataset for training a model to navigate mazes.

        The dataset is a list of tuples where each tuple consists of local context
        information about the current position in a maze and the corresponding
        action required to move to the next position in the solution path.

        The function iterates through each training maze and calculates the solution
        path from the start to the exit. For each transition along the solution path,
        the function determines the local context of the current position and the
        required action to move to the next position. These are stored as a tuple in
        the resulting dataset.

        Each action is mapped to an integer based on a predefined direction-to-target
        mapping. The local context is determined by analyzing the maze structure around
        the current position.

        :param self:  
            Reference to the object instance containing training mazes and helper
            methods such as `_compute_local_context`.

        :return:  
            A list of tuples, where each tuple contains the local context of a position
            in the maze and the target action as an integer.
        :rtype: list[tuple[any, int]]

        :raises KeyError:  
            If a move delta calculated between consecutive positions in the solution
            is invalid (not present in the direction-to-target mapping).
        """
        dataset = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        direction_to_target = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}

        for maze in self.training_mazes:
            # Retrieve the solution path for the current maze (list of coordinates from start to exit)
            solution = maze.get_solution()
            for i in range(len(solution) - 1):
                # Get the current position and the next position in the solution path
                current_pos = solution[i]
                next_pos = solution[i + 1]
                # Compute the local context around the current position
                local_context = self._compute_local_context(maze, current_pos, directions)
                # Calculate the delta between the current position and the next position
                move_delta = (next_pos[0] - current_pos[0], next_pos[1] - current_pos[1])
                # Raise an error if the move is not valid (not in the predefined direction mapping)
                if move_delta not in direction_to_target:
                    raise KeyError(f"Invalid move delta: {move_delta}")
                # Map the move delta to the corresponding target action
                target_action = direction_to_target[move_delta]
                # Append the local context and target action as a training sample
                dataset.append((local_context, target_action))
        return dataset

    def _compute_local_context(self, maze, position, directions):
        """
        Computes the local context of a given position in a maze by checking the states 
        of its neighbors in the specified directions. The method evaluates each 
        neighbor's cell state, considering whether the position is within the bounds 
        of the maze grid.

        :param maze: The maze object containing the grid and logic to determine 
                     whether a position is within bounds.
        :type maze: Maze
        :param position: The current position in the maze given as a tuple (row, column).
        :type position: tuple[int, int]
        :param directions: A list of directions to evaluate, where each direction is 
                           represented as a tuple (delta_row, delta_column).
        :type directions: list[tuple[int, int]]
        :return: A list of cell states representing the local context around the 
                 specified position.
        :rtype: list[Any]
        """
        r, c = position
        local_context = []  # Initialize a list to store the state of neighboring cells
        for dr, dc in directions:
            # Calculate the neighbor's position relative to the current position
            neighbor = (r + dr, c + dc)
            # Determine the state of the neighbor cell:
            # If the neighbor is out of bounds, treat it as a wall
            cell_state = maze.grid[neighbor] if maze.is_within_bounds(neighbor) else WALL
            # Append the cell state to the local context
            local_context.append(cell_state)
        # Return the list of cell states representing the local context
        return local_context


# -------------------------------
# Maze Solver (Inference) Class
# -------------------------------
class RNN2MazeSolver(MazeSolver):
    def __init__(self, maze):
        """
        Initializes the RNN2MazeSolver with a Maze object.
        Args:
            maze (Maze): The maze to solve.
        """
        self.maze = maze
        maze.set_algorithm(self.__class__.__name__)
        # Placeholder: Load your trained model here (not implemented yet)

    def solve(self):
        # Placeholder: Implement the logic to solve the maze using the trained model.
        path = []
        return path

def main():
    TRAINING_MAZES_FILE = "input/training_mazes.pkl"
    TEST_MAZES_FILE = "input/mazes.pkl"

    # Instantiate the trainer with the file path for training mazes.
    trainer = RNN2MazeTrainer(TRAINING_MAZES_FILE)
    dataset = trainer.create_dataset()
    logging.info(f"Created {len(dataset)} training samples.")

    # Example of solving new mazes using the solver class.
    mazes = load_mazes(TEST_MAZES_FILE)
    for i, maze_data in enumerate(mazes):
        maze = Maze(maze_data)
        if maze.self_test():
            maze.plot_maze(show_path=True, show_solution=False, show_position=False)
        else:
            logging.warning("Test maze failed self-test.")

if __name__ == "__main__":
    main()
