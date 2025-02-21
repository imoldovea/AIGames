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

# Define constants

EPOCHS = 20  # Number of training epochs


# Action mapping
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

    def predict(self, maze):
        return Empty

class RRN2MazeSolver(MazeSolver):
    def __init__(self, maze):
        """
        Initializes the RRN2MazeSolver with a Maze object.
        Args:
            maze (Maze): The maze to solve.
        """
        self.maze = maze
        maze.set_algorithm(self.__class__.__name__)

    def solve(self):
        path = []
        return path

    def load_training_mazes(file_path="input/training_mazes.pkl"):
        """
        Loads and processes training mazes from a file, validates them, and computes solutions.
    
        Args:
            file_path (str): Path to the file containing the training mazes. Default is "input/training_mazes.pkl".
    
        Returns:
            list: A list of Maze objects with computed solutions.
    
        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If any maze in the file is not valid or fails the self-test.
        """
        try:
            # Fetch training data
            training_mazes = utils.load_mazes(file_path)
        except FileNotFoundError as e:
            logging.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"Could not find the specified file: {file_path}") from e
        except Exception as e:
            logging.error(f"An unexpected error occurred while loading mazes: {str(e)}")
            raise RuntimeError("Failed to load training mazes.") from e

        solved_training_models = []
        for i, training_data in enumerate(training_mazes):
            try:
                training_maze = Maze(training_data)
                if training_maze.self_test():
                    # Compute solution
                    solver = BacktrackingMazeSolver(training_maze)
                    solution = solver.solve()
                    training_maze.set_solution(solution)
                    solved_training_models.append(training_maze)
                else:
                    logging.warning(f"Maze {i + 1} is not valid.")
                    raise ValueError(f"Maze {i + 1} failed the self-test.")
            except ValueError as e:
                logging.error(f"Validation error for maze {i + 1}: {str(e)}")
                raise
            except Exception as e:
                logging.error(f"An unexpected error occurred with maze {i + 1}: {str(e)}")
                raise RuntimeError(f"Failed to process maze {i + 1}.") from e
    
        return solved_training_models


def create_local_context_dataset(training_mazes):
    """
    Creates a dataset for one-move-at-a-time training using local context.

    Each training sample is a tuple (local_context, target_action), where:
      - local_context: A list of 4 values representing the state of the maze cells
                       in the order [up, down, left, right] relative to the current position.
                       (0 for corridor, 1 for wall; out-of-bound cells are treated as walls.)
      - target_action: An integer (0: up, 1: down, 2: left, 3: right) representing the
                       move from the current position to the next position in the solution.

    Args:
        training_mazes (list): A list of Maze objects with a valid solution path.

    Returns:
        list: A list of (local_context, target_action) training samples.
    """
    dataset = []
    # Define the order of directions and the corresponding target mapping
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
    direction_to_target = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}

    for maze in training_mazes:
        solution = maze.get_solution()  # List of coordinates from start to exit
        # Iterate over each step in the solution path except the last one.
        for i in range(len(solution) - 1):
            current_pos = solution[i]
            next_pos = solution[i + 1]

            # Build the local context vector
            local_context = []
            r, c = current_pos
            for dr, dc in directions:
                neighbor = (r + dr, c + dc)
                # If neighbor is within bounds, use the cell's state; otherwise, treat as wall.
                if maze.is_within_bounds(neighbor):
                    cell_state = maze.grid[neighbor]
                else:
                    cell_state = 1  # out-of-bound => wall
                local_context.append(cell_state)

            # Compute the difference to determine the move taken.
            move_delta = (next_pos[0] - current_pos[0], next_pos[1] - current_pos[1])
            target_action = direction_to_target.get(move_delta)
            if target_action is None:
                # This should not happen if the solution path is valid.
                raise ValueError(f"Unexpected move {move_delta} from {current_pos} to {next_pos}")

            # Append the (input, target) pair to the dataset.
            dataset.append((local_context, target_action))

    return dataset


# Example usage:
# training_mazes is a list of Maze objects with solutions already set.
local_context_dataset = create_local_context_dataset(training_mazes)
print(f"Created {len(local_context_dataset)} training samples.")


def create_local_context_dataset(training_mazes):
    """
    Creates a dataset for one-move-at-a-time training using local context.

    Each training sample is a tuple (local_context, target_action), where:
      - local_context: A list of 4 values representing the state of the maze cells
                       in the order [up, down, left, right] relative to the current position.
                       (0 for corridor, 1 for wall; out-of-bound cells are treated as walls.)
      - target_action: An integer (0: up, 1: down, 2: left, 3: right) representing the
                       move from the current position to the next position in the solution.

    Args:
        training_mazes (list): A list of Maze objects with a valid solution path.

    Returns:
        list: A list of (local_context, target_action) training samples.
    """
    dataset = []
    # Define the order of directions and the corresponding target mapping
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
    direction_to_target = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}

    for maze in training_mazes:
        solution = maze.get_solution()  # List of coordinates from start to exit
        # Iterate over each step in the solution path except the last one.
        for i in range(len(solution) - 1):
            current_pos = solution[i]
            next_pos = solution[i + 1]

            # Build the local context vector
            local_context = []
            r, c = current_pos
            for dr, dc in directions:
                neighbor = (r + dr, c + dc)
                # If neighbor is within bounds, use the cell's state; otherwise, treat as wall.
                if maze.is_within_bounds(neighbor):
                    cell_state = maze.grid[neighbor]
                else:
                    cell_state = 1  # out-of-bound => wall
                local_context.append(cell_state)

            # Compute the difference to determine the move taken.
            move_delta = (next_pos[0] - current_pos[0], next_pos[1] - current_pos[1])
            target_action = direction_to_target.get(move_delta)
            if target_action is None:
                # This should not happen if the solution path is valid.
                raise ValueError(f"Unexpected move {move_delta} from {current_pos} to {next_pos}")

            # Append the (input, target) pair to the dataset.
            dataset.append((local_context, target_action))

    return dataset


# Example usage:
# training_mazes is a list of Maze objects with solutions already set.
local_context_dataset = create_local_context_dataset(training_mazes)
print(f"Created {len(local_context_dataset)} training samples.")


def main():
    """
    Main entry point for training and utilizing a maze-solving neural network model.

    This script configures the device to be used for computation, loads the training data,
    creates a dataset and dataloader, initializes and trains the maze-solving neural
    network model, and saves the trained model for subsequent inference. An example usage
    of solving a maze with the trained model is also illustrated.

    :raises FileNotFoundError: If the specified file for loading mazes does not exist.
    :raises RuntimeError: If a GPU is expected but not available.
    """
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    # Fetch training data
    training_mazes = load_training_mazes("input/training_mazes.pkl")
    #torch.save(model.state_dict(), 'output/maze_rnn_model.pth')
    logging.info("Model trained and saved successfully.")

    # Example of solving a new maze
    mazes = load_mazes("input/mazes.pkl")
    for i, maze_data in enumerate(mazes):
        # Initialize and configure your test_maze as needed
        maze = Maze(maze_data)
        if maze.self_test():
            maze.plot_maze(show_path=True, show_solution=False, show_position=False)  # Assuming this method visualizes the maze and solution
        else:
            logging.warning("Test maze failed self-test.")

if __name__ == "__main__":
    main()
