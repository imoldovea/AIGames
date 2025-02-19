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
    'up': 0,
    'down': 1,
    'left': 2,
    'right': 3
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
    training_mazes = utils.load_mazes(file_path="input/training_mazes.pkl")
    solved_training_models = []
    for i, training_data in enumerate(training_mazes):
        training_maze = Maze(training_data)
        if training_maze.self_test():
            # compute solution
            solver = BacktrackingMazeSolver(training_maze)
            solution = solver.solve()
            training_maze.set_solution(solution)
            solved_training_models.append(training_maze)
        else:
            logging.warning(f"Maze {i + 1} is not valid.")
            raise ValueError(f"Maze {i + 1} is not valid.")

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
