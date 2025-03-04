"""
rnn2_maze_solver.py
RNN2MazeSolver

This module implements the RNN2MazeSolver class and its supporting methods to solve mazes via neural networks (RNN, GRU, LSTM models).
It includes functionalities to load mazes, solve them using trained models, and visualize the results.

Key Features:
- Supports multiple recurrent neural network (RNN) architectures: RNN, GRU, LSTM.
- Provides mechanisms for training models, solving mazes, visualizing activation states, and generating outputs.
- Leverages PyTorch for deep learning and matplotlib for visualizations.

The solver interacts with external configuration files to define hyperparameters.
"""

import numpy as np
import torch
import logging
import os, subprocess, traceback
import matplotlib.pyplot as plt
from configparser import ConfigParser
import wandb
from maze_solver import MazeSolver
from maze import Maze  # Assumes maze.py exists
from utils import load_mazes, save_mazes_as_pdf
from chart_utility import save_neural_network_diagram
from maze_trainer import train_models  # Import the training function from maze_trainer.py
from model import MazeRNN2Model, MazeGRUModel, MazeLSTMModel  # Models now live in model.py

# -------------------------------
# Global Configurations and Constants
# -------------------------------
RETRAIN_MODEL = True

# Maze encoding constants
PATH = 0
WALL = 1
START = 3

# File paths and configuration files
PARAMETERS_FILE = "config.properties"
OUTPUT = "output/"
INPUT = "input/"
RNN_MODEL_PATH = f"{INPUT}rnn_model.pth"
GRU_MODEL_PATH = f"{INPUT}gru_model.pth"
LSTM_MODEL_PATH = f"{INPUT}lstm_model.pth"
LOSS_FILE = f"{OUTPUT}loss_data.csv"
LOSS_PLOT_FILE = f"{OUTPUT}loss_plot.png"
MODELS_DIAGRAM = f"{OUTPUT}models_diagram.pdf"
OUTPUT_PDF = f"{OUTPUT}solved_mazes.pdf"
SECRETS = "secrets.properties"
TEST_MAZES_FILE = f"{INPUT}mazes.pkl"

logging.getLogger().setLevel(logging.INFO)
log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING)
log.disabled = True

# -------------------------------
# RNN2MazeSolver Class (Inference)
# -------------------------------
class RNN2MazeSolver(MazeSolver):
    """
    RNN2MazeSolver is an inference-based solver class that uses trained recurrent models to navigate and solve mazes.

    Attributes:
        maze (Maze): The maze to solve.
        device (torch.device): Specifies the hardware (CPU/GPU) for computation.
        model (torch.nn.Module): The trained neural network model.
        activations (dict): Stores activations from the RNN layers during inference for visualization.
    """

    def __init__(self, maze, model, device="cpu"):
        """
        Initializes the solver with a Maze and a trained model.

        Args:
            maze (Maze): The maze to solve.
            model (torch.nn.Module): The trained neural network model.
            device (str): Specifies the computation device (default "cpu").
        """
        self.maze = maze
        self.device = torch.device(device)
        self.model = model
        maze.set_algorithm(self.__class__.__name__)
        model.to(self.device)
        model.eval()
        self.activations = {}
        self.fig, self.ax = plt.subplots()
        self.img = None
        if hasattr(self.model, 'rnn'):
            self.model.rnn.register_forward_hook(self.save_activation)

    def save_activation(self, module, input, output):
        """
        Stores RNN activation outputs into the activations dictionary.
    
        Args:
            module (torch.nn.Module): The neural network module.
            input (torch.Tensor): The input to the module.
            output (torch.Tensor or tuple): The output from the module.
        """
        if isinstance(output, tuple):
            activation_tensor = output[0]
        else:
            activation_tensor = output
        self.activations['rnn'] = activation_tensor.detach().cpu().numpy()

    def solve(self, max_steps=25):
        """
        Solves the maze using the trained model.
    
        Args:
            max_steps (int): Maximum number of steps allowed to solve the maze.
    
        Returns:
            List[tuple]: The path taken through the maze as a list of coordinate tuples.
        """
        # Initialize the current position to the maze's start position
        current_pos = self.maze.start_position
        # Start the solution path with the initial position
        path = [current_pos]
        # Possible movement directions: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        # Initialize step counter
        step_number = 0

        # Loop to iteratively move through the maze until the exit is reached or max steps are exceeded
        while self.maze.exit != current_pos and len(path) < max_steps:
            # Increment step number and normalize it for model input
            step_number += 1
            step_number_normalized = step_number / max_steps

            # Compute the local context of the maze at the current position
            local_context = self._compute_local_context(current_pos, directions)

            # Create input features by combining local context and normalized step number
            input_features = np.append(local_context, step_number_normalized).astype(np.float32)
            # Convert features to a tensor and adjust dimensions to match model input requirements
            input_tensor = torch.tensor(input_features).unsqueeze(0).unsqueeze(0).to(self.device)

            # Perform inference with the trained model to determine the next action
            with torch.no_grad():
                output = self.model(input_tensor)  # Get model's output
                action = torch.argmax(output, dim=1).item()  # Decode the action with the highest probability

            # Visualize live RNN activations if available
            if 'rnn' in self.activations:
                act = self.activations['rnn'][0]
                self.ax.clear()  # Clear previous visualizations
                self.img = self.ax.imshow(act, aspect='auto', cmap='viridis')  # Display activations as an image
                self.ax.set_title("Live RNN Activations")

            # Compute the movement vector based on the action
            move_delta = directions[action]
            # Determine the resulting position after taking the action
            next_pos = (current_pos[0] + move_delta[0], current_pos[1] + move_delta[1])

            # Check if the predicted move is valid; if not, log it and terminate the solving process
            if not self.maze.is_within_bounds(next_pos) or self.maze.grid[next_pos] == WALL:
                logging.debug(f"Predicted invalid move to {next_pos} from {current_pos}.")
                break

            # Update the path with the new position and move in the maze
            path.append(next_pos)
            current_pos = next_pos
            self.maze.move(current_pos)

        # Close the visualization figure after the solving completes
        plt.close(self.fig)

        # Return the final path taken through the maze
        return path

    def _compute_local_context(self, position, directions):
        """
        Computes the local context (surrounding cells) of the agent's current position.
    
        Args:
            position (tuple): Current position of the agent in the maze (row, column).
            directions (list): List of relative movement directions.
    
        Returns:
            List[int]: The state of neighboring cells (0 for PATH, 1 for WALL, 3 for START).
        """
        r, c = position
        context = []
        for dr, dc in directions:
            neighbor = (r + dr, c + dc)
            cell_state = self.maze.grid[neighbor] if self.maze.is_within_bounds(neighbor) else WALL
            context.append(cell_state)
        return context

# -------------------------------
# Integration Functions
# -------------------------------
def rnn2_solver(models, device="cpu"):
"""
Solves multiple mazes using the provided models and visualizes the results.

Args:
    models (list): A list of tuples where each tuple contains the model name (str) and object (torch.nn.Module).
    device (str): Specifies the computation device (default "cpu").
"""
# Load test mazes from the pre-saved file
# Test mazes are stored in a predefined file (e.g., a binary file or dataset)
mazes = load_mazes(TEST_MAZES_FILE)

# Initialize an empty list to store the solved mazes
# This will store the results of each maze after being solved
solved_mazes = []

# Read the configuration properties file to get the maximum number of steps
# Retrieves configuration value for controlling the solver's limit
config = ConfigParser()
config.read("config.properties")
max_steps = config.getint("DEFAULT", "max_steps")

# Loop through each model to solve the test mazes
# Iterate through the provided models to evaluate performance
for model_name, model_obj in models:
    # Counter to keep track of successfully solved mazes
    successful_solutions = 0

    # Iterate through each maze and solve it with the current model
    for i, maze_data in enumerate(mazes):
        # Initialize the maze from the dataset
        maze = Maze(maze_data)
        # Create a solver instance for the maze using the current model
        solver = RNN2MazeSolver(maze, model_obj, device=device)
        # Record the algorithm used for solving this maze
        maze.set_algorithm(model_name)

        # Solve the maze using the current model
        # The solution path is generated and stored
        solution_path = solver.solve(max_steps=max_steps)
        maze.set_solution(solution_path)

        # Check if the solution is valid and log outcomes
        # Validate the solution for correctness within the step limit
        if len(solution_path) < max_steps and maze.test_solution():
            # Log the successfully solved maze and visualize the solution
            logging.debug(f"Solved Maze {i + 1}: {solution_path}")
            maze.plot_maze(show_path=False, show_solution=True, show_position=False)
            successful_solutions += 1
        else:
            # Log failures and visualize the path taken by the solver
            maze.plot_maze(show_path=True, show_solution=False, show_position=False)
            logging.debug(f"Maze {i + 1} failed self-test.")

        # Append the solved maze to the list of results
        # Collect the maze object with its solution for later visualization
        solved_mazes.append(maze)

    # Calculate and log the error rate and success statistics for the current model
    # Summarize performance of the current model on all test mazes
    total_mazes = len(mazes)
    error_rate = (total_mazes - successful_solutions) / total_mazes * 100
    logging.info(
        f"{model_name} Total mazes: {total_mazes}, Successful solutions: {successful_solutions}, Error rate: {error_rate:.2f}%")

# Save all solved mazes into a PDF file for visualization
# Generate a visual report of all solved mazes in a single output file
save_mazes_as_pdf(solved_mazes, OUTPUT_PDF)
(models, device="cpu"):
    """
    Solves multiple mazes using the provided models and visualizes the results.

    Args:
        models (list): A list of tuples where each tuple contains the model name (str) and object (torch.nn.Module).
        device (str): Specifies the computation device (default "cpu").
    """
    # Load test mazes from the pre-saved file
    mazes = load_mazes(TEST_MAZES_FILE)

    # Initialize an empty list to store the solved mazes
    solved_mazes = []

    # Read the configuration properties file to get the maximum number of steps
    config = ConfigParser()
    config.read("config.properties")
    max_steps = config.getint("DEFAULT", "max_steps")

    # Loop through each model to solve the test mazes
    for model_name, model_obj in models:
        successful_solutions = 0

        # Iterate through each maze and solve it with the current model
        for i, maze_data in enumerate(mazes):
            maze = Maze(maze_data)
            solver = RNN2MazeSolver(maze, model_obj, device=device)
            maze.set_algorithm(model_name)

            # Solve the maze using the current model
            solution_path = solver.solve(max_steps=max_steps)
            maze.set_solution(solution_path)

            # Check if the solution is valid and log outcomes
            if len(solution_path) < max_steps and maze.test_solution():
                logging.debug(f"Solved Maze {i + 1}: {solution_path}")
                maze.plot_maze(show_path=False, show_solution=True, show_position=False)
                successful_solutions += 1
            else:
                maze.plot_maze(show_path=True, show_solution=False, show_position=False)
                logging.debug(f"Maze {i + 1} failed self-test.")

            # Append the solved maze to the list of results
            solved_mazes.append(maze)

        # Calculate and log the error rate and success statistics for the current model
        total_mazes = len(mazes)
        error_rate = (total_mazes - successful_solutions) / total_mazes * 100
        logging.info(
            f"{model_name} Total mazes: {total_mazes}, Successful solutions: {successful_solutions}, Error rate: {error_rate:.2f}%")

    # Save all solved mazes into a PDF file for visualization
    save_mazes_as_pdf(solved_mazes, OUTPUT_PDF)


def main():
    """
    Main function to handle training and inference for the maze solver.

    Performs the following:
    - Loads configuration and secrets files.
    - Logs into WandB for tracking training progress.
    - Trains the models and evaluates their performance on test mazes.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = ConfigParser()
    os.environ['WANDB_MODE'] = 'online'
    config.read(SECRETS)
    wandb.login(key=config.get("WandB", "api_key"))
    dashboard_process = subprocess.Popen(["python", "dashboard.py"])
    logging.info("Training models...")
    config.read("config.properties")
    batch_size = config.getint("DEFAULT", "batch_size")
    wandb.init(project="maze_solver_training", config={
        "batch_size": batch_size,
        "device": str(device),
    })
    models = train_models(device=device, batch_size=batch_size)
    dashboard_process.terminate()
    try:
        save_neural_network_diagram(models, "output/")
    except Exception as e:
        logging.error(f"An error occurred: {e}\n\nStack Trace:{traceback.format_exc()}")
    rnn2_solver(models, device)
    wandb.finish()

if __name__ == "__main__":
    main()
