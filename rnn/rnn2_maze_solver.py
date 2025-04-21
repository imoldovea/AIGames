# rnn2_maze_solver.py
# RNN2MazeSolver
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import logging

import shutil
import socket
import subprocess
import sys
import traceback
from configparser import ConfigParser

import matplotlib.pyplot as plt
import numpy as np
import torch

import wandb
from chart_utility import (save_latest_loss_chart, save_neural_network_diagram, visualize_model_weights,
                           visualize_model_activations)
from maze_solver import MazeSolver
from maze_trainer import get_models  # Training function from maze_trainer.py
from utils import load_mazes, save_mazes_as_pdf, setup_logging
from utils import save_movie

# -------------------------------
# Global Configurations and Constants
# -------------------------------

# Maze encoding constants
PATH = 0
WALL = 1
START = 3

# File paths and configuration files
PARAMETERS_FILE = "config.properties"
config = ConfigParser()
config.read(PARAMETERS_FILE)
OUTPUT = config.get("FILES", "OUTPUT", fallback="output/")
INPUT = config.get("FILES", "INPUT", fallback="input/")

RNN_MODEL = config.get("FILES", "RNN_MODEL", fallback="rnn_model.pth")
GRU_MODEL = config.get("FILES", "RNN_MODEL", fallback="gru_model.pth")
LSTM_MODEL = config.get("FILES", "RNN_MODEL", fallback="lstm_model.pth")

RNN_MODEL_PATH = f"{INPUT}{RNN_MODEL}"
GRU_MODEL_PATH = f"{INPUT}{GRU_MODEL}"
LSTM_MODEL_PATH = f"{INPUT}{LSTM_MODEL}"
LOSS_PLOT_FILE = f"{OUTPUT}loss_plot.png"
LOSS_FILE = f"{OUTPUT}loss_data.csv"
MODELS_DIAGRAM = f"{OUTPUT}models_diagram.pdf"
OUTPUT_PDF = f"{OUTPUT}solved_mazes.pdf"
SECRETS = "secrets.properties"
MAZES = LSTM_MODEL = config.get("FILES", "MAZES", fallback="mazes.pkl")
TEST_MAZES_FILE = f"{INPUT}{MAZES}"


# -------------------------------
# RNN2MazeSolver Class (Inference)
# -------------------------------
class RNN2MazeSolver(MazeSolver):
    DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

    def __init__(self, maze, model, device="cpu"):
        """
        Initializes the solver with a Maze and a trained model.
        """
        super().__init__(maze)  # Pass the maze parameter to the parent constructor
        self.maze = maze
        self.device = torch.device(device)
        self.model = model
        self.model_type = model.mode_type
        maze.set_algorithm(self.__class__.__name__)
        model.to(self.device)
        model.eval()
        self.fig, self.ax = plt.subplots()
        self.img = None
        # Initialize the "recurrent" activations as an empty list
        self.activations = {'recurrent': []}
        recurrent = getattr(self.model, 'recurrent', None)
        if isinstance(recurrent, torch.nn.Module):
            recurrent.register_forward_hook(self.save_activation)

    # def save_activation(self, module, input, output):
    #     if isinstance(output, tuple):
    #         activation_tensor = output[0]
    #     else:
    #         activation_tensor = output
    #     self.activations['recurrent'].append(activation_tensor.detach().cpu().numpy())

    def solve(self):
        """
        Solves the maze by iteratively predicting the next move using a trained recurrent neural network model.
    
        Steps involved:
        1. Initialize the starting position and tracking variables (path and step count).
        2. Load the maximum allowable steps from the configuration file.
        3. In each iteration, perform the following:
           - Increment the step count and normalize it based on the maximum steps.
           - Compute the local context of the maze, which includes the states of cells surrounding the current position.
           - Calculate the relative position as the offset from the maze's starting point.
           - Combine local context, relative position, and normalized step number into an input feature vector.
           - Use the neural network to predict the next move based on the input features.
           - Compute the next position based on the predicted move and validate it (checking boundaries and wall collisions).
           - If valid, append the position to the solution path and update the current position.
           - Optionally, save the RNN activation values for visualization purposes.
        4. Continue until the exit is found or the maximum steps are reached.
        5. If the algorithm terminates without solving the maze, log a warning.
    
        Visualization:
        - Generate and save a video of the RNN activations if applicable.
    
        Returns:
            list: A list of positions representing the solution path from start to exit.
        """
        logging.debug("Solving maze...")
        current_pos = self.maze.start_position
        path = [current_pos]
        step_number = 0
        max_steps = config.getint("DEFAULT", "max_steps", fallback=40)
        while self.maze.exit != current_pos and len(path) < max_steps:
            step_number += 1
            step_number_normalized = step_number / max_steps
            local_context = self._compute_local_context(current_pos, self.DIRECTIONS)
            # Compute the relative position as the offset from the starting point
            relative_position = (
                current_pos[0] - self.maze.start_position[0],
                current_pos[1] - self.maze.start_position[1]
            )
            # Create input feature vector: local_context (4 values) + relative_position (2 values) + normalized step (1 value)
            input_features = np.array(local_context + list(relative_position) + [step_number_normalized],
                                      dtype=np.float32)
            input_tensor = torch.tensor(input_features).unsqueeze(0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                # Perform inference using the trained model to predict the next move
                output, hidden_activations = self.model(input_tensor, return_activations=True)
                self.activations['recurrent'].append(hidden_activations.squeeze(0).cpu().numpy())

                # Determine the action (direction) with the highest probability
                action = torch.argmax(output[0, -1], dim=0).item()

            # Calculate the move delta based on the predicted action
            move_delta = self.DIRECTIONS[action]
            # Compute the next position by applying the move delta to the current position
            next_pos = (current_pos[0] + move_delta[0], current_pos[1] + move_delta[1])

            # Check if the next position is within bounds or hits a wall
            if not self.maze.is_within_bounds(next_pos) or self.maze.grid[next_pos] == WALL:
                # If invalid, log the prediction and terminate the solution process
                logging.debug(f"Predicted invalid move to {next_pos} from {current_pos}.")
                break
            else:
                # Append the valid move to the solution path and update the current position
                path.append(next_pos)
                current_pos = next_pos

                # Save the activations at each step if available
                if 'recurrent' in self.activations:
                    # Compute the current activation and append it to the list.
                    act = self._compute_current_activation(
                        current_pos=current_pos,
                        relative_position=relative_position,
                        step_number_normalized=step_number_normalized,
                    )
                    self.activations['recurrent'].append(act.cpu().numpy())

                self.maze.move(current_pos)
        else:
            if self.maze.exit != current_pos:
                logging.debug(f"Reached max steps ({max_steps}) without finding a solution.")

        return path

    def get_recurrent_activations(self):
        return self.activations.get('recurrent')

    def _compute_current_activation(self, current_pos=None, relative_position=None,
                                    step_number_normalized=None):
        """
        Computes the current activation of the model based on the local context.

        This method:
        1. Retrieves the local context of the maze (e.g., surroundings of the current position)
           by calling _compute_local_context().
        2. Converts the local context (assumed to be a NumPy array) into a PyTorch tensor.
        3. Passes the tensor through the model (which is assumed to be a neural network) in evaluation mode.
        4. Appends the resulting activation (converted to numpy array) to the activations list.
        5. Returns the computed activation.
        """

        # Step 1: Get the local context as a NumPy array.
        local_context = self._compute_local_context(current_pos, self.DIRECTIONS)

        # Step 2: Preprocess the local context to match the model's expected input format.
        # We add a batch dimension (unsqueeze(0)) and convert to a FloatTensor.
        # It is assumed that self.device holds the target device (e.g., 'cpu' or 'cuda').
        input_features = np.array(local_context + list(relative_position) + [step_number_normalized],
                                  dtype=np.float32)
        input_tensor = torch.tensor(input_features).unsqueeze(0).unsqueeze(0).to(self.device)

        # Step 3: Set the model into evaluation mode and compute the activation.
        self.model.eval()
        with torch.no_grad():
            current_activation = self.model(input_tensor)

        # Step 5: Return the computed activation.
        return current_activation

    def _compute_local_context(self, position, directions):
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
def rnn2_solver(models, mazes, device):
    """
    Applies trained models to test data, solves mazes, and evaluates performance.

    Args:
        models (list): List of tuples containing model name and model object.
        mazes (list): List of maze data.
        device (str): Device to run the models on ('cpu' or 'cuda').

    Returns:
        list: Success rates of models in solving the test mazes.

    Steps:
    1. Load the test mazes from the specified file.
    2. Initialize an empty list to store solved mazes.
    3. Read configuration settings, including the maximum number of steps.
    4. Initialize an empty list to store success rates of models.
    5. Process each model:
        a. Initialize counters for successful solutions and total mazes.
        b. Iterate through each maze in the test data:
            i. Create a Maze object from maze data.
            ii. Create an RNN2MazeSolver instance for the maze and model.
            iii. Set the algorithm name for the maze.
            iv. Solve the maze using the solver and retrieve the solution path.
            v. Set the maze solution and test its validity.
            vi. If valid and within the step limit, log success; otherwise, log failure.
        c. Calculate the success rate for the model and log it.
    6. Plot mazes marked for further analysis or failures.
    7. Save the solved mazes to a PDF file.
    8. Return the success rates of all models.
    """
    logging.debug("Applying models to test data...")

    # Step 2: Initialize list for solved mazes
    solved_mazes = []

    # Step 3: Load configuration and maximum step settings
    config = ConfigParser()
    config.read("config.properties")
    max_steps = config.getint("DEFAULT", "max_steps")

    # Step 4: Initialize model success rates
    model_success_rates = []

    # Step 5: Process each model
    for model_name, model_obj in models:
        successful_solutions = 0  # Successful solution counter
        total_mazes = len(mazes)  # Total mazes to solve
        all_model_acivations = []
        # Step 5.b: Iterate through each maze
        for i, maze_data in enumerate(mazes):
            # Step 5.b.i: Create a Maze object
            maze = maze_data
            maze.save_movie = True

            if config.getboolean("MONITORING", "save_solution_movie", fallback=False):
                maze.set_save_movie(True)

            # Step 5.b.ii: Create a solver instance
            solver = RNN2MazeSolver(maze=maze, model=model_obj,  device=device)

            # Step 5.b.iii: Set the algorithm name for the maze
            maze.set_algorithm(model_name)

            # Step 5.b.iv: Solve the maze
            solution_path = solver.solve()
            activations = solver.get_recurrent_activations()
            all_model_acivations.append(activations)

            # Step 5.b.v: Set the solution and test its validity
            maze.set_solution(solution_path)
            if len(solution_path) < max_steps and maze.test_solution():
                # Step 5.b.vi (success): Log and increment successful solutions
                logging.debug(f"Solved Maze {i + 1}: {solution_path}")
                successful_solutions += 1
            else:
                # Step 5.b.vi (failure): Log and add maze to solved_mazes list
                logging.debug(f"Maze {i + 1} failed self-test.")
                solved_mazes.append(maze)

        # Step 5.c: Calculate success rate for the model
        success_rate = successful_solutions / total_mazes * 100
        model_success_rates.append((model_name, success_rate))
        logging.info(
            f"{model_name} Total mazes: {total_mazes}, Successful solutions: {successful_solutions}, "
            f"Success rate: {success_rate:.2f}%"
        )

        if config.getboolean("MONITORING", "generate_activations", fallback=False):
            visualize_model_activations(
                all_activations=all_model_acivations,
                output_folder=OUTPUT,
                model_name=model_name,
                video_filename=f"recurrent_activations_movie_{model_name}.mp4",
                fps=25
            )
    # Step 8: Return success rates
    return solved_mazes, model_success_rates

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def main():
    setup_logging()

    config = ConfigParser()
    config.read("config.properties")
    batch_size = config.getint("DEFAULT", "batch_size", fallback=128)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup Monitoring
    try:
        if config.getboolean("MONITORING", "wandb",fallback=True):
            os.environ['WANDB_MODE'] = 'offline'
            config_secrets = ConfigParser()
            config_secrets.read(SECRETS)
            wandb.login(key=config_secrets.get("WandB", "api_key"))
            wandb.init(project="maze_solver_training", config={
                "batch_size": batch_size,
                "device": str(device),
            }, reinit=True)

            wandb_run_url = wandb.run.get_url()
        else:
            wandb_run_url = "WandB Disabled"

        if config.getboolean("MONITORING", "dashboard", fallback=True):
            dashboard_process = subprocess.Popen(["python", "rnn/dashboard.py"])

        for handler in logging.getLogger('werkzeug').handlers:
            logging.getLogger('werkzeug').removeHandler(handler)

        tensorboard_process = None  # Default Declaration
        if config.getboolean("MONITORING", "tensorboard", fallback=True):
            if is_port_in_use(6006):
                logging.warning("TensorBoard is already running on port 6006. Skipping startup.")
            else:
                # Try to locate the tensorboard executable in the PATH.
                tb_exec = shutil.which("tensorboard")
                if tb_exec:
                    tensorboard_cmd = [tb_exec, "--logdir", f"{OUTPUT}tensorboard_data", "--port", "6006"]
                else:
                    # Optionally, try to run tensorboard as a module using an alternate entry point.
                    # This fallback uses 'tensorboard.main' which might exist depending on your installation.
                    logging.warning("TensorBoard executable not found in PATH. Trying to use module fallback.")
                    tensorboard_cmd = [sys.executable, "-m", "tensorboard.main", "--logdir",
                                       f"{OUTPUT}tensorboard_data", "--port", "6006"]
                    tensorboard_process = subprocess.Popen(tensorboard_cmd)

        tensorboard_url = "http://localhost:6006/"
        dash_dashboard_url = "http://127.0.0.1:8050/"
        # Log the URLs
        logging.info(
            f"\nWandB dashboard: {wandb_run_url}, \n"
            f"TensorBoard: {tensorboard_url}, \n"
            f"Dash Dashboard: {dash_dashboard_url}.\n "
        )

        # training
        logging.debug("Training models...")

        models = get_models()
        try:
            if config.getboolean("MONITORING", "save_neural_network_diagram", fallback=True):
                logging.info("Saving neural network diagram...")
                save_neural_network_diagram(models, OUTPUT)
            if config.getboolean("MONITORING", "save_last_loss_chart", fallback=True):
                logging.info("Generating latest loss chart...")
                if os.path.isfile(LOSS_FILE):
                    save_latest_loss_chart()
                else:
                    logging.warning(f"Loss file {LOSS_PLOT_FILE} is missing. Skipping loss chart generation.")
        except Exception as e:
            logging.error(f"An error occurred: {e}\n\n{traceback.format_exc()}")

        if config.getboolean("MONITORING", "generate_weights", fallback=True):
            logging.info("Generating model weights...")
            visualize_model_weights(models)

        # Apply the model to the test data.
        mazes = load_mazes(TEST_MAZES_FILE)
        solved_mazes, model_success_rates = rnn2_solver(models=models, mazes=mazes, device=device.type)
        logging.info(f"Model success rates: {model_success_rates}")

        #Plot mazes
        if config.getboolean("MONITORING", "save_mazes_as_pdf", fallback=True):
            logging.info("Saving mazes as PDF...")
            save_mazes_as_pdf(solved_mazes, OUTPUT_PDF)
        if config.getboolean("MONITORING", "print_mazes", fallback=True):
            logging.info("Printing mazes...")
            for maze in solved_mazes:
                maze.plot_maze(show_path=True, show_solution=False, show_position=False)
        if config.getboolean("MONITORING", "save_solution_movie", fallback=True):
            save_movie(solved_mazes, f"{OUTPUT}solved_mazes_rnn.mp4")

        if config.getboolean("MONITORING", "plotly", fallback=True):
            logging.info("Closing Dash Dashboard...")
            dashboard_process = subprocess.Popen(["python", "dashboard.py"])

        # save the running configuations
        with open(f"{OUTPUT}config.properties", "w") as configfile:
            config.write(configfile)
        logging.info("All done!")

    finally:
        try:
            if config.getboolean("MONITORING", "dashboard", fallback=True):
                dashboard_process.terminate()
        except Exception as e:
            logging.debug(f"Error finalizing Dash: {e}")
        try:
            if config.getboolean("MONITORING", "wandb", fallback=True):
                wandb.finish()
        except Exception as e:
            logging.debug(f"Error finalizing WandB: {e}")
        try:
            if config.getboolean("MONITORING", "tensorboard", fallback=True) and tensorboard_process is not None:
                tensorboard_process.terminate()
        except Exception as e:
            logging.debug(f"Error finalizing Tensorboard: {e}")

if __name__ == "__main__":
    main()
