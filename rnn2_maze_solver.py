# rnn2_maze_solver.py
# RNN2MazeSolver

import numpy as np
import torch
import logging
from utils import setup_logging
import os, subprocess, traceback
import matplotlib.pyplot as plt
from configparser import ConfigParser
import wandb
from maze_solver import MazeSolver
from maze import Maze  # Assumes maze.py exists
from utils import load_mazes, save_mazes_as_pdf
from chart_utility import save_neural_network_diagram
from maze_trainer import train_models  # Training function from maze_trainer.py

# -------------------------------
# Global Configurations and Constants
# -------------------------------

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

# -------------------------------
# RNN2MazeSolver Class (Inference)
# -------------------------------
class RNN2MazeSolver(MazeSolver):
    DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

    def __init__(self, maze, model, device="cpu"):
        """
        Initializes the solver with a Maze and a trained model.
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
        if hasattr(self.model, 'recurrent'):
            self.model.recurrent.register_forward_hook(self.save_activation)

    def save_activation(self, module, input, output):
        if isinstance(output, tuple):
            activation_tensor = output[0]
        else:
            activation_tensor = output
        self.activations['recurrent'] = activation_tensor.detach().cpu().numpy()

    def solve(self, max_steps=25):
        current_pos = self.maze.start_position
        path = [current_pos]
        step_number = 0
        while self.maze.exit != current_pos and len(path) < max_steps:
            step_number += 1
            step_number_normalized = step_number / max_steps
            local_context = self._compute_local_context(current_pos, self.DIRECTIONS)
            input_features = np.append(local_context, step_number_normalized).astype(np.float32)
            input_tensor = torch.tensor(input_features).unsqueeze(0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(input_tensor)
                action = torch.argmax(output, dim=1).item()
            if 'recurrent' in self.activations:
                act = self.activations['recurrent'][0]
                self.ax.clear()
                self.img = self.ax.imshow(act, aspect='auto', cmap='viridis')
                self.ax.set_title("Live Recurrent Activations")
            move_delta = self.DIRECTIONS[action]
            next_pos = (current_pos[0] + move_delta[0], current_pos[1] + move_delta[1])
            if not self.maze.is_within_bounds(next_pos) or self.maze.grid[next_pos] == WALL:
                logging.debug(f"Predicted invalid move to {next_pos} from {current_pos}.")
                break
            path.append(next_pos)
            current_pos = next_pos
            self.maze.move(current_pos)
        plt.close(self.fig)
        return path

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
def rnn2_solver(models, device="cpu"):
    mazes = load_mazes(TEST_MAZES_FILE)
    solved_mazes = []
    config = ConfigParser()
    config.read("config.properties")
    max_steps = config.getint("DEFAULT", "max_steps")
    for model_name, model_obj in models:
        successful_solutions = 0
        for i, maze_data in enumerate(mazes):
            maze = Maze(maze_data)
            solver = RNN2MazeSolver(maze, model_obj, device=device)
            maze.set_algorithm(model_name)
            solution_path = solver.solve(max_steps=max_steps)
            maze.set_solution(solution_path)
            if len(solution_path) < max_steps and maze.test_solution():
                logging.debug(f"Solved Maze {i + 1}: {solution_path}")
                maze.plot_maze(show_path=False, show_solution=True, show_position=False)
                successful_solutions += 1
            else:
                maze.plot_maze(show_path=True, show_solution=False, show_position=False)
                logging.debug(f"Maze {i + 1} failed self-test.")
            solved_mazes.append(maze)
        total_mazes = len(mazes)
        error_rate = (total_mazes - successful_solutions) / total_mazes * 100
        logging.info(f"{model_name} Total mazes: {total_mazes}, Successful solutions: {successful_solutions}, Error rate: {error_rate:.2f}%")
    save_mazes_as_pdf(solved_mazes, OUTPUT_PDF)

def main():
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
    #setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.debug("Logging is configured.")

    main()
