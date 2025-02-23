# rnn2_maze_solver.py
from base_model import MazeBaseModel
from maze_trainer import MazeTrainingDataset, RNN2MazeTrainer
import logging
from numpy.f2py.auxfuncs import throw_error
from maze_solver import MazeSolver
from torch.utils.data import DataLoader, Dataset
from maze import Maze  # Adjust the import path if necessary
from utils import load_mazes
from configparser import ConfigParser
import os, csv, subprocess
import torch
import torch.nn as nn
from chart_utility import save_latest_loss_chart

# -------------------------------
# Hyperparameters and Configurations
# -------------------------------
RETRAIN_MODEL = True

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
PARAMETERS_FILE = "config.properties"
OUTPUT = "output/"
INPUT = "input/"
# Define the path to save/load the models
RNN_MODEL_PATH = f"{INPUT}rnn_model.pth"
GRU_MODEL_PATH = f"{INPUT}gru_model.pth"
LSTM_MODEL_PATH = f"{INPUT}lstm_model.pth"
LOSS_FILE = f"{OUTPUT}loss_data.csv"
LOSS_PLOT_FILE = f"{OUTPUT}loss_plot.png"

TRAINING_MAZES_FILE = f"{INPUT}training_mazes.pkl"
TEST_MAZES_FILE = f"{INPUT}mazes.pkl"
MAX_STEPS = 40

logging.getLogger().setLevel(logging.INFO)

# -------------------------------
# RNN Model Definition
# -------------------------------
class MazeRNN2Model(MazeBaseModel):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MazeRNN2Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.model_name = "RNN"

    def forward(self, x):
        """
        Forward pass for the vanilla RNN model.

        x: Tensor of shape [batch_size, seq_length, input_size].
        Returns:
            Tensor of shape [batch_size, output_size].
        """
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        # Forward propagate RNN
        out, _ = self.rnn(x, h0)
        # Get the output from the last time step and pass it through the FC layer
        out = self.fc(out[:, -1, :])
        return out


# -------------------------------
# GRU Model Definition
# -------------------------------
class MazeGRUModel(MazeBaseModel):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MazeGRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.model_name = "GRU"

    def forward(self, x):
        """
        Forward pass for the GRU model.

        x: Tensor of shape [batch_size, seq_length, input_size].
        Returns:
            Tensor of shape [batch_size, output_size].
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out


# -------------------------------
# LSTM Model Definition
# -------------------------------
class MazeLSTMModel(MazeBaseModel):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MazeLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.model_name = "LSTM"

    def forward(self, x):
        """
        Forward pass for the LSTM model.

        x: Tensor of shape [batch_size, seq_length, input_size].
        Returns:
            Tensor of shape [batch_size, output_size].
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# -------------------------------
# Maze Solver (Inference) Class
# -------------------------------
class RNN2MazeSolver(MazeSolver):
    def __init__(self, maze, model_type="RNN", device="cpu"):
        """
        Initializes the RNN2MazeSolver with a Maze object and loads the specified model.

        Args:
            maze (Maze): The maze to solve.
            model_type (str): Model type ("RNN", "GRU", "LSTM").
            device (str): Device for computation ("cpu" or "cuda").
        """
        self.maze = maze
        self.model_type = model_type
        self.device = torch.device(device)
        maze.set_algorithm(self.__class__.__name__)

        # Load model based on type
        if model_type == "RNN":
            self.model = MazeRNN2Model(4, 14, 1, 4)  # Update parameters if dynamic loading is required
            self.model.load_state_dict(torch.load(RNN_MODEL_PATH, map_location=device))
        elif model_type == "GRU":
            self.model = MazeGRUModel(4, 14, 1, 4)
            self.model.load_state_dict(torch.load(GRU_MODEL_PATH, map_location=device))
        elif model_type == "LSTM":
            self.model = MazeLSTMModel(4, 14, 1, 4)
            self.model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=device))
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.model.to(self.device)
        self.model.eval()

    def solve(self, max_steps = 50):
        """
        Solve the maze using the loaded model.

        Returns:
            list: A list of coordinates representing the path from start to exit.
        """
        current_pos = self.maze.start_position
        path = [current_pos]
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

        while self.maze.exit != current_pos and len(path) < max_steps:
            # Compute local context
            local_context = self._compute_local_context(current_pos, directions)
            input_tensor = torch.tensor([local_context], device=self.device).float().unsqueeze(1)

            # Predict the action
            with torch.no_grad():
                action_probs = self.model(input_tensor)
                action = torch.argmax(action_probs, dim=1).item()

            # Update position based on action
            move_delta = directions[action]
            next_pos = (current_pos[0] + move_delta[0], current_pos[1] + move_delta[1])

            # Validate move
            if not self.maze.is_within_bounds(next_pos) or self.maze.grid[next_pos] == WALL:
                logging.error(f"Predicted invalid move to {next_pos} from {current_pos}.")
                break
            path.append(next_pos)
            current_pos = next_pos
            self.maze.move(current_pos)

        return path

    def _compute_local_context(self, position, directions):
        """
        Computes the local context around a given position in the maze.
        
        Args:
            position (tuple): Current position in the maze (row, col).
            directions (list): List of direction deltas.

        Returns:
            list: Local context (0 for path, 1 for wall).
        """
        r, c = position
        context = []
        for dr, dc in directions:
            neighbor = (r + dr, c + dc)
            cell_state = self.maze.grid[neighbor] if self.maze.is_within_bounds(neighbor) else WALL
            context.append(cell_state)
        return context

def rnn2_solver(device = "cpu"):
    # Solve the maze using the RNN
    mazes = load_mazes(TEST_MAZES_FILE)
    successful_solutions = 0
    for i, maze_data in enumerate(mazes):
        maze = Maze(maze_data)
        solver = RNN2MazeSolver(maze, model_type="RNN", device=device)
        maze.set_algorithm("RNN")
        solution_path = solver.solve(max_steps=MAX_STEPS)
        maze.set_solution(solution_path)
        # Validate and visualize the solution
        if len(solution_path)<MAX_STEPS and maze.test_solution():
            logging.info(f"Solved Maze {i + 1}: {solution_path}")
            maze.plot_maze(show_path=False, show_solution=True, show_position=False)
            successful_solutions += 1
        else:
            maze.plot_maze(show_path=True, show_solution=False, show_position=False)
            logging.debug(f"Maze {i + 1} failed self-test.")
    total_mazes = len(mazes)
    error_rate = (total_mazes - successful_solutions) / total_mazes * 100
    logging.info(f"RNN Total mazes: {total_mazes}, Successful solutions: {successful_solutions}, Error rate: {error_rate:.2f}%")

    # Solve the maze using the GRU
    mazes = load_mazes(TEST_MAZES_FILE)
    successful_solutions = 0
    for i, maze_data in enumerate(mazes):
        maze = Maze(maze_data)
        solver = RNN2MazeSolver(maze, model_type="GRU", device=device)
        maze.set_algorithm("GRU")
        solution_path = solver.solve(max_steps=MAX_STEPS)
        maze.set_solution(solution_path)
        # Validate and visualize the solution
        if len(solution_path)<MAX_STEPS and maze.test_solution():
            logging.info(f"Solved Maze {i + 1}: {solution_path}")
            maze.plot_maze(show_path=False, show_solution=True, show_position=False)
            successful_solutions += 1
        else:
            maze.plot_maze(show_path=True, show_solution=False, show_position=False)
            logging.debug(f"Maze {i + 1} failed self-test.")
    total_mazes = len(mazes)
    error_rate = (total_mazes - successful_solutions) / total_mazes * 100
    logging.info(f"GRU Total mazes: {total_mazes}, Successful solutions: {successful_solutions}, Error rate: {error_rate:.2f}%")


    # Solve the maze using the LSTM
    mazes = load_mazes(TEST_MAZES_FILE)
    successful_solutions = 0
    for i, maze_data in enumerate(mazes):
        maze = Maze(maze_data)
        solver = RNN2MazeSolver(maze, model_type="LSTM", device=device)
        maze.set_algorithm("LSTM")
        solution_path = solver.solve(max_steps=MAX_STEPS)
        maze.set_solution(solution_path)
        # Validate and visualize the solution
        if len(solution_path)<MAX_STEPS and maze.test_solution():
            logging.info(f"Solved Maze {i + 1}: {solution_path}")
            maze.plot_maze(show_path=False, show_solution=True, show_position=False)
            successful_solutions += 1
        else:
            maze.plot_maze(show_path=True, show_solution=False, show_position=False)
            logging.debug(f"Maze {i + 1} failed self-test.")
    total_mazes = len(mazes)
    error_rate = (total_mazes - successful_solutions) / total_mazes * 100
    logging.info(f"LSTM Total mazes: {total_mazes}, Successful solutions: {successful_solutions}, Error rate: {error_rate:.2f}%")


def main():
    # Start the dashboard.py script as a separate process.
    dashboard_process = subprocess.Popen(["python", "dashboard.py"])

    try:
        # CSV file for storing training progress.
        with open(LOSS_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "epoch", "loss"])  # Write header
    except Exception as e:
        throw_error(e)
        logging.error(f"Training mazes file not found: {e}")

    # Instantiate the trainer with the file path for training mazes.
    trainer = RNN2MazeTrainer(TRAINING_MAZES_FILE)
    dataset = trainer.create_dataset()
    logging.info(f"Created {len(dataset)} training samples.")
    train_ds = MazeTrainingDataset(dataset)
    # Create a DataLoader from the dataset
    dataloader = DataLoader(train_ds, batch_size=32, shuffle=True)

    # Read configurations
    config = ConfigParser()
    config.read("config.properties")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    #RNN model:
    if not RETRAIN_MODEL and os.path.exists(RNN_MODEL_PATH):
        rnn_model = torch.load(RNN_MODEL_PATH)
        logging.info("RNN model loaded from file")
    else:
        logging.info("Training RNN model")
        rnn_model = MazeRNN2Model(
            input_size=config.getint("RNN", "input_size"),
            hidden_size=config.getint("RNN", "hidden_size"),
            num_layers=config.getint("RNN", "num_layers"),
            output_size=config.getint("RNN", "output_size"),
        )
        loss = rnn_model.train_model(
            dataloader,
            num_epochs=config.getint("RNN", "num_epochs"),
            learning_rate=config.getfloat("RNN", "learning_rate"),
            device=device,
            writer=writer
        )
        logging.info(f"Done training RNN model. Loss {loss:.4f}")
        torch.save(rnn_model.state_dict(), RNN_MODEL_PATH)
        logging.info("Saved RNN model")

    # Initialize GRU Model
    if not RETRAIN_MODEL and os.path.exists(GRU_MODEL_PATH):
        gru_model = torch.load(GRU_MODEL_PATH)
        logging.debug("GRU model loaded from file.")
    else:
        logging.info("Training GRU model")
        gru_model = MazeGRUModel(
            input_size=config.getint("GRU", "input_size"),
            hidden_size=config.getint("GRU", "hidden_size"),
            num_layers=config.getint("GRU", "num_layers"),
            output_size=config.getint("GRU", "output_size"),
        )
        loss = gru_model.train_model(
            dataloader,
            num_epochs=config.getint("GRU", "num_epochs"),
            learning_rate=config.getfloat("GRU", "learning_rate"),
            device=device,
            writer=writer
        )
        torch.save(gru_model.state_dict(), GRU_MODEL_PATH)
        logging.info(f"Done training GRU model. Loss {loss:.4f}")

    # Initialize LSTM Model
    if not RETRAIN_MODEL and os.path.exists(LSTM_MODEL_PATH):
        lstm_model = torch.load(LSTM_MODEL_PATH)
        logging.info("LSTM model loaded from file.")
    else:
        logging.info("Training LSTM model")
        lstm_model = MazeLSTMModel(
            input_size=config.getint("LSTM", "input_size"),
            hidden_size=config.getint("LSTM", "hidden_size"),
            num_layers=config.getint("LSTM", "num_layers"),
            output_size=config.getint("LSTM", "output_size"),
        )
        loss = lstm_model.train_model(
            dataloader,
            num_epochs=config.getint("LSTM", "num_epochs"),
            learning_rate=config.getfloat("LSTM", "learning_rate"),
            device=device,
            writer=writer
        )
        torch.save(lstm_model.state_dict(), LSTM_MODEL_PATH)
        logging.info(f"Done training LSTM model. Loss {loss:.4f}")

        # After training ends, save the latest loss chart
        save_latest_loss_chart(
            loss_file_path=LOSS_FILE,
            loss_chart=LOSS_PLOT_FILE
        )

    # Ensure the dashboard process is terminated after training.
    dashboard_process.terminate()

    rnn2_solver(device)

if __name__ == "__main__":
    main()
