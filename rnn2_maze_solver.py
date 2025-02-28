# rnn2_maze_solver.py
from base_model import MazeBaseModel
from maze_trainer import MazeTrainingDataset, RNN2MazeTrainer
import logging
from numpy.f2py.auxfuncs import throw_error
from maze_solver import MazeSolver
from torch.utils.data import DataLoader, Dataset
from maze import Maze  # Adjust the import path if necessary
from utils import load_mazes, save_mazes_as_pdf
from configparser import ConfigParser
import os, csv, subprocess
import torch
import torch.nn as nn
from chart_utility import save_latest_loss_chart, save_neural_network_diagram

# -------------------------------
# Hyperparameters and Configurations
# -------------------------------
RETRAIN_MODEL = False

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
MODELS_DIAGRAM = f"{OUTPUT}models_diagram.pdf"
OUTPUT_PDF = f"{OUTPUT}solved_mazes.pdf"

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
    def __init__(self, maze, model, device = "cpu" ):
        """
        Initializes the RNN2MazeSolver with a Maze object and loads the specified model.

        Args:
            maze (Maze): The maze to solve.
            model_type (str): Model type ("RNN", "GRU", "LSTM").
            device (str): Device for computation ("cpu" or "cuda").
        """
        self.maze = maze
        self.device = torch.device(device)
        self.model = model
        maze.set_algorithm(self.__class__.__name__)

        model.to(self.device)
        model.eval()
    def solve(self, max_steps=50):
        """
        Solve the maze using the loaded model.
        
        Args:
            max_steps (int): Maximum number of steps allowed to solve the maze.
        
        Returns:
            list: A list of coordinates representing the path from the start to the exit of the maze.
        """
        # Initialize starting position and path
        current_pos = self.maze.start_position
        path = [current_pos]
        # Predefined directions: up, down, left, and right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Loop until reaching the exit or exceeding the maximum steps
        while self.maze.exit != current_pos and len(path) < max_steps:
            # Compute the local context around the current position
            local_context = self._compute_local_context(current_pos, directions)
            # Convert the local context to a tensor suitable for the model
            input_tensor = torch.tensor([local_context], device=self.device).float().unsqueeze(1)

            # Predict the next action using the model
            with torch.no_grad():
                action_probs = self.model(input_tensor)
                action = torch.argmax(action_probs, dim=1).item()

            # Calculate the next position based on the chosen action
            move_delta = directions[action]
            next_pos = (current_pos[0] + move_delta[0], current_pos[1] + move_delta[1])

            # Check if the move is valid (within bounds and not a wall)
            if not self.maze.is_within_bounds(next_pos) or self.maze.grid[next_pos] == WALL:
                logging.error(f"Predicted invalid move to {next_pos} from {current_pos}.")
                break

            # Add the next position to the path and update the current position
            path.append(next_pos)
            current_pos = next_pos
            # Execute the move in the maze environment
            self.maze.move(current_pos)
        
        # Return the computed path
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

def rnn2_solver(models , device = "cpu"):
    # Solve the maze using the RNN
    mazes = load_mazes(TEST_MAZES_FILE)
    solved_mazes = []

    for model_name, model_obj in models:
        # Solve the maze using the model
        mazes = load_mazes(TEST_MAZES_FILE)
        successful_solutions = 0
        for i, maze_data in enumerate(mazes):
            maze = Maze(maze_data)
            solver = RNN2MazeSolver(maze, model_obj, device=device)
            maze.set_algorithm(model_name)
            solution_path = solver.solve(max_steps=MAX_STEPS)
            maze.set_solution(solution_path)
            # Validate and visualize the solution
            if len(solution_path) < MAX_STEPS and maze.test_solution():
                logging.debug(f"Solved Maze {i + 1}: {solution_path}")
                maze.plot_maze(show_path=False, show_solution=True, show_position=False)
                successful_solutions += 1
            else:
                maze.plot_maze(show_path=True, show_solution=False, show_position=False)
                logging.debug(f"Maze {i + 1} failed self-test.")
            solved_mazes.append(maze)
        total_mazes = len(mazes)
        error_rate = (total_mazes - successful_solutions) / total_mazes * 100
        logging.info(
            f"{model_name} Total mazes: {total_mazes}, Successful solutions: {successful_solutions}, Error rate: {error_rate:.2f}%")
    #print mazes solution as PDF
    save_mazes_as_pdf(solved_mazes, OUTPUT_PDF)


def train_models(device="cpu"):
    """
    Train RNN, GRU, and LSTM models on maze-solving data.

    Args:
        device (str): The device to use for training, either 'cpu' or 'cuda'.

    Steps:
        1. Initialize and load training data from the specified file path.
        2. Configure and train each of the models (RNN, GRU, LSTM).
        3. Save the trained model states to files for reuse.
        4. Optionally save the loss chart if retraining.

    Returns:
        list: List containing model names ["RNN", "GRU", "LSTM"] and the corresponding trained models.
    """
    models = []
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

    # RNN model:
    rnn_model = MazeRNN2Model(
        input_size=config.getint("RNN", "input_size"),
        hidden_size=config.getint("RNN", "hidden_size"),
        num_layers=config.getint("RNN", "num_layers"),
        output_size=config.getint("RNN", "output_size"),
    )
    if not RETRAIN_MODEL and os.path.exists(RNN_MODEL_PATH):
        state_dict = torch.load(RNN_MODEL_PATH)
        rnn_model.load_state_dict(state_dict)
        logging.info("RNN model loaded from file")
    else:
        logging.info("Training RNN model")
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
    models.append(("RNN", rnn_model))

    # Initialize GRU Model
    gru_model = MazeGRUModel(
        input_size=config.getint("GRU", "input_size"),
        hidden_size=config.getint("GRU", "hidden_size"),
        num_layers=config.getint("GRU", "num_layers"),
        output_size=config.getint("GRU", "output_size"),
    )
    if not RETRAIN_MODEL and os.path.exists(GRU_MODEL_PATH):
        state_dict = torch.load(GRU_MODEL_PATH)
        gru_model.load_state_dict(state_dict)
        logging.debug("GRU model loaded from file.")
    else:
        logging.info("Training GRU model")
        loss = gru_model.train_model(
            dataloader,
            num_epochs=config.getint("GRU", "num_epochs"),
            learning_rate=config.getfloat("GRU", "learning_rate"),
            device=device,
            writer=writer
        )
        torch.save(gru_model.state_dict(), GRU_MODEL_PATH)
        logging.info(f"Done training GRU model. Loss {loss:.4f}")
    models.append(("GRU", gru_model))

    # Initialize LSTM Model
    lstm_model = MazeLSTMModel(
        input_size=config.getint("LSTM", "input_size"),
        hidden_size=config.getint("LSTM", "hidden_size"),
        num_layers=config.getint("LSTM", "num_layers"),
        output_size=config.getint("LSTM", "output_size"),
    )
    if not RETRAIN_MODEL and os.path.exists(LSTM_MODEL_PATH):
        state_dict = torch.load(LSTM_MODEL_PATH)
        lstm_model.load_state_dict(state_dict)
        logging.info("LSTM model loaded from file.")
    else:
        logging.info("Training LSTM model")
        loss = lstm_model.train_model(
            dataloader,
            num_epochs=config.getint("LSTM", "num_epochs"),
            learning_rate=config.getfloat("LSTM", "learning_rate"),
            device=device,
            writer=writer
        )
        torch.save(lstm_model.state_dict(), LSTM_MODEL_PATH)
        logging.info(f"Done training LSTM model. Loss {loss:.4f}")
    models.append(("LSTM", lstm_model))

    # After training ends, save the latest loss chart
    if RETRAIN_MODEL:
        save_latest_loss_chart(loss_file_path=LOSS_FILE, loss_chart=LOSS_PLOT_FILE)

    # Return model names and trained models
    return models

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Start the dashboard.py script as a separate process.
    dashboard_process = subprocess.Popen(["python", "dashboard.py"])
    
    # Train models
    models = train_models(device)
    # Ensure the dashboard process is terminated after training.
    dashboard_process.terminate()

    try:
        save_neural_network_diagram(models,MODELS_DIAGRAM)
    except Exception as e:
        logging.error(f"Error saving neural network diagram: {e}")

    # Test models.
    rnn2_solver(models, device)

if __name__ == "__main__":
    main()