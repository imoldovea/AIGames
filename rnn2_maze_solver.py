# rnn2_maze_solver.py
from base_model import MazeBaseModel
from maze_trainer import MazeTrainingDataset, RNN2MazeTrainer
import logging
from numpy.f2py.auxfuncs import throw_error
import numpy as np
from maze_solver import MazeSolver
from torch.utils.data import DataLoader, Dataset
from maze import Maze  # Adjust the import path if necessary
from utils import load_mazes, save_mazes_as_pdf
from configparser import ConfigParser
import os, csv, subprocess
import torch
import torch.nn as nn
from chart_utility import save_latest_loss_chart, save_neural_network_diagram, visualize_model_weights
import traceback
import wandb
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

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
SECRETS = "secrets.properties"

TRAINING_MAZES_FILE = f"{INPUT}training_mazes.pkl"
TEST_MAZES_FILE = f"{INPUT}mazes.pkl"

logging.getLogger().setLevel(logging.INFO)
log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING)
log.disabled = True

# -------------------------------
# RNN Model Definition
# -------------------------------
class MazeRNN2Model(MazeBaseModel):
    def __init__(self, input_size=5, hidden_size=128, num_layers=2, output_size=4):
        super(MazeRNN2Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True  # Specifies if the first dimension is the batch size
        )
        self.fc = nn.Linear(
            in_features=hidden_size,  # Number of input features to the fully connected layer
            out_features=output_size  # Number of output features (actions in this case)
        )
        self.model_name = "RNN"

        # Initialize weights and biases with random values
        self._initialize_weights()

        self.fig, self.ax = plt.subplots()
        self.img = None

    def _initialize_weights(self):
        """
        Initializes the weights and biases of the RNN layers.
        
        This method assigns Xavier uniform initialization to the weights
        and zero initialization to the biases of the RNN model.
        
        Steps:
        - For each parameter in the RNN, check if it's a weight or a bias.
        - If it's a weight, use Xavier uniform initialization.
        - If it's a bias, set it to zero.
        
        Args:
            None
        
        Returns:
            None
        """
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        """
        Forward pass for the vanilla RNN model.

        x: Tensor of shape [batch_size, seq_length, input_size].
        Returns:
            Tensor of shape [batch_size, output_size].
        """
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # Forward propagate RNN
        out, _ = self.rnn(x, h0)
        # Take the output of the last time step
        out = out[:, -1, :]
        # Fully connected layer
        out = self.fc(out)
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

        # Initialize activations dictionary
        self.activations = {}

        # Initialize plotting components if needed
        self.fig, self.ax = plt.subplots()
        self.img = None

        # Example hook setup (modify as per your actual model architecture)
        # Assuming the model has an attribute 'rnn'
        if hasattr(self.model, 'rnn'):
            self.model.rnn.register_forward_hook(self.save_activation)

    def save_activation(self, module, input, output):
        """
        Hook function to save activations from the RNN layer.
        """
        if isinstance(output, tuple):
            # Assuming the first element is the activation tensor
            activation_tensor = output[0]
        else:
            activation_tensor = output
        self.activations['rnn'] = activation_tensor.detach().cpu().numpy()

    def solve(self, max_steps=25):
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
        step_number = 0
        # Loop until reaching the exit or exceeding the maximum steps
        while self.maze.exit != current_pos and len(path) < max_steps:
            step_number += 1
            step_number_normalized = step_number / max_steps
            # Compute the local context around the current position
            local_context = self._compute_local_context(current_pos, directions)

            # Combine local_context with step_number_normalized
            input_features = np.append(local_context,step_number_normalized).astype(np.float32)

            # Convert the local context to a tensor suitable for the model
                #input_tensor = torch.tensor([local_context], device=self.device).float().unsqueeze(1)
            # Combine local_context with step_number_normalized
            input_tensor = torch.tensor(input_features).unsqueeze(0).unsqueeze(0).to(self.device)  # Shape: [1, 1, 5]

            # Predict the next action using the model

            temperature = 1  # Increase for more exploration, decrease for more certainty
            with torch.no_grad():
                output = self.model(input_tensor)
                action = torch.argmax(output, dim=1).item()

            # Update the live plot with the current activation (if available)
            if 'rnn' in self.activations:
                # For example, visualizing the first sequence's activations:
                act = self.activations['rnn'][0]  # shape: [seq_length, hidden_size]
                self.ax.clear()
                self.img = self.ax.imshow(act, aspect='auto', cmap='viridis')
                self.ax.set_title("Live RNN Activations")
                #self.fig.canvas.draw()
                #self.fig.canvas.flush_events()
                #plt.pause(0.1)  # Pause briefly to update the plot

            # Calculate the next position based on the chosen action
            move_delta = directions[action]
            next_pos = (current_pos[0] + move_delta[0], current_pos[1] + move_delta[1])

            # Check if the move is valid (within bounds and not a wall)
            if not self.maze.is_within_bounds(next_pos) or self.maze.grid[next_pos] == WALL:
                logging.debug(f"Predicted invalid move to {next_pos} from {current_pos}.")
                break

            # Add the next position to the path and update the current position
            path.append(next_pos)
            current_pos = next_pos
            # Execute the move in the maze environment
            self.maze.move(current_pos)

        # Clean up the hook after solving
        if hasattr(self, 'hook_handle'):
            self.hook_handle.remove()

        # Close the plot if it's open
        plt.close(self.fig)

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
    config = ConfigParser()
    config.read("config.properties")
    max_steps = config.getint("DEFAULT", "max_steps")

    for model_name, model_obj in models:
        # Solve the maze using the model
        successful_solutions = 0
        for i, maze_data in enumerate(mazes):
            maze = Maze(maze_data)
            solver = RNN2MazeSolver(maze, model_obj, device=device)
            maze.set_algorithm(model_name)
            solution_path = solver.solve(max_steps=max_steps)
            maze.set_solution(solution_path)
            # Validate and visualize the solution
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
        logging.info(
            f"{model_name} Total mazes: {total_mazes}, Successful solutions: {successful_solutions}, Error rate: {error_rate:.2f}%")
    #print mazes solution as PDF
    save_mazes_as_pdf(solved_mazes, OUTPUT_PDF)


def train_models(device="cpu", batch_size=32):
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
            loss_writer = csv.writer(f)
            loss_writer.writerow(["model", "epoch", "loss"])  # Write header
    except Exception as e:
        throw_error(e)
        logging.error(f"Training mazes file not found: {e}")

    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir="output/maze_training")

    # Instantiate the trainer with the file path for training mazes.
    trainer = RNN2MazeTrainer(TRAINING_MAZES_FILE)
    dataset = trainer.create_dataset()
    logging.info(f"Created {len(dataset)} training samples.")
    train_ds = MazeTrainingDataset(dataset)
    # Create a DataLoader from the dataset
    num_workers = 2
    if device == "cuda":
        num_workers = 16
    dataloader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers)

    # Read configurations
    config = ConfigParser()
    config.read("config.properties")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    # RNN model:
    rnn_model = MazeRNN2Model(
        input_size=config.getint("RNN", "input_size", fallback=5),
        hidden_size=config.getint("RNN", "hidden_size"),
        num_layers=config.getint("RNN", "num_layers"),
        output_size=config.getint("RNN", "output_size", fallback=4),
    )
    rnn_model.to(device)
    # Start watching the model for gradients and parameters
    wandb.watch(rnn_model, log="all", log_freq=10)
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
            training_samples=config.getint("RNN", "training_samples"),
            weight_decay=config.getfloat("RNN", "weight_decay"),
            device=device,
            tensorboard_writer = writer
        )
        logging.info(f"Done training RNN model. Loss {loss:.4f}")
        torch.save(rnn_model.state_dict(), RNN_MODEL_PATH)
        logging.info("Saved RNN model")
        # Log final loss metric to wandb
        wandb.log({"RNN_final_loss": loss})
        writer.add_scalar("Loss/RNN_final_loss", loss)
    models.append(("RNN", rnn_model))

    # Initialize GRU Model
    gru_model = MazeGRUModel(
        input_size=config.getint("GRU", "input_size", fallback=5),
        hidden_size=config.getint("GRU", "hidden_size"),
        num_layers=config.getint("GRU", "num_layers"),
        output_size=config.getint("GRU", "output_size", fallback=4),
    )
    gru_model.to(device)
    wandb.watch(gru_model, log="all", log_freq=10)
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
            training_samples=config.getint("GRU", "training_samples"),
            weight_decay=config.getfloat("GRU", "weight_decay"),
            device=device,
            tensorboard_writer=writer
        )
        torch.save(gru_model.state_dict(), GRU_MODEL_PATH)
        logging.info(f"Done training GRU model. Loss {loss:.4f}")
        wandb.log({"GRU_final_loss": loss})
        writer.add_scalar("Loss/GRU_final_loss", loss)
    models.append(("GRU", gru_model))

    # Initialize LSTM Model
    lstm_model = MazeLSTMModel(

        input_size=config.getint("LSTM", "input_size", fallback=5),
        output_size=config.getint("LSTM", "output_size", fallback=4),
        hidden_size=config.getint("LSTM", "hidden_size"),
        num_layers=config.getint("LSTM", "num_layers"),
    )
    lstm_model.to(device)
    wandb.watch(lstm_model, log="all", log_freq=10)
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
            training_samples=config.getint("LSTM", "training_samples"),
            weight_decay=config.getfloat("LSTM", "weight_decay"),
            device=device,
            tensorboard_writer=writer
        )
        torch.save(lstm_model.state_dict(), LSTM_MODEL_PATH)
        logging.info(f"Done training LSTM model. Loss {loss:.4f}")
        wandb.log({"LSTM_final_loss": loss})
        writer.add_scalar("Loss/LSTM_final_loss", loss)
    models.append(("LSTM", lstm_model))

    # After training ends, save the latest loss chart
    if RETRAIN_MODEL:
        save_latest_loss_chart(loss_file_path=LOSS_FILE, loss_chart=LOSS_PLOT_FILE)

    writer.close()

    logging.info("Training complete.")
    # Return model names and trained models
    return models

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = ConfigParser()
    os.environ['WANDB_MODE'] = 'online'
    config.read(SECRETS)
    wandb.login(key=config.get("WandB", "api_key"))


    # Start the dashboard.py script as a separate process.
    dashboard_process = subprocess.Popen(["python", "dashboard.py"])
    # Train models
    logging.info("Training models...")
    config.read("config.properties")
    batch_size = config.getint(section="DEFAULT",option="batch_size")
    # Initialize wandb with your project and configuration
    wandb.init(project="maze_solver_training", config={
        "batch_size": batch_size,
        "device": str(device),
        # You can also add other hyperparameters here
    })
    models = train_models(device=device,batch_size=batch_size)
    # Ensure the dashboard process is terminated after training.
    dashboard_process.terminate()

    try:
        save_neural_network_diagram(models,"output/")
        #visualize_model_weights(models)
    except Exception as e:
        logging.error(f"An error occurred: {e}\n\nStack Trace:{traceback.format_exc()}")

    # Test models.
    rnn2_solver(models, device)

    # Finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    main()