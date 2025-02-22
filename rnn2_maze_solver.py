# rnn2_maze_solver.py
import logging
from maze_solver import MazeSolver
import numpy as np
from torch.utils.data import DataLoader, Dataset
import utils
from maze import Maze  # Adjust the import path if necessary
from utils import load_mazes
from backtrack_maze_solver import BacktrackingMazeSolver
from configparser import ConfigParser
import os

# -------------------------------
# Hyperparameters and Configurations
# -------------------------------
PARAMETERS_FILE = "config.properties"
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
OUTPUT = "output/"
# Define the path to save/load the models
RNN_MODEL_PATH = "path/to/your/rnn_model.pth"
GRU_MODEL_PATH = "path/to/your/gru_model.pth"
LSTM_MODEL_PATH = "path/to/your/lstm_model.pth"


logging.getLogger().setLevel(logging.INFO)

import torch
import torch.nn as nn
import torch.optim as optim


class MazeBaseModel(nn.Module):
    def __init__(self):
        super(MazeBaseModel, self).__init__()

    def forward(self, x):
        """
        This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")

    def train_model(self, dataloader, num_epochs=20, learning_rate=0.001, device='cpu'):
        """
        Generic training loop using CrossEntropyLoss and Adam optimizer.
    
        Parameters:
            dataloader (DataLoader): Dataloader for training data.
            num_epochs (int): Number of epochs to train.
            learning_rate (float): Learning rate for the optimizer.
            device (str): Device to train on ('cpu' or 'cuda').
    
        Returns:
            self: The trained model.
        """
        # Move the model to the specified device ('cpu' or 'cuda').
        self.to(device)

        # Define the optimizer as Adam and set the learning rate.
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Define the loss function as cross-entropy loss.
        criterion = nn.CrossEntropyLoss()

        # Set the model to training mode.
        self.train()

        # Loop over the specified number of epochs.
        for epoch in range(num_epochs):
            running_loss = 0.0  # Accumulate loss for the current epoch

            # Iterate through batches of inputs and targets from the dataloader.
            for inputs, targets in dataloader:
                # Add a sequence length dimension to inputs and move to the specified device.
                inputs = inputs.unsqueeze(1).to(device).float()


                # Move targets to the specified device.
                targets = targets.to(device)

                # Reset the gradients of model parameters.
                optimizer.zero_grad()

                # Perform a forward pass through the model to get the outputs.
                outputs = self.forward(inputs)

                # Compute the loss between the outputs and the targets.
                loss = criterion(outputs, targets)

                # Backpropagate the loss to compute gradients.
                loss.backward()

                # Update the model parameters using the optimizer.
                optimizer.step()

                # Accumulate the loss scaled by the batch size.
                running_loss += loss.item() * inputs.size(0)

            # Calculate the average loss for the current epoch.
            epoch_loss = running_loss / len(dataloader.dataset)

            # Print the epoch number and the corresponding loss.
            logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        # Return the trained model.
        return self


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
# Custom Dataset for Training Samples
# -------------------------------
class MazeTrainingDataset(Dataset):
    def __init__(self, data):
        """
        data: a list of tuples (local_context, target_action)
              local_context: list of 4 scalar values (e.g., [0, 0, 1, 1])
              target_action: integer (0-3)
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        local_context, target_action = self.data[idx]
        # Convert list of scalars to a numpy array then to a tensor of shape [4]
        local_context = torch.from_numpy(np.array(local_context)).float()
        return local_context, torch.tensor(target_action, dtype=torch.long)


# -------------------------------
# Training Utilities (Imitation Learning Setup)
# -------------------------------
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
        Each training sample is a tuple (local_context, target_action), where:
          - local_context: A list of 4 values representing the state of the maze cells
                           in the order [up, down, left, right] relative to the current position.
                           (0 for corridor, 1 for wall; out-of-bound cells are treated as walls.)
          - target_action: An integer (0: up, 1: down, 2: left, 3: right) representing the move
                           from the current position to the next position in the solution.
        Returns:
            list: A list of (local_context, target_action) training samples.
        """
        dataset = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        direction_to_target = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}

        for maze in self.training_mazes:
            solution = maze.get_solution()  # List of coordinates from start to exit
            for i in range(len(solution) - 1):
                current_pos = solution[i]
                next_pos = solution[i + 1]
                local_context = self._compute_local_context(maze, current_pos, directions)
                move_delta = (next_pos[0] - current_pos[0], next_pos[1] - current_pos[1])
                if move_delta not in direction_to_target:
                    raise KeyError(f"Invalid move delta: {move_delta}")
                target_action = direction_to_target[move_delta]
                dataset.append((local_context, target_action))
        return dataset

    def _compute_local_context(self, maze, position, directions):
        """
        Computes the local context around a given position in a maze.
        Returns a list of cell states in the order defined by 'directions'.
        """
        r, c = position
        local_context = []
        for dr, dc in directions:
            neighbor = (r + dr, c + dc)
            cell_state = maze.grid[neighbor] if maze.is_within_bounds(neighbor) else WALL
            local_context.append(cell_state)
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


# Ensure other necessary imports are present

def main():
    TRAINING_MAZES_FILE = "input/training_mazes.pkl"
    TEST_MAZES_FILE = "input/mazes.pkl"

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
    device = config.get("DEFAULT", "device")

    #RNN model:
    if not RETRAIN_MODEL and os.path.exists(RNN_MODEL_PATH):
        rnn_model = torch.load(f"{OUTPUT}rnn_model.pth")
        logging.info("Loaded RNN model")
    else:
        logging.info("Training RNN model")
        rnn_model = MazeRNN2Model(
            input_size=config.getint("RNN", "input_size"),
            hidden_size=config.getint("RNN", "hidden_size"),
            num_layers=config.getint("RNN", "num_layers"),
            output_size=config.getint("RNN", "output_size"),
        )
        rnn_model.train_model(
            dataloader,
            num_epochs=config.getint("RNN", "num_epochs"),
            learning_rate=config.getfloat("RNN", "learning_rate"),
            device=device,
        )
        logging.info("Done training RNN model")
        torch.save(rnn_model.state_dict(), RNN_MODEL_PATH)
        logging.info("Saved RNN model")

    # Initialize GRU Model
    if not RETRAIN_MODEL and os.path.exists(GRU_MODEL_PATH):
        gru_model = MazeGRUModel(
            input_size=config.getint("GRU", "input_size"),
            hidden_size=config.getint("GRU", "hidden_size"),
            num_layers=config.getint("GRU", "num_layers"),
            output_size=config.getint("GRU", "output_size"),
        )
        gru_model.load_state_dict(torch.load(GRU_MODEL_PATH))
        gru_model.to(device)
        print("GRU model loaded from file.")
    else:
        logging.info("Training GRU model")
        gru_model = MazeGRUModel(
            input_size=config.getint("GRU", "input_size"),
            hidden_size=config.getint("GRU", "hidden_size"),
            num_layers=config.getint("GRU", "num_layers"),
            output_size=config.getint("GRU", "output_size"),
        )
        gru_model.train_model(
            dataloader,
            num_epochs=config.getint("GRU", "num_epochs"),
            learning_rate=config.getfloat("GRU", "learning_rate"),
            device=device,
        )
        torch.save(gru_model.state_dict(), GRU_MODEL_PATH)
        print("GRU model trained and saved to file.")

    # Initialize LSTM Model
    if not RETRAIN_MODEL and os.path.exists(LSTM_MODEL_PATH):
        lstm_model = MazeLSTMModel(
            input_size=config.getint("LSTM", "input_size"),
            hidden_size=config.getint("LSTM", "hidden_size"),
            num_layers=config.getint("LSTM", "num_layers"),
            output_size=config.getint("LSTM", "output_size"),
        )
        lstm_model.load_state_dict(torch.load(LSTM_MODEL_PATH))
        lstm_model.to(device)
        print("LSTM model loaded from file.")
    else:
        logging.info("Training LSTM model")
        lstm_model = MazeLSTMModel(
            input_size=config.getint("LSTM", "input_size"),
            hidden_size=config.getint("LSTM", "hidden_size"),
            num_layers=config.getint("LSTM", "num_layers"),
            output_size=config.getint("LSTM", "output_size"),
        )
        lstm_model.train_model(
            dataloader,
            num_epochs=config.getint("LSTM", "num_epochs"),
            learning_rate=config.getfloat("LSTM", "learning_rate"),
            device=device,
        )
        torch.save(lstm_model.state_dict(), LSTM_MODEL_PATH)
        print("LSTM model trained and saved to file.")

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
