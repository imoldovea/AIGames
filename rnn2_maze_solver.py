# rnn2_maze_solver.py
import logging
from maze_solver import MazeSolver
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
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
OUTPUT = "output/"

logging.getLogger().setLevel(logging.INFO)

# -------------------------------
# RNN Model Definition (Vanilla RNN)
# -------------------------------
class MazeRNN2Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MazeRNN2Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch_size, seq_length, input_size)
        # Initialize the hidden state (h0) with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)

        # Pass the input sequence and hidden state through the RNN layer
        # out contains the RNN outputs for all time steps
        out, _ = self.rnn(x, h0)

        # Use the output of the last time step for prediction by applying the fully connected layer
        out = self.fc(out[:, -1, :])  # out[:, -1, :] extracts the last time step output
        return out

    def predict(self, maze: torch.Tensor) -> torch.Tensor:
        return self.forward(maze)

# -------------------------------
# GRU Model Definition
# -------------------------------
class MazeGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MazeGRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch_size, seq_length, input_size)
        # Initialize the hidden state (h0) with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)

        # Pass the input sequence (x) and hidden state (h0) through the GRU layer
        # Output `out` contains the GRU outputs for all time steps
        out, _ = self.gru(x, h0)

        # Use the output of the last time step for prediction by applying the fully connected layer
        out = self.fc(out[:, -1, :])  # Extract the last time step's output
        return out

    def predict(self, maze: torch.Tensor) -> torch.Tensor:
        return self.forward(maze)

# -------------------------------
# LSTM Model Definition
# -------------------------------
class MazeLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MazeLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch_size, seq_length, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

    def predict(self, maze: torch.Tensor) -> torch.Tensor:
        return self.forward(maze)

# -------------------------------
# Custom Dataset for Training Samples
# -------------------------------
class MazeTrainingDataset(Dataset):
    def __init__(self, data):
        """
        data: a list of tuples (local_context, target_action)
        local_context: list of 4 values
        target_action: integer (0-3)
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        local_context, target_action = self.data[idx]
        # Convert local_context to tensor; unsqueeze later to add seq_length dimension
        return torch.tensor(local_context, dtype=torch.float32), torch.tensor(target_action, dtype=torch.long)

# -------------------------------
# Generic Training Function
# -------------------------------
def train_model(model, dataloader, num_epochs=EPOCHS, learning_rate=0.001, device='cpu'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, targets in dataloader:
            # inputs: shape [batch_size, 4] -> add seq_length dimension: [batch_size, 1, 4]
            inputs = inputs.unsqueeze(1).to(device)  # seq_length is 1 here
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)  # outputs: [batch_size, output_size]
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
        avg_loss = epoch_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    return model

# -------------------------------
# Training Methods for Each Model
# -------------------------------
def train_rnn_model(dataset, input_size=4, hidden_size=16, num_layers=1, output_size=4,
                    num_epochs=EPOCHS, learning_rate=0.001, batch_size=32, device='cpu'):
    model = MazeRNN2Model(input_size, hidden_size, num_layers, output_size)
    train_ds = MazeTrainingDataset(dataset)
    dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    print("Training RNN model...")
    trained_model = train_model(model, dataloader, num_epochs, learning_rate, device)
    return trained_model

def train_gru_model(dataset, input_size=4, hidden_size=16, num_layers=1, output_size=4,
                    num_epochs=EPOCHS, learning_rate=0.001, batch_size=32, device='cpu'):
    model = MazeGRUModel(input_size, hidden_size, num_layers, output_size)
    train_ds = MazeTrainingDataset(dataset)
    dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    print("Training GRU model...")
    trained_model = train_model(model, dataloader, num_epochs, learning_rate, device)
    return trained_model

def train_lstm_model(dataset, input_size=4, hidden_size=16, num_layers=1, output_size=4,
                     num_epochs=EPOCHS, learning_rate=0.001, batch_size=32, device='cpu'):
    model = MazeLSTMModel(input_size, hidden_size, num_layers, output_size)
    train_ds = MazeTrainingDataset(dataset)
    dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    print("Training LSTM model...")
    trained_model = train_model(model, dataloader, num_epochs, learning_rate, device)
    return trained_model

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

def main():
    TRAINING_MAZES_FILE = "input/training_mazes.pkl"
    TEST_MAZES_FILE = "input/mazes.pkl"

    # Instantiate the trainer with the file path for training mazes.
    trainer = RNN2MazeTrainer(TRAINING_MAZES_FILE)
    dataset = trainer.create_dataset()
    logging.info(f"Created {len(dataset)} training samples.")

    # Uncomment one of the following lines to train your desired model:
    trained_rnn = train_rnn_model(dataset, device='cpu')
    trained_gru = train_gru_model(dataset, device='cpu')
    trained_lstm = train_lstm_model(dataset, device='cpu')

    # Optionally, save the trained models
    torch.save(trained_rnn.state_dict(), f"{OUTPUT}rnn_model.pth")
    torch.save(trained_gru.state_dict(), "{OUTPUT}gru_model.pth")
    torch.save(trained_lstm.state_dict(), "{OUTPUT}lstm_model.pth")

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
