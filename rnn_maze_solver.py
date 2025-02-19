# rnn_maze_solver.py
import logging
from venv import logger
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from backtrack_maze_solver import BacktrackingMazeSolver
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


import utils
from maze import Maze  # Adjust the import path if necessary
from utils import load_mazes

# -------------------------------
# Hyperparameters and Configurations
# -------------------------------

# Define constants
INPUT_SIZE = 625  # Number of input features
HIDDEN_SIZE = 128  # Size of the hidden state in the RNN
NUM_LAYERS = 2  # Number of stacked RNN layers
OUTPUT_SIZE = 10  # Number of output classes
BATCH_SIZE = 32  # Batch size
LEARNING_RATE = 0.001  # Learning rate
EPOCHS = 20  # Number of training epochs
PADDING_VALUE = -1  # Padding value for labels


# Action mapping
ACTION_MAPPING = {
    'up': 0,
    'down': 1,
    'left': 2,
    'right': 3
}

# Reverse mapping for decoding
REVERSE_ACTION_MAPPING = {v: k for k, v in ACTION_MAPPING.items()}

logging.getLogger().setLevel(logging.INFO)

# -------------------------------
# Dataset Class
# -------------------------------

class MazeDataset(Dataset):
    """
    Represents a dataset for training machine learning models to solve mazes.

    The class processes a collection of mazes into a dataset format suitable for
    PyTorch. It preprocesses the maze grids, extracts solution paths, encodes
    the actions, and pads the action sequences to ensure uniformity of shape.

    :ivar inputs: Preprocessed maze input data represented as flattened 1D arrays.
    :type inputs: numpy.ndarray
    :ivar labels: Solution paths corresponding to each maze input, stored as a list
        of action sequences.
    :type labels: list
    :ivar encoded_labels: Action sequences encoded into numerical format for
        model training.
    :type encoded_labels: list
    :ivar padded_labels: Padded action sequences with numerical encoding, containing
        sequences of uniform length.
    :type padded_labels: torch.Tensor
    :ivar label_lengths: List of original lengths of the action sequences before padding.
    :type label_lengths: list
    """
    def __init__(self, mazes: list[Maze]):
        self.inputs, self.labels = self.preprocess_mazes(mazes)
        self.encoded_labels = self.encode_actions(self.labels)
        self.padded_labels, self.label_lengths = self.pad_labels(self.encoded_labels)

    def preprocess_mazes(self, training_mazes: list[Maze]):
        """
        Preprocesses a list of mazes by padding the grids to the largest dimensions
        within the list and transforming them into flattened input arrays suitable for
        training machine learning models. The function also extracts solution paths
        of mazes as labels corresponding to input data.

        :param mazes: A list of maze objects, where each maze contains a 2D grid and
            provides a method to fetch its solution as a list of actions.
        :type mazes: list of Maze
        :return: A tuple containing two elements:
            1. A numpy array of preprocessed maze inputs with padded grids flattened into
               1D arrays.
            2. A corresponding list of labels representing the solution paths for the
               input mazes.
        :rtype: tuple (numpy.ndarray, list)
        """

        logger.info("Preprocessing mazes for training.")
        # Determine maximum dimensions
        max_rows = max(maze.shape[0] for maze in training_mazes)
        max_cols = max(maze.shape[1] for maze in training_mazes)

        inputs = []
        labels = []

        for i, maze_matrix in enumerate(training_mazes):
            training_maze = Maze(maze_matrix)
            if training_maze.self_test():
                #compute solution
                solver = BacktrackingMazeSolver(training_maze)
                solution = solver.solve()
                training_maze.set_solution(solution)

                grid = training_maze.grid
                rows, cols = grid.shape

                # Padding the grid
                padded_grid = np.zeros((max_rows, max_cols))
                padded_grid[:rows, :cols] = grid

                inputs.append(padded_grid.flatten())

                # Extract the solution path as actions
                exit_path  = training_maze.get_solution()
                solution = self.coordinates_to_actions(exit_path)
                labels.append(solution)
            else:
                logger.warning("Test maze failed self-test.")

        inputs = np.array(inputs, dtype=np.float32)

        return inputs, labels

    def coordinates_to_actions(self, coordinate_path):
        """
        Convert a list of coordinates to a list of actions.

        Parameters:
            coordinate_path (list of tuple): List of (row, col) tuples representing the path.

        Returns:
            list of str: List of actions ('up', 'down', 'left', 'right').
        """
        actions = []
        for i in range(1, len(coordinate_path)):
            prev_row, prev_col = coordinate_path[i - 1]
            current_row, current_col = coordinate_path[i]

            if current_row == prev_row - 1 and current_col == prev_col:
                actions.append('up')
            elif current_row == prev_row + 1 and current_col == prev_col:
                actions.append('down')
            elif current_row == prev_row and current_col == prev_col - 1:
                actions.append('left')
            elif current_row == prev_row and current_col == prev_col + 1:
                actions.append('right')
            else:
                raise ValueError(f"Invalid movement from {coordinate_path[i - 1]} to {coordinate_path[i]}")

        return actions

    def encode_actions(self, action_sequences):
        """
        Encode action sequences into numerical format.
        """
        encoded = []
        for actions in action_sequences:
            encoded_seq = [ACTION_MAPPING[action] for action in actions]
            encoded.append(encoded_seq)
        return encoded

    def pad_labels(self, encoded_labels):
        """
        Pad action sequences to the same length.
        """
        label_lengths = [len(seq) for seq in encoded_labels]
        max_length = max(label_lengths)

        padded = []
        for seq in encoded_labels:
            padded_seq = seq + [PADDING_VALUE] * (max_length - len(seq))
            padded.append(padded_seq)

        padded = torch.tensor(padded, dtype=torch.long)
        return padded, label_lengths

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_tensor = torch.tensor(self.inputs[idx], dtype=torch.float32)
        label_tensor = self.padded_labels[idx]
        length = self.label_lengths[idx]
        return input_tensor, label_tensor, length


# -------------------------------
# RNN Model Definition
# -------------------------------

class MazeRNNModel(nn.Module):
    """
    Represents a Recurrent Neural Network (RNN) model for solving Maze-related
    tasks or other sequence modeling problems.

    This class is built using PyTorch's nn.Module and includes defined layers for input
    transformation, RNN-based processing, and output generation. It is customizable
    based on input size, hidden layer size, number of RNN layers, and output dimensions.
    The model leverages GRU (Gated Recurrent Unit) for recurrent operations and can
    be adapted for various sequence or time-series data tasks.

    :ivar hidden_size: The size of the hidden state in the RNN layer.
    :type hidden_size: int
    :ivar num_layers: The number of stacked RNN layers in the model.
    :type num_layers: int
    :ivar fc1: Fully connected linear layer to transform input into hidden size space.
    :type fc1: torch.nn.Linear
    :ivar relu: ReLU activation layer applied after the first linear transformation.
    :type relu: torch.nn.ReLU
    :ivar rnn: GRU-based RNN layer with defined hidden size and layers.
    :type rnn: torch.nn.GRU
    :ivar fc2: Fully connected linear layer to transform RNN outputs to the desired
        output size.
    :type fc2: torch.nn.Linear
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MazeRNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Assuming x is of shape (batch_size, input_size)
        out = self.fc1(x)
        out = self.relu(out)
        out = out.unsqueeze(1)  # Add sequence dimension: (batch_size, 1, hidden_size)
        out, _ = self.rnn(out)  # Output shape: (batch_size, seq_length=1, hidden_size)
        out = self.fc2(out.squeeze(1))  # Final output: (batch_size, output_size)
        return out


# -------------------------------
# Helper Functions
# -------------------------------

def encode_actions(action_sequences):
    """
    Encodes a list of action sequences into their corresponding integer
    representations using a predefined mapping.

    Each action in a sequence is mapped to its corresponding integer value
    as per the `ACTION_MAPPING` dictionary. The method processes multiple
    sequences of actions and outputs their encoded representations.

    :param action_sequences: A list of lists, where each inner list contains
                              actions (strings) representing a sequence.
    :type action_sequences: list[list[str]]
    :return: A list of lists containing the encoded integer representations of
             the input action sequences.
    :rtype: list[list[int]]
    """
    encoded = []
    for actions in action_sequences:
        encoded_seq = [ACTION_MAPPING[action] for action in actions]
        encoded.append(encoded_seq)
    return encoded


def decode_actions(encoded_actions):
    """
    Decodes a list of encoded actions into their corresponding human-readable
    strings using a predefined mapping. For each encoded action, the function
    uses `REVERSE_ACTION_MAPPING` to retrieve its decoded value. If an action
    is not present in the mapping, it is ignored.

    :param encoded_actions: A list of encoded actions to be decoded.
    :type encoded_actions: list
    :return: A list of decoded human-readable actions.
    :rtype: list
    """
    decoded = [REVERSE_ACTION_MAPPING[action] for action in encoded_actions if action in REVERSE_ACTION_MAPPING]
    return decoded


def collate_fn(batch):
    """
    Combines a list of data points into a single batch for processing in a DataLoader.

    This function is typically used as the `collate_fn` in a PyTorch DataLoader to process
    batches of sequences with corresponding labels and lengths. It collects inputs,
    labels, and lengths from the individual data points and prepares them as tensors for
    further usage in model training or evaluation.

    :param batch: List of tuples where each tuple contains a sequence input (torch.Tensor),
                  its label (torch.Tensor), and the sequence length (int).
    :type batch: list[tuple[torch.Tensor, torch.Tensor, int]]
    :return: A tuple containing batched input tensors, batched labels, and a tensor of
             sequence lengths for the batch.
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    inputs, labels, lengths = zip(*batch)
    inputs = torch.stack(inputs)
    labels = torch.stack(labels)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return inputs, labels, lengths


# -------------------------------
# Training Function
# -------------------------------

def train_model(model, dataloader, criterion, optimizer, epochs, device):
    """
    Trains the given model over a specified number of epochs using the provided
    dataloader, loss function, and optimizer. The training process involves computing
    the loss, backpropagation, and updating the model weights through the optimizer.
    The function outputs the loss for each epoch.

    :param model: The neural network model to be trained.
    :param dataloader: Iterable dataloader providing batches of input data and
        corresponding labels.
    :param criterion: Loss function to evaluate the model's predictions against
        the ground truth.
    :param optimizer: Optimization algorithm to adjust the model's parameters based
        on the computed gradients.
    :param epochs: Total number of epochs for which the training will run.
    :param device: Specifies whether to run computations on 'cpu' or 'cuda'.
    :return: None
    """

    logging.info("Starting training.")
    model.train()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for inputs, labels, lengths in dataloader:
            # Retrieve batch size from inputs
            batch_size = inputs.size(0)

            # Ensure inputs have shape (batch_size, sequence_length, input_size)
            inputs = inputs.to(device)  # Shape: [32, 140, 625]
            labels = labels.to(device)  # Shape: [32, 140]

            optimizer.zero_grad()

            outputs = model(inputs)  # Shape: [32, 140, OUTPUT_SIZE]
            logging.info(f'Outputs shape before reshaping: {outputs.shape}')
            logging.info(f'Labels shape before reshaping: {labels.shape}')

            # Ensure labels are of shape [32]
            if labels.dim() > 1:
                labels = labels.argmax(dim=2)


            # Reshape outputs and labels for loss computation
            outputs = outputs.view(-1, outputs.size(-1))  # [32*140, OUTPUT_SIZE]
            labels = labels.view(-1)  # [32*140]

            logging.info(f'Outputs shape after reshaping: {outputs.shape}')
            logging.info(f'Labels shape after reshaping: {labels.shape}')


            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch [{epoch}/{epochs}], Loss: {avg_loss:.4f}')

    logging.info("Training completed.")


# -------------------------------
# Solving Function
# -------------------------------

def solve_maze_with_model(model, maze_instance, device):
    """
    Solves a maze problem using a pre-trained machine learning model. This function performs
    the necessary preprocessing of the maze data to align with the model's expected input
    format, predicts the next action using the model, and applies the action to the maze.
    The final solution path is obtained from the maze instance.

    The method ensures no changes are made to the model during inference by using evaluation
    mode and disabling gradient computation.

    :param model: The pre-trained machine learning model used for predicting actions in the maze.
    :param maze_instance: The instance of the maze to be solved. It provides access
        to the grid structure and the ability to perform actions within the maze.
    :param device: The computation device (e.g., CPU or GPU) where the model
        and data should be processed.
    :return: The solution of the maze, represented as the path taken to reach the endpoint.
    :rtype: List[Tuple[int, int]]
    """
    model.eval()

    with torch.no_grad():
        # Preprocess the maze instance
        grid = maze_instance.grid
        rows, cols = grid.shape

        # Determine maximum dimensions (assuming it's consistent with training)
        max_rows = model.input_size  # Adjust if necessary
        max_cols = model.input_size // max_rows  # Example calculation

        padded_grid = np.zeros((max_rows, max_cols))
        padded_grid[:rows, :cols] = grid
        input_tensor = torch.tensor(padded_grid.flatten(), dtype=torch.float32).unsqueeze(0).to(
            device)  # Shape: (1, input_size)

        # Forward pass
        output = model(input_tensor)  # Shape: (1, output_size)
        predicted_action = torch.argmax(output, dim=1).item()

        action = REVERSE_ACTION_MAPPING.get(predicted_action, None)
        if action:
            maze_instance.move(action)

    return maze_instance.get_solution()

# -------------------------------
# Main Execution
# -------------------------------

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

    # Create dataset and dataloader
    dataset = MazeDataset(training_mazes)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    for inputs, labels, lengths in dataloader:
        logging.info(f'Inputs shape: {inputs.shape}')  # Expected: [32, 140, 625]
        logging.info(f'Labels shape: {labels.shape}')  # Expected: [32, 140]
        break

    # Initialize the model, loss function, and optimizer
    model = MazeRNNModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PADDING_VALUE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    train_model(model, dataloader, criterion, optimizer, EPOCHS, device)

    # Save the trained model
    torch.save(model.state_dict(), 'output/maze_rnn_model.pth')
    logger.info("Model trained and saved successfully.")

    # Example of solving a new maze
    mazes = load_mazes("input/mazes.pkl")
    for i, maze in mazes:
    # Initialize and configure your test_maze as needed
        if maze.self_test():
            solution = solve_maze_with_model(model, maze, device)
            logger.info("Solution Path:", solution)
            maze.plot_maze(self, show_path=True, show_solution=False, show_position=False)  # Assuming this method visualizes the maze and solution
        else:
            logger.warning("Test maze failed self-test.")


if __name__ == "__main__":
    main()
