# maze_trainer.py
# MazeTrainingDataset and RNN2MazeTrainer

import csv
import logging
import os
import pickle
from configparser import ConfigParser

import numpy as np
import torch
import wandb
from numpy.f2py.auxfuncs import throw_error
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils
from classical_algorithms.backtrack_maze_solver import BacktrackingMazeSolver
from classical_algorithms.optimized_backtrack_maze_solver import OptimizedBacktrackingMazeSolver
from classical_algorithms.pladge_maze_solver import PledgeMazeSolver
from maze import Maze
# Import the unified model
from model import MazeRecurrentModel

WALL = 1
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRECTION_TO_ACTION = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}

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
LOSS_DATA = config.get("FILES", "LOSS_DATA", fallback="loss_data.csv")
LOSS_FILE = f"{OUTPUT}{LOSS_DATA}"

training_mazes = config.get("FILES", "TRAINING_MAZES", fallback="mazes.pkl")
validation_mazes = config.get("FILES", "VALIDATION_MAZES", fallback="mazes.pkl")

TRAINING_MAZES_FILE = f"{INPUT}{training_mazes}"
VALIDATION_MAZES_FILE = f"{INPUT}{training_mazes}"
wandb_enabled = config.getboolean("MONITORING", "wandb", fallback=True)

# Mapping of available solver names to their classes
solver_mapping = {
    'BacktrackingMazeSolver': BacktrackingMazeSolver,
    'OptimizedBacktrackingMazeSolver': OptimizedBacktrackingMazeSolver,
    'PledgeMazeSolver': PledgeMazeSolver
}


# -------------------------------
# Custom Dataset for Training Samples
# -------------------------------
class MazeTrainingDataset(Dataset):
    """
    Custom dataset for training the maze navigation model.
    This class processes the data to return context about the current maze cell,
    the relative position, the target action, and the normalized step number.
    """

    def __init__(self, data):
        """
        Initializes the dataset.

        Args:
            data (list): List of tuples containing:
                - local_context (list): State of maze cells around the current position.
                - target_action (int): Action to take (0: up, 1: down, 2: left, 3: right).
                - step_number (int): Step number in the solution path.
        """
        self.data = data
        # Get the maximum number of steps allowed for a maze solution from the configuration.
        max_steps = config.getint("DEFAULT", "max_steps")
        self.max_steps = max_steps

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        local_context, relative_position, target_action, step_number = self.data[idx]
        step_number_normalized = step_number / self.max_steps
        return (np.array(local_context, dtype=np.float32),
                np.array(relative_position, dtype=np.float32),
                target_action,
                step_number_normalized)


class ValidationDataset(MazeTrainingDataset):
    """
    Subclass of MazeTrainingDataset used for handling validation datasets.
    Ensures that the validation dataset is not empty and calculates
    the maximum steps required for normalized step calculations.
    """

    def __init__(self, data):
        """
        Initializes the validation dataset.

        Args:
            data (list): List of tuples containing:
                - local_context (list): State of maze cells around the current position.
                - target_action (int): Action to take (0: up, 1: down, 2: left, 3: right).
                - step_number (int): Step number in the solution path.
        """
        super().__init__(data)  # Call the parent dataset initializations
        self.data = data
        if len(data) == 0:
            # Validation datasets must not be empty, raise an error if empty.
            self.max_steps = 0
            raise ValueError("Validation dataset is empty.")
        else:
            # Calculate the maximum step count from the dataset.
            max_steps = max(sample[2] for sample in data)
            self.max_steps = max_steps

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        local_context, relative_position, target_action, step_number = self.data[idx]
        step_number_normalized = step_number / self.max_steps if self.max_steps != 0 else 0
        return (np.array(local_context, dtype=np.float32),
                np.array(relative_position, dtype=np.float32),
                target_action,
                step_number_normalized)

# -------------------------------
# Training Utilities (Imitation Learning Setup)
# -------------------------------
class RNN2MazeTrainer:
    """
    Handles training and validation data preparation for maze navigation models.
    The class processes training mazes, applies solutions, and prepares the datasets
    for recurrent neural network models.
    """

    def __init__(self, training_file_path="/input/training_mazes.pkl",
                 validation_file_path="/input/validation_mazes.pkl"):
        """
        Initializes the trainer by loading and processing training and validation mazes.

        Args:
            training_file_path (str): Path to the training mazes pickle file.
            validation_file_path (str): Path to the validation mazes pickle file.
        """
        self.training_file_path = training_file_path
        self.validation_file_path = validation_file_path

        # Load the configuration file to retrieve parameters for training.
        config = ConfigParser()
        config.read("config.properties")

        # Check if development mode is enabled, reducing the training dataset size.
        if config.getboolean("DEFAULT", "development_mode", fallback=False):
            logging.warning("Development mode is enabled. Training mazes will be loaded from the development folder.")
            training_samples = 10
        else:
            # Load the total number of training samples allowed.
            training_samples = config.getint("DEFAULT", "training_samples", fallback=100000)

        if training_samples < 10:
            logging.error("Training samples must be at least 10.")
            throw_error(
                "Training samples must be at least 10 "
                "Please adjust the training_samples parameter in the config file.")

        self.training_mazes = self._load_and_process_training_mazes(training_file_path,training_samples)
        self.validation_mazes = self._load_and_process_training_mazes(validation_file_path,training_samples//10)

    def _load_and_process_training_mazes(self, path, training_samples):
        """
        Loads and processes training mazes from the specified file path.

        Parameters:
            path (str): Path to the training mazes file.
            training_samples (int): Maximum number of mazes to process.

        Returns:
            List of processed training mazes.
        """
        training_mazes = self._load_mazes_safe(file_path=path)
        solved_training_mazes = []

        # Using tqdm to display progress bar - slicing the list to ensure we only process training_samples
        for i, maze_data in enumerate(tqdm(training_mazes[:training_samples], desc="Load training mazes")):
            try:
                solved_training_mazes.append(self._process_maze(maze_data, i))
            except Exception as e:
                logging.error(f"Failed to process maze {i + 1}: {str(e)}")
                raise RuntimeError(f"Processing maze {i + 1} failed.") from e
        return solved_training_mazes

    @staticmethod
    def _load_mazes_safe(file_path):
        try:
            return utils.load_mazes(file_path)
        except FileNotFoundError as e:
            logging.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"Could not find the specified file: {file_path}") from e
        except Exception as e:
            logging.error(f"Error loading mazes: {str(e)}")
            raise RuntimeError("Maze loading failed.") from e

    @staticmethod
    def _process_maze(data, index):
        maze = Maze(data)
        if not maze.self_test():
            logging.warning(f"Maze {index + 1} failed validation.")

        # Retrieve the solver name from the configuration (defaulting to BacktrackingMazeSolver)
        solver_obj = config.get("DEFAULT", "solver", fallback="BacktrackingMazeSolver")
        solver_cls = solver_mapping.get(solver_obj)
        if not solver_cls:
            raise RuntimeError(f"Solver '{solver_obj}' not found. Available solvers: {list(solver_mapping.keys())}")

        # Instantiate and use the solver.
        solver = solver_cls(maze)

        maze.set_solution(solver.solve())
        return maze

    def create_dataset(self):
        """
        Constructs a dataset for training a maze-navigating model.
        """
        logging.info("Creating dataset.")
        dataset = []
        for maze in tqdm(self.training_mazes, desc="Creating training dataset"):
            solution = maze.get_solution()
            start_position = maze.start_position
            for i, (current_pos, next_pos) in enumerate(zip(solution[:-1], solution[1:])):
                steps_number = i
                local_context = self._compute_local_context(maze, current_pos, DIRECTIONS)
                # Calculate relative position as (dx, dy)
                relative_position = (current_pos[0] - start_position[0], current_pos[1] - start_position[1])
                move_delta = (next_pos[0] - current_pos[0], next_pos[1] - current_pos[1])
                if move_delta not in DIRECTION_TO_ACTION:
                    raise KeyError(f"Invalid move delta: {move_delta}")
                target_action = DIRECTION_TO_ACTION[move_delta]
                # Append the tuple with the additional relative_position information
                dataset.append((local_context, relative_position, target_action, steps_number))

        validation_dataset = []
        for maze in tqdm(self.validation_mazes, desc="Creating validation dataset"):
            solution = maze.get_solution()
            if maze.self_test():  # avoid validating on mazes with no solution
                for i, (current_pos, next_pos) in enumerate(zip(solution[:-1], solution[1:])):
                    steps_number = i
                    local_context = self._compute_local_context(maze, current_pos, DIRECTIONS)
                    relative_position = (current_pos[0] - start_position[0], current_pos[1] - start_position[1])
                    move_delta = (next_pos[0] - current_pos[0], next_pos[1] - current_pos[1])
                    if move_delta not in DIRECTION_TO_ACTION:
                        raise KeyError(f"Invalid move delta: {move_delta}")
                    target_action = DIRECTION_TO_ACTION[move_delta]
                    validation_dataset.append((local_context, relative_position, target_action, steps_number))
            else:
                logging.warning(f"Maze failed validation.")
        return dataset, validation_dataset

    def _compute_local_context(self, maze, position, directions):
        """
        Generates the local context around a position in the maze.
    
        Args:
            maze (Maze): The current maze grid.
            position (tuple): Current position coordinates (row, column).
            directions (list): Possible movement directions in the maze.
    
        Returns:
            list: Values of the maze cells around the position, with invalid
                  cells marked as WALL.
        """
        r, c = position
        directions = np.array(directions)

        # Compute the coordinates of neighboring cells for each direction.
        neighbors = directions + np.array([r, c])

        # Create a mask to retain only valid neighbors within the maze boundaries.
        mask = np.apply_along_axis(maze.is_within_bounds, 1, neighbors)

        # Initialize local context array, marking all as walls by default.
        local_context = np.full(len(directions), WALL, dtype=maze.grid.dtype)

        # For valid neighbors, update the local context with their values.
        valid_neighbors = neighbors[mask]
        local_context[mask] = maze.grid[valid_neighbors[:, 0], valid_neighbors[:, 1]]

        return local_context.tolist()


# -------------------------------
# Model Training Function
# -------------------------------
def load_model_state(model, model_path, old_prefix, new_prefix):
    """
    Helper function to remap keys from the old state_dict prefix to the new one.
    """
    state_dict = torch.load(model_path)
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(old_prefix):
            new_key = new_prefix + key[len(old_prefix):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    model.load_state_dict(new_state_dict)


def get_models():
    """
    Loads models from the specified paths, or trainmodels 
    :return: list of model paths and trained models.
    """
    config = ConfigParser()
    config.read("config.properties")

    # Read the allowed models from the config file. Expected format: "GRU, LSTM, RNN"
    models_config = config.get("DEFAULT", "models", fallback="GRU,LSTM,RNN")
    allowed_models = [model.strip().upper() for model in models_config.split(",")]

    retrain_model = config.getboolean("GRU", "retrain_model", fallback=False)
    if retrain_model:
        models = train_models(allowed_models)
    else:
        models = load_models(allowed_models)

    return models


def load_models(allowed_models):
    """
    Loads models from the specified paths.
    :param allowed_models: List of model types to load (e.g., ["RNN", "GRU", "LSTM"])
    :return: List of loaded models with weights from the input folder
    """

    models = []
    for model_name in allowed_models:
        if model_name == "GRU":
            gru_model = MazeRecurrentModel(
                mode_type="GRU",
                input_size=config.getint("GRU", "input_size", fallback=7),
                hidden_size=config.getint("GRU", "hidden_size"),
            )
            model_path = os.path.join(INPUT, GRU_MODEL_PATH)
            if os.path.exists(model_path):
                gru_model.load_state_dict(torch.load(model_path))
            models.append(("GRU", gru_model))
        elif model_name == "RNN":
            rnn_model = MazeRecurrentModel(
                mode_type="RNN",
                input_size=config.getint("RNN", "input_size", fallback=7),
                hidden_size=config.getint("RNN", "hidden_size"),
            )
            model_path = os.path.join(INPUT, RNN_MODEL_PATH)
            if os.path.exists(model_path):
                rnn_model.load_state_dict(torch.load(model_path))
            models.append(("RNN", rnn_model))
        elif model_name == "LSTM":
            lstm_model = MazeRecurrentModel(
                mode_type="LSTM",
                input_size=config.getint("LSTM", "input_size", fallback=7),
                hidden_size=config.getint("LSTM", "hidden_size"),
            )
            model_path = os.path.join(INPUT, LSTM_MODEL_PATH)
            if os.path.exists(model_path):
                lstm_model.load_state_dict(torch.load(model_path))
            models.append(("RNN", lstm_model))

    return models


def collate_fn(batch):
    return tuple(torch.as_tensor(x) for x in zip(*batch))

def train_models(allowed_models):
    logging.debug("Starting training.")
    trained_models = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = config.getint("DEFAULT", "batch_size", fallback=128)

    # Delete previous tensorboard files.
    try:
        with open(LOSS_FILE, "w", newline="") as f:
            loss_writer = csv.writer(f)
            loss_writer.writerow(
                ["model", "epoch", "training_loss", "validation_loss", "accuracy", "validation_accuracy",
                 "time", "time_per_step", "cpu_load", "gpu_load", "ram_usage"])
    except Exception as e:
        logging.error(f"Error setting up loss file: {e}")

    tensorboard_data_sever = SummaryWriter(log_dir=f"{OUTPUT}tensorboard_data")

    trainer = RNN2MazeTrainer(TRAINING_MAZES_FILE, VALIDATION_MAZES_FILE)

    # -------------------------------
    # Added caching for dataset creation
    # -------------------------------
    cache_path = os.path.join(OUTPUT, "dataset_cache.pkl")
    if os.path.exists(cache_path) and not config.getboolean("DEFAULT", "use_dataset_cache=True", fallback=False):
        with open(cache_path, "rb") as f:
            dataset, validation_dataset = pickle.load(f)
            logging.info(f"Loading dataset from cache. Length:{len(dataset)}")
    else:
        dataset, validation_dataset = trainer.create_dataset()
        with open(cache_path, "wb") as f:
            pickle.dump((dataset, validation_dataset), f)
    logging.info(f"Created {len(dataset)} training samples.")

    train_ds = MazeTrainingDataset(dataset)
    validation_ds = ValidationDataset(validation_dataset)

    # -------------------------------
    # Adjust DataLoader worker count based on environment.
    # -------------------------------
    try:
        # Starting with a safe baseline
        num_workers = config.getint("DEFAULT", "dataloader_workers", fallback=0)
        if num_workers:
            logging.info(f"Using {num_workers} workers from config")
        else:
            if device.type == 'cuda':
                max_workers = config.getint("DEFAULT", "max_num_workers", fallback=2)
                num_workers = min(max_workers, os.cpu_count() or 1)
                logging.info(f"Using {num_workers} workers for GPU training")
            else:
                num_workers = max(1, (os.cpu_count() or 1) // 4)
                logging.info(f"Using {num_workers} workers for CPU training")

        dataloader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2,
            collate_fn=collate_fn
        )
        logging.info(f"Number of workers: {num_workers}")
        logging.info(f'Using device: {device}')

    except (MemoryError, RuntimeError) as e:
        logging.warning(f"Failed to create DataLoader with {num_workers} workers: {str(e)}")
        logging.info("Falling back to single-process loading")
        dataloader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )

    # (Remaining training loop continues below as before ...)
    for model_name in allowed_models:
        if model_name == "GRU":
            gru_model = _train_gru_model(
                device=device,
                dataloader=dataloader,
                validation_ds=validation_ds,
                tensorboard_data_sever=tensorboard_data_sever
            )
            trained_models.append(("GRU", gru_model))
        elif model_name == "LSTM":
            lstm_model = _train_lstm_model(
                device=device,
                dataloader=dataloader,
                validation_ds=validation_ds,
                tensorboard_data_sever=tensorboard_data_sever
            )
            trained_models.append(("LSTM", lstm_model))
        elif model_name == "RNN":
            rnn_model = _train_rnn_model(
                device=device,
                dataloader=dataloader,
                validation_ds=validation_ds,
                tensorboard_data_sever=tensorboard_data_sever
            )
            trained_models.append(("RNN", rnn_model))
        else:
            logging.warning(f"Model {model_name} is not recognized and will be skipped.")

    tensorboard_data_sever.close()
    logging.info("Training complete.")
    return trained_models

def _train_rnn_model(device, dataloader, validation_ds , tensorboard_data_sever) -> MazeRecurrentModel:
    # RNN Model Training
    rnn_model = MazeRecurrentModel(
        mode_type="RNN",
        input_size=config.getint("RNN", "input_size", fallback=7),
        hidden_size=config.getint("RNN", "hidden_size"),
        num_layers=config.getint("RNN", "num_layers"),
        output_size=config.getint("RNN", "output_size", fallback=4),
    )
    rnn_model.to(device)
    if wandb_enabled:
        wandb.watch(rnn_model, log="all", log_freq=1000)

    logging.info("Training RNN model")
    rnn_model = rnn_model.train_model(
        dataloader=dataloader,
        val_loader=validation_ds,
        num_epochs=config.getint("RNN", "num_epochs"),
        learning_rate=config.getfloat("RNN", "learning_rate"),
        weight_decay=config.getfloat("RNN", "weight_decay"),
        device=device,
        tensorboard_writer=tensorboard_data_sever
    )
    loss = rnn_model.last_loss
    logging.info(f"Done training RNN model. Loss {loss:.4f}")
    torch.save(rnn_model.state_dict(), RNN_MODEL_PATH)
    logging.info("Saved RNN model")
    if wandb_enabled:
        wandb.log({"RNN_final_loss": loss})
    tensorboard_data_sever.add_scalar("Loss/RNN_final_loss", loss)

    return rnn_model

def _train_gru_model(device, dataloader, validation_ds , tensorboard_data_sever) -> MazeRecurrentModel:
    #GRU model training
    gru_model = MazeRecurrentModel(
        mode_type="GRU",
        input_size=config.getint("GRU", "input_size", fallback=7),
        hidden_size=config.getint("GRU", "hidden_size"),
        num_layers=config.getint("GRU", "num_layers"),
        output_size=config.getint("GRU", "output_size", fallback=4),
    )
    gru_model.to(device)
    if wandb_enabled:
        wandb.watch(gru_model, log="all", log_freq=1000)

    logging.info("Training GRU model")
    gru_model = gru_model.train_model(
        dataloader=dataloader,
        val_loader=validation_ds,
        num_epochs=config.getint("GRU", "num_epochs"),
        learning_rate=config.getfloat("GRU", "learning_rate"),
        weight_decay=config.getfloat("GRU", "weight_decay"),
        device=device,
        tensorboard_writer=tensorboard_data_sever
    )
    loss = gru_model.last_loss
    torch.save(gru_model.state_dict(), GRU_MODEL_PATH)
    logging.info(f"Done training GRU model. Loss {loss:.4f}")
    if wandb_enabled:
        wandb.log({"GRU_final_loss": loss})
    tensorboard_data_sever.add_scalar("Loss/GRU_final_loss", loss)

    return gru_model

def _train_lstm_model(device, dataloader, validation_ds, tensorboard_data_sever) -> MazeRecurrentModel:
    #LSTM Model training
    lstm_model = MazeRecurrentModel(
        mode_type="LSTM",
        input_size=config.getint("LSTM", "input_size", fallback=7),
        hidden_size=config.getint("LSTM", "hidden_size"),
        num_layers=config.getint("LSTM", "num_layers"),
        output_size=config.getint("LSTM", "output_size", fallback=4),
    )
    lstm_model.to(device)
    if wandb_enabled:
        wandb.watch(lstm_model, log="all", log_freq=1000)

    logging.info("Training LSTM model")
    lstm_model = lstm_model.train_model(
        dataloader=dataloader,
        val_loader=validation_ds,
        num_epochs=config.getint("LSTM", "num_epochs"),
        learning_rate=config.getfloat("LSTM", "learning_rate"),
        weight_decay=config.getfloat("LSTM", "weight_decay"),
        device=device,
        tensorboard_writer=tensorboard_data_sever
    )
    loss = lstm_model.last_loss
    torch.save(lstm_model.state_dict(), LSTM_MODEL_PATH)
    logging.info(f"Done training LSTM model. Loss {loss:.4f}")
    if wandb_enabled:
        wandb.log({"LSTM_final_loss": loss})
    tensorboard_data_sever.add_scalar("Loss/LSTM_final_loss", loss)

    return lstm_model