# maze_trainer.py
# MazeTrainingDataset and RNN2MazeTrainer

import json
import logging
import os
import pickle
from configparser import ConfigParser

import numpy as np
import pandas as pd
import torch
import wandb
from numpy.f2py.auxfuncs import throw_error
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils
from classical_algorithms.backtrack_maze_solver import BacktrackingMazeSolver
from classical_algorithms.bfs_maze_solver import BFSMazeSolver
from classical_algorithms.grpah_maze_solver import AStarMazeSolver
from classical_algorithms.optimized_backtrack_maze_solver import OptimizedBacktrackingMazeSolver
from classical_algorithms.pladge_maze_solver import PledgeMazeSolver
from maze import Maze
# Import the unified model
from rnn.model import MazeRecurrentModel
from utils import setup_logging, profile_method, clean_outupt_folder

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
    'PledgeMazeSolver': PledgeMazeSolver,
    'BFSMazeSolver': BFSMazeSolver,
    'AStarMazeSolver': AStarMazeSolver,
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
        self.samples = []

        # Case A: already-windowed data
        if len(data) > 0 and isinstance(data[0], tuple) and isinstance(data[0][0], np.ndarray):
            for inputs_np, targets_np in data:
                # convert numpy arrays to tensors
                self.samples.append((
                    torch.tensor(inputs_np, dtype=torch.float32),
                    torch.tensor(targets_np, dtype=torch.int64)
                ))
        else:
            # Case B: raw sequences → sliding‑window
            seq_len = config.getint("DEFAULT", "max_steps", fallback=40)
            self.max_steps = None

            for sequence in data:
                # each sequence is a list of tuples:
                #   (local_context:list[4], relative_position:tuple[2], target_action:int, step_number:int)
                if len(sequence) >= seq_len:
                    for i in range(len(sequence) - seq_len + 1):
                        window = sequence[i: i + seq_len]
                        inputs_list, targets_list = [], []

                        for local_ctx, rel_pos, action, step_num in window:
                            # establish max_steps for normalization
                            if self.max_steps is None:
                                # pick last step_number of first window (or at least 1)
                                self.max_steps = window[-1][-1] or 1

                            step_norm = step_num / self.max_steps
                            feature = np.array(local_ctx + list(rel_pos) + [step_norm],
                                               dtype=np.float32)
                            inputs_list.append(feature)
                            targets_list.append(action)

                        # stack into arrays of shape (seq_len, input_size)
                        inputs_np = np.stack(inputs_list)
                        targets_np = np.array(targets_list, dtype=np.int64)

                        self.samples.append((inputs_np, targets_np))

        if len(self.samples) == 0:
            raise RuntimeError("MazeTrainingDataset initialized with zero samples!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class ValidationDataset(Dataset):
    """
    Wraps either:
      A) pre-windowed List[ (inputs_np, targets_np) ],
      B) raw sequences to be windowed exactly like MazeTrainingDataset does.
    """
    def __init__(self, data):
        if not data:
            raise ValueError("Validation dataset is empty.")
        self.samples = []

        # A) already-windowed?
        if isinstance(data[0], tuple) and isinstance(data[0][0], np.ndarray):
            for inputs_np, targets_np in data:
                self.samples.append((
                    torch.tensor(inputs_np, dtype=torch.float32),
                    torch.tensor(targets_np, dtype=torch.int64)
                ))
        else:
            # B) fall back to sliding-window (identical to MazeTrainingDataset)
            seq_len = config.getint("DEFAULT", "max_steps", fallback=40)
            max_steps = None
            for sequence in data:
                if len(sequence) >= seq_len:
                    for i in range(len(sequence) - seq_len + 1):
                        window = sequence[i: i + seq_len]
                        inputs_list, targets_list = [], []
                        for local_ctx, rel_pos, action, step_num in window:
                            if max_steps is None:
                                max_steps = window[-1][-1] or 1
                            step_norm = step_num / max_steps
                            feat = np.array(local_ctx + list(rel_pos) + [step_norm],
                                            dtype=np.float32)
                            inputs_list.append(feat)
                            targets_list.append(action)
                        inp_np = np.stack(inputs_list)
                        tgt_np = np.array(targets_list, dtype=np.int64)
                        self.samples.append((inp_np, tgt_np))

        if not self.samples:
            raise RuntimeError("ValidationDataset initialized with zero samples!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class CurriculumSampler(Sampler):
    """
    A PyTorch sampler that enables curriculum learning by gradually increasing the difficulty
    of training samples over time.

    How it works:
    - Assumes the dataset is pre-sorted from easiest to hardest.
    - Divides the dataset into N "difficulty phases" (e.g., 10 chunks).
    - Starts training using only the first phase (e.g., easiest 10% of samples).
    - Every few epochs, it unlocks an additional phase (e.g., next 10% of samples).
    - Each epoch, it randomly samples within the currently unlocked portion.

    Example:
        - Dataset length: 1000 samples
        - phase_count = 10 → 100 samples per phase
        - epoch 0 → samples from 0:100
        - epoch 1 → samples from 0:200
        - epoch 2 → samples from 0:300
        ...

    Args:
        dataset (Dataset): The training dataset (must be sorted by difficulty).
        phase_count (int): Number of incremental curriculum phases (default: 10).
        unlock_every_n_epochs (int): How often to unlock a new phase (default: every 1 epoch).
    """

    def __init__(self, dataset, phase_count=10, unlock_every_n_epochs=1):
        """
        Args:
            dataset: The training dataset (already sorted by difficulty!).
            phase_count: Into how many difficulty chunks we divide the dataset (default: 10 → 10% increments).
            unlock_every_n_epochs: How often to unlock the next difficulty band.
        """
        self.dataset = dataset
        self.num_samples = len(dataset)
        self.phase_count = phase_count
        self.unlock_every = unlock_every_n_epochs
        self.current_epoch = 0

        # Precompute difficulty boundaries
        self.phase_size = self.num_samples // self.phase_count

    def set_epoch(self, epoch):
        self.current_epoch = epoch


    def __iter__(self):
        # Determine how many difficulty phases to include this epoch
        phases_unlocked = min(self.phase_count, (self.current_epoch // self.unlock_every) + 1)
        upper_bound = phases_unlocked * self.phase_size
        indices = np.arange(0, upper_bound)
        np.random.shuffle(indices)
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples

class RollingSubsetSampler(Sampler):
    """
    A Sampler that maintains a rolling subset of indices from a dataset.

    In each iteration (epoch), a specified `fraction` of the active indices
    are randomly selected and replaced with indices that were not part of
    the active set in the previous iteration. The total number of indices
    yielded in each iteration remains constant and equal to the size of the
    original dataset.

    This allows for gradual shifting of the data subset used per epoch,
    ensuring variety while potentially focusing training on a dynamic window
    of the full dataset over time.

    Args:
        dataset: The dataset to sample from.
        fraction (float): The fraction of indices to replace in each iteration.
                          Must be between 0 and 1. Defaults to 0.1 (10%).
    """
    def __init__(self, dataset, fraction=0.1):
        self.dataset = dataset
        self.total_indices = list(range(len(dataset)))
        self.fraction = fraction
        self.size = len(dataset)
        self.active_indices = set(np.random.choice(self.total_indices, size=self.size, replace=False))

    def __iter__(self):
        # Replace fraction of current active set
        num_replace = int(self.fraction * self.size)
        # Retain the lowest N% indices (favoring easier mazes)
        retained = set(np.random.choice(list(self.active_indices), self.size - num_replace, replace=False))

        # Add harder ones next in line
        available_pool = [i for i in self.total_indices if i not in retained]
        new_samples = set(available_pool[:num_replace])

        self.active_indices = retained | new_samples
        return iter(list(self.active_indices))

    def __len__(self):
        return self.size


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
            training_samples = 20
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

    @profile_method(output_file=f"load_mazes_profile")
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
                solved_maze = self._process_maze(maze_data, i)
                # only add valid and solved mazes to the list.
                if solved_maze.valid_solution:
                    solved_training_mazes.append(solved_maze)
                else:
                    logging.warning(f"Maze {i + 1} failed validation.")
            except Exception as e:
                logging.error(f"Failed to process maze {i + 1}: {str(e)}")
                raise RuntimeError(f"Processing maze {i + 1} failed.") from e
        # curriculum learning: Sort by solution path length
        solved_training_mazes.sort(key=lambda maze: maze.complexity)
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
        maze = data
        if not maze.self_test():
            logging.warning(f"Maze {index + 1} failed validation.")

        # solve the maze if not already solved
        if not maze.test_solution():
            # Retrieve the solver name from the configuration (defaulting to BacktrackingMazeSolver)
            solver_obj = config.get("DEFAULT", "solver", fallback="BacktrackingMazeSolver")
            solver_cls = solver_mapping.get(solver_obj)
            if not solver_cls:
                raise RuntimeError(f"Solver '{solver_obj}' not found. Available solvers: {list(solver_mapping.keys())}")

            # Instantiate and use the solver.
            solver = solver_cls(maze)

            maze.set_solution(solver.solve())
            maze.animate = False
            maze.save_movie = False

            # Precompute context map and attach it
            maze.context_map = self._build_context_map(maze)
        return maze

    @profile_method(output_file=f"create_dataset_profile")
    def create_dataset(self):
        """
        Constructs a dataset for training a maze-navigating model.
        Uses caching logic only if use_dataset_cache = True and use_rolling_sampler = False.
        Deletes any existing cache if use_dataset_cache = False.
        Returns:
            Tuple[List[Tuple[np.ndarray, np.ndarray]], List[Tuple[np.ndarray, np.ndarray]]]
        """
        cache_path = os.path.join(INPUT, "training_dataset_cache.pkl")
        val_cache_path = os.path.join(INPUT, "validation_dataset_cache.pkl")

        use_cache = config.getboolean("DEFAULT", "use_dataset_cache", fallback=False)
        use_rolling = config.getboolean("DEFAULT", "use_rolling_sampler", fallback=False)

        # Remove old cache if caching is disabled
        if not use_cache:
            for path in [cache_path, val_cache_path]:
                if os.path.exists(path):
                    os.remove(path)
                    logging.info(f"Deleted dataset cache file: {path}")

        # Try to use cache if allowed
        if use_cache and not use_rolling and os.path.exists(cache_path) and os.path.exists(val_cache_path):
            logging.info("Using cached dataset files.")
            with open(cache_path, "rb") as f:
                dataset = pickle.load(f)
            with open(val_cache_path, "rb") as f:
                validation_dataset = pickle.load(f)
            return dataset, validation_dataset

        # Else: create from scratch
        logging.info("Building dataset from scratch.")
        dataset = []
        for maze in tqdm(self.training_mazes, desc="Creating training dataset"):
            solution = maze.get_solution()
            start_position = maze.start_position
            sequence_inputs = []
            sequence_targets = []
            for i, (current_pos, next_pos) in enumerate(zip(solution[:-1], solution[1:])):
                local_context = maze.context_map.get(current_pos, [WALL] * 4)
                relative_position = (current_pos[0] - start_position[0],
                                     current_pos[1] - start_position[1])
                step_norm = i / (len(solution) - 1)
                input_features = local_context + list(relative_position) + [step_norm]
                sequence_inputs.append(input_features)

                move_delta = (next_pos[0] - current_pos[0], next_pos[1] - current_pos[1])
                if move_delta not in DIRECTION_TO_ACTION:
                    raise KeyError(f"Invalid move delta: {move_delta}")
                target_action = DIRECTION_TO_ACTION[move_delta]
                # Check if this is the step right before the exit
                is_exit_step = next_pos == maze.exit
                if is_exit_step:
                    target_action = 4  # Exit signal (index 4)
                sequence_targets.append(target_action)

            sample_inputs = np.array(sequence_inputs, dtype=np.float32)
            sample_targets = np.array(sequence_targets, dtype=np.int64)
            dataset.append((sample_inputs, sample_targets))

        validation_dataset = []
        for maze in tqdm(self.validation_mazes, desc="Creating validation dataset"):
            start_position = maze.start_position
            solution = maze.get_solution()
            if maze.self_test():
                sequence_inputs = []
                sequence_targets = []
                for i, (current_pos, next_pos) in enumerate(zip(solution[:-1], solution[1:])):
                    # local_context = self._compute_local_context(maze, current_pos, DIRECTIONS)
                    local_context = maze.context_map.get(current_pos, [WALL] * 4)  # use context cache
                    relative_position = (current_pos[0] - start_position[0],
                                         current_pos[1] - start_position[1])
                    step_norm = i / (len(solution) - 1)
                    input_features = local_context + list(relative_position) + [step_norm]
                    sequence_inputs.append(input_features)

                    move_delta = (next_pos[0] - current_pos[0], next_pos[1] - current_pos[1])
                    if move_delta not in DIRECTION_TO_ACTION:
                        raise KeyError(f"Invalid move delta: {move_delta}")
                    target_action = DIRECTION_TO_ACTION[move_delta]
                    sequence_targets.append(target_action)

                sample_inputs = np.array(sequence_inputs, dtype=np.float32)
                sample_targets = np.array(sequence_targets, dtype=np.int64)
                validation_dataset.append((sample_inputs, sample_targets))
            else:
                logging.warning("Maze failed validation.")

        # Cache the newly built datasets
        # Always cache validation dataset
        with open(val_cache_path, "wb") as f:
            pickle.dump(validation_dataset, f)
            logging.info("Cached validation dataset.")

        # Conditionally cache training set
        if use_cache and not use_rolling:
            with open(cache_path, "wb") as f:
                pickle.dump(dataset, f)
            logging.info("Cached training dataset.")

        return dataset, validation_dataset

    def _compute_local_context(self, maze, position, directions):
        """
        Vectorized version of local context computation using NumPy.

        Args:
            maze (Maze): Maze object with .grid attribute (2D numpy array)
            position (tuple): (row, col) position in the maze
            directions (list): List of (dr, dc) direction offsets

        Returns:
            list: Values of the 4 surrounding cells, defaulting to WALL if out of bounds
        """
        r, c = position
        rows, cols = maze.grid.shape
        dr_dc = np.array(directions)
        positions = dr_dc + np.array([r, c])  # shape: (4, 2)

        # Check which are in bounds
        in_bounds = (
                (positions[:, 0] >= 0) & (positions[:, 0] < rows) &
                (positions[:, 1] >= 0) & (positions[:, 1] < cols)
        )

        # Clip indices to valid range, so we can use them safely
        clamped = np.clip(positions, [0, 0], [rows - 1, cols - 1])

        # Use fancy indexing to get neighbor values
        neighbor_vals = maze.grid[clamped[:, 0], clamped[:, 1]]

        # Apply WALL where out of bounds
        context = np.where(in_bounds, neighbor_vals, WALL)

        return context.tolist()

    def _build_context_map(self, maze, directions=DIRECTIONS):
        """
        Precomputes the local context for every position in the maze.

        Args:
            maze (Maze): Maze instance with a grid
            directions (list): Direction deltas [(dr, dc), ...]

        Returns:
            dict: {position: local_context} for each valid cell
        """
        context_map = {}
        for r in range(maze.rows):
            for c in range(maze.cols):
                context_map[(r, c)] = self._compute_local_context(maze, (r, c), directions)
        return context_map

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
            models.append(("LSTM", lstm_model))

    return models


def collate_fn(batch):
    """
    Custom collate function for batching variable-length maze sequences.

    This function:
    - Unpacks a batch of (input_sequence, target_sequence) pairs
    - Converts sequences to PyTorch tensors
    - Pads all sequences in the batch to match the longest one
    - Uses padding_value=-100 for targets so CrossEntropyLoss can ignore them

    Returns:
        inputs_padded (Tensor): [batch_size, max_seq_len, input_size]
        targets_padded (Tensor): [batch_size, max_seq_len] with -100 padding
    """
    inputs, targets = zip(*batch)

    # Convert list of arrays to list of tensors
    inputs = [inp.clone().detach().float() for inp in inputs]
    targets = [tgt.clone().detach().long() for tgt in targets]

    # Pad sequences to same length
    inputs_padded = pad_sequence(inputs, batch_first=True)  # → (batch, max_seq_len, input_size)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=-100)  # → (batch, max_seq_len)

    return inputs_padded, targets_padded


def train_models(allowed_models=None):
    setup_logging()
    clean_outupt_folder()
    logger = logging.getLogger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary_data = []  # traning metadata

    # Load dataset
    trainer = RNN2MazeTrainer(TRAINING_MAZES_FILE, VALIDATION_MAZES_FILE)
    train_data, val_data = trainer.create_dataset()
    train_ds = MazeTrainingDataset(train_data)
    val_ds = ValidationDataset(val_data)
    logger.info(f"Training on {len(train_ds)} total samples")

    # Config flags
    subset_fraction = config.getfloat("DEFAULT", "subset_fraction", fallback=0.1)
    batch_size = config.getint("DEFAULT", "batch_size", fallback=64)
    num_workers = config.getint("DEFAULT", "max_num_workers", fallback=0)

    sampler_option = config.get("DEFAULT", "sampler", fallback="None")
    # Initialize sampler to None by default
    sampler = None
    # Check the sampler option string
    if sampler_option == "RollingSubsetSampler":
        # Ensure subset_fraction is defined before this block
        sampler = RollingSubsetSampler(train_ds, fraction=subset_fraction)
    elif sampler_option == "CurriculumSampler":
        sampler = CurriculumSampler(train_ds)
    elif sampler_option == "None":
        sampler = None
    else:
        raise ValueError(f"Invalid sampler option: {sampler_option}")
    shuffle = sampler is None

    shuffle = (sampler is None)
    # DataLoader config
    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              sampler=sampler,
                              num_workers=num_workers,
                              collate_fn=collate_fn,
                              pin_memory=True,
                              persistent_workers=(num_workers > 0)  # only allowed if num_workers > 0
                              )
    val_loader = DataLoader(val_ds,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            collate_fn=collate_fn,
                            pin_memory=True,
                            persistent_workers=(num_workers > 0)  # only allowed if num_workers > 0
                            )
    # Fallback to config if no model list provided
    if allowed_models is None:
        models_config = config.get("DEFAULT", "models", fallback="GRU,LSTM,RNN")
        allowed_models = [m.strip().upper() for m in models_config.split(",")]

    trained_models = []

    for model_name in allowed_models:
        logging.info(f"Training model: {model_name}")
        model = MazeRecurrentModel(
            mode_type=model_name,
            input_size=config.getint(model_name, "input_size", fallback=7),
            hidden_size=config.getint(model_name, "hidden_size"),
            num_layers=config.getint(model_name, "num_layers"),
            output_size=config.getint(model_name, "output_size", fallback=5),
        ).to(device)

        if wandb_enabled:
            wandb.watch(model, log="all", log_freq=1000)

        model = model.train_model(
            dataloader=train_loader,
            val_loader=val_loader,
            num_epochs=config.getint(model_name, "num_epochs"),
            learning_rate=config.getfloat(model_name, "learning_rate"),
            weight_decay=config.getfloat(model_name, "weight_decay"),
            device=device,
            tensorboard_writer=SummaryWriter(log_dir=f"{OUTPUT}/tensorboard_data/{model_name}")
        )

        trained_models.append((model_name, model))
        model_path = os.path.join(INPUT, config.get("FILES", f"{model_name}_MODEL",
                                                    fallback=f"{model_name.lower()}_model.pth"))
        torch.save(model.state_dict(), model_path)
        logging.info(f"Saved {model_name} model to {model_path}")

        summary_data.append({
            "model": model_name,
            "final_train_loss": model.last_loss,
            "final_val_loss": model.validation_loss if hasattr(model, "validation_loss") else None,
            "final_train_acc": model.training_accuracy if hasattr(model, "training_accuracy") else None,
            "final_val_acc": model.validation_accuracy if hasattr(model, "validation_accuracy") else None,
            "output_file": model_path
        })
        if wandb_enabled:
            wandb.log({f"{model_name}_final_loss": model.last_loss})
    summary_csv = os.path.join(OUTPUT, "summary_metrics.csv")
    summary_json = os.path.join(OUTPUT, "summary_metrics.json")

    # Save to CSV
    df = pd.DataFrame(summary_data)
    df.to_csv(summary_csv, index=False)

    # Save to JSON
    with open(summary_json, "w") as f:
        json.dump(summary_data, f, indent=4)

    logging.info(f"Saved model training summary to {summary_csv} and {summary_json}")

    return trained_models

