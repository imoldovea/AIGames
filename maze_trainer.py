# maze_trainer.py
# MazeTrainingDataset and RNN2MazeTrainer

import numpy as np
import utils
from backtrack_maze_solver import BacktrackingMazeSolver
from torch.utils.data import Dataset
import logging
from maze import Maze
import os, csv
import wandb
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
from configparser import ConfigParser
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

RNN_MODEL_PATH = f"{INPUT}rnn_model.pth"
GRU_MODEL_PATH = f"{INPUT}gru_model.pth"
LSTM_MODEL_PATH = f"{INPUT}lstm_model.pth"
LOSS_FILE = f"{OUTPUT}loss_data.csv"

TRAINING_MAZES_FILE = f"{INPUT}training_mazes.pkl"
VALIDATION_MAZES_FILE = f"{INPUT}validation_mazes.pkl"

# -------------------------------
# Custom Dataset for Training Samples
# -------------------------------
class MazeTrainingDataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data (list): List of tuples containing:
                - local_context (list): State of maze cells around the current position.
                - target_action (int): Action to take (0: up, 1: down, 2: left, 3: right).
                - step_number (int): Step number in the solution path.
        """
        self.data = data
        max_steps = max(sample[2] for sample in data)
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
    Subclass of MazeTrainingDataset to handle validation data.
    Loads validation mazes from a specified file.
    """

    def __init__(self, data):
        """
        Args:
            data (list): List of tuples containing:
                - local_context (list): State of maze cells around the current position.
                - target_action (int): Action to take (0: up, 1: down, 2: left, 3: right).
                - step_number (int): Step number in the solution path.
        """
        self.data = data
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
    def __init__(self, training_file_path="input/training_mazes.pkl", validation_file_path="input/validation_mazes.pkl"):
        """
        Loads and processes training mazes.
        """
        self.training_file_path = training_file_path
        self.validation_file_path = validation_file_path
        config = ConfigParser()
        config.read("config.properties")
        training_samples = config.getint("DEFAULT", "training_samples")

        self.training_mazes = self._load_and_process_training_mazes(training_file_path,training_samples)
        self.validation_mazes = self._load_and_process_training_mazes(validation_file_path,training_samples//10)

    def _load_and_process_training_mazes(self, path, training_samples):
        """
            Loads and processes training mazes from the specified file path.
        
            Parameters:
                file_path (str): Path to the training mazes file.
        
            Returns:
                List of processed training mazes.
            """
        training_mazes = self._load_mazes_safe(path)
        solved_training_mazes = []
        for i, maze_data in enumerate(training_mazes):
            if i >= training_samples:
                break
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
        Constructs a dataset for training a maze-navigating model.
        """
        dataset = []
        for maze in self.training_mazes:
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
        for maze in self.validation_mazes:
            solution = maze.get_solution()
            for i, (current_pos, next_pos) in enumerate(zip(solution[:-1], solution[1:])):
                steps_number = i
                local_context = self._compute_local_context(maze, current_pos, DIRECTIONS)
                relative_position = (current_pos[0] - start_position[0], current_pos[1] - start_position[1])
                move_delta = (next_pos[0] - current_pos[0], next_pos[1] - current_pos[1])
                if move_delta not in DIRECTION_TO_ACTION:
                    raise KeyError(f"Invalid move delta: {move_delta}")
                target_action = DIRECTION_TO_ACTION[move_delta]
                validation_dataset.append((local_context, relative_position, target_action, steps_number))

        return dataset, validation_dataset

    def _compute_local_context(self, maze, position, directions):
        r, c = position
        local_context = []
        for dr, dc in directions:
            neighbor = (r + dr, c + dc)
            cell_state = maze.grid[neighbor] if maze.is_within_bounds(neighbor) else WALL
            local_context.append(cell_state)
        return local_context

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

def train_models(device="cpu", batch_size=32):
    """
    Trains RNN, GRU, and LSTM models on maze-solving data.
    Returns:
        list: Tuples of model name and trained model.
    """
    logging.debug("Starting training.")

    config = ConfigParser()
    config.read("config.properties")

    #Delete previous tensorboard files.
    try:
        with open(LOSS_FILE, "w", newline="") as f:
            loss_writer = csv.writer(f)
            loss_writer.writerow(["model", "epoch", "loss", "validation_loss"])
    except Exception as e:
        logging.error(f"Error setting up loss file: {e}")

    tensorboard_data_sever = SummaryWriter(log_dir=f"{OUTPUT}tensorboard_data")

    trainer = RNN2MazeTrainer(TRAINING_MAZES_FILE, VALIDATION_MAZES_FILE)
    dataset, validation_dataset= trainer.create_dataset()
    logging.info(f"Created {len(dataset)} training samples.")
    train_ds = MazeTrainingDataset(dataset)
    validation_ds = ValidationDataset(validation_dataset)
    num_workers = 16 if device == "cuda" else 2
    dataloader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers)

    config = ConfigParser()
    config.read("config.properties")
    logging.info(f'Using device: {device}')

    models = []

    # RNN Model Training
    rnn_model = MazeRecurrentModel(
        mode_type="RNN",
        input_size=config.getint("RNN", "input_size", fallback=7),
        hidden_size=config.getint("RNN", "hidden_size"),
        num_layers=config.getint("RNN", "num_layers"),
        output_size=config.getint("RNN", "output_size", fallback=4),
    )
    rnn_model.to(device)
    retrain_model = config.getboolean("RNN", "retrain_model", fallback=False)
    wandb.watch(rnn_model, log="all", log_freq=10)
    if not retrain_model and os.path.exists(RNN_MODEL_PATH):
        load_model_state(rnn_model, RNN_MODEL_PATH, "rnn.", "recurrent.")
        logging.info("RNN model loaded from file")
    else:
        logging.info("Training RNN model")
        rnn_model = rnn_model.train_model(
            dataloader = dataloader,
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
        wandb.log({"RNN_final_loss": loss})
        tensorboard_data_sever.add_scalar("Loss/RNN_final_loss", loss)
    models.append(("RNN", rnn_model))

    # GRU Model Training
    gru_model = MazeRecurrentModel(
        mode_type="GRU",
        input_size=config.getint("GRU", "input_size", fallback=7),
        hidden_size=config.getint("GRU", "hidden_size"),
        num_layers=config.getint("GRU", "num_layers"),
        output_size=config.getint("GRU", "output_size", fallback=4),
    )
    gru_model.to(device)
    retrain_model = config.getboolean("GRU", "retrain_model", fallback=False)
    wandb.watch(gru_model, log="all", log_freq=10)
    if not retrain_model and os.path.exists(GRU_MODEL_PATH):
        load_model_state(gru_model, GRU_MODEL_PATH, "gru.", "recurrent.")
        logging.debug("GRU model loaded from file.")
    else:
        logging.info("Training GRU model")
        gru_model = gru_model.train_model(
            dataloader = dataloader,
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
        wandb.log({"GRU_final_loss": loss})
        tensorboard_data_sever.add_scalar("Loss/GRU_final_loss", loss)
    models.append(("GRU", gru_model))

    # LSTM Model Training
    lstm_model = MazeRecurrentModel(
        mode_type="LSTM",
        input_size=config.getint("LSTM", "input_size", fallback=7),
        hidden_size=config.getint("LSTM", "hidden_size"),
        num_layers=config.getint("LSTM", "num_layers"),
        output_size=config.getint("LSTM", "output_size", fallback=4),
    )
    lstm_model.to(device)
    retrain_model = config.getboolean("LSTM", "retrain_model", fallback=False)
    wandb.watch(lstm_model, log="all", log_freq=10)
    if not retrain_model and os.path.exists(LSTM_MODEL_PATH):
        load_model_state(lstm_model, LSTM_MODEL_PATH, "lstm.", "recurrent.")
        logging.info("LSTM model loaded from file.")
    else:
        logging.info("Training LSTM model")
        lstm_model = lstm_model.train_model(
            dataloader = dataloader,
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
        wandb.log({"LSTM_final_loss": loss})
        tensorboard_data_sever.add_scalar("Loss/LSTM_final_loss", loss)
    models.append(("LSTM", lstm_model))
    tensorboard_data_sever.close()
    logging.info("Training complete.")
    return models
