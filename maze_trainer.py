# maze_trainer.py
# MazeTrainingDataset and RNN2MazeTrainer

import numpy as np
import utils
from backtrack_maze_solver import BacktrackingMazeSolver
from torch.utils.data import Dataset
import logging
from maze import Maze
from configparser import ConfigParser

WALL = 1

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
        local_context, target_action, step_number = self.data[idx]
        step_number_normalized = step_number / self.max_steps
        return np.array(local_context, dtype=np.float32), target_action, step_number_normalized

# -------------------------------
# Training Utilities (Imitation Learning Setup)
# -------------------------------
class RNN2MazeTrainer:
    def __init__(self, training_file_path="input/training_mazes.pkl"):
        """
        Loads and processes training mazes.
        """
        self.training_file_path = training_file_path
        config = ConfigParser()
        config.read("config.properties")
        self.training_mazes = self._load_and_process_training_mazes()[:int(config.getint("DEFAULT", "training_samples"))]

    def _load_and_process_training_mazes(self):
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
        Constructs a dataset for training a maze-navigating model.
        """
        dataset = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        direction_to_target = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}
        for maze in self.training_mazes:
            solution = maze.get_solution()
            for i, (current_pos, next_pos) in enumerate(zip(solution[:-1], solution[1:])):
                steps_number = i
                local_context = self._compute_local_context(maze, current_pos, directions)
                move_delta = (next_pos[0] - current_pos[0], next_pos[1] - current_pos[1])
                if move_delta not in direction_to_target:
                    raise KeyError(f"Invalid move delta: {move_delta}")
                target_action = direction_to_target[move_delta]
                dataset.append((local_context, target_action, steps_number))
        return dataset

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
def train_models(device="cpu", batch_size=32):
    """
    Trains RNN, GRU, and LSTM models on maze-solving data.
    Returns:
        list: Tuples of model name and trained model.
    """
    import os, csv, subprocess, traceback
    import torch
    import wandb
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from configparser import ConfigParser
    from model import MazeRNN2Model, MazeGRUModel, MazeLSTMModel

    OUTPUT = "output/"
    INPUT = "input/"
    RNN_MODEL_PATH = f"{INPUT}rnn_model.pth"
    GRU_MODEL_PATH = f"{INPUT}gru_model.pth"
    LSTM_MODEL_PATH = f"{INPUT}lstm_model.pth"
    LOSS_FILE = f"{OUTPUT}loss_data.csv"
    LOSS_PLOT_FILE = f"{OUTPUT}loss_plot.png"
    RETRAIN_MODEL = False
    TRAINING_MAZES_FILE = f"{INPUT}training_mazes.pkl"

    try:
        with open(LOSS_FILE, "w", newline="") as f:
            loss_writer = csv.writer(f)
            loss_writer.writerow(["model", "epoch", "loss"])
    except Exception as e:
        logging.error(f"Error setting up loss file: {e}")

    writer = SummaryWriter(log_dir="output/maze_training")
    trainer = RNN2MazeTrainer(TRAINING_MAZES_FILE)
    dataset = trainer.create_dataset()
    logging.info(f"Created {len(dataset)} training samples.")
    train_ds = MazeTrainingDataset(dataset)
    num_workers = 16 if device == "cuda" else 2
    dataloader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers)

    config = ConfigParser()
    config.read("config.properties")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    models = []

    # RNN Model Training
    rnn_model = MazeRNN2Model(
        input_size=config.getint("RNN", "input_size", fallback=5),
        hidden_size=config.getint("RNN", "hidden_size"),
        num_layers=config.getint("RNN", "num_layers"),
        output_size=config.getint("RNN", "output_size", fallback=4),
    )
    rnn_model.to(device)
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
            tensorboard_writer=writer
        )
        logging.info(f"Done training RNN model. Loss {loss:.4f}")
        torch.save(rnn_model.state_dict(), RNN_MODEL_PATH)
        logging.info("Saved RNN model")
        wandb.log({"RNN_final_loss": loss})
        writer.add_scalar("Loss/RNN_final_loss", loss)
    models.append(("RNN", rnn_model))

    # GRU Model Training
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

    # LSTM Model Training
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

    writer.close()
    logging.info("Training complete.")
    return models
