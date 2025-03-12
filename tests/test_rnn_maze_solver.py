import numpy as np
import pytest
from rnn2_maze_solver import RNN2MazeSolver, MazeLSTMModel
from maze import Maze
import torch
from configparser import ConfigParser

PARAMETERS_FILE = "config.properties"
config = ConfigParser()
config.read(PARAMETERS_FILE)
OUTPUT = config.get("FILES", "OUTPUT", fallback="output/")
INPUT = config.get("FILES", "INPUT", fallback="input/")

LSTM_MODEL_PATH = f"{INPUT}lstm_model.pth"

# Load the model state dictionary once
state_dict = torch.load(LSTM_MODEL_PATH)

# Read configurations
config = ConfigParser()
config.read("config.properties")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the LSTM model once
lstm_model = MazeLSTMModel(
    input_size=config.getint("LSTM", "input_size", fallback=5),
    output_size=config.getint("LSTM", "output_size", fallback=4),
    hidden_size=config.getint("LSTM", "hidden_size"),
    num_layers=config.getint("LSTM", "num_layers"),
)
lstm_model.to(device)
lstm_model.load_state_dict(state_dict)


@pytest.fixture
def sample_grid():
    return np.array([
        [1, 1, 1, 1, 1],
        [1, 3, 0, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 0, 1]
    ])


@pytest.fixture
def maze_instance(sample_grid):
    return Maze(sample_grid)


@pytest.fixture
def solver_instance(maze_instance):
    return RNN2MazeSolver(maze=maze_instance, model=lstm_model, device=device)


def test_get_local_patch_center_position(solver_instance, sample_grid):
    position = (1, 1)
    patch = solver_instance.get_local_patch(sample_grid, position)
    expected_patch = np.array([
        [1, 0, 0],
        [0, 1, 1],
        [1, 1, 0]
    ])
    assert np.array_equal(patch, expected_patch), "The extracted patch does not match the expected patch."


def test_get_local_patch_edge_position(solver_instance, sample_grid):
    position = (0, 0)
    patch = solver_instance.get_local_patch(sample_grid, position)
    expected_patch = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    assert np.array_equal(patch,
                          expected_patch), "The extracted patch for edge position does not match the expected patch."


def test_get_local_patch_correct_size(solver_instance, sample_grid):
    position = (2, 2)
    patch = solver_instance.get_local_patch(sample_grid, position)
    assert patch.shape == (3, 3), f"Expected patch size (3, 3), but got {patch.shape}."
