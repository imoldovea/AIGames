import numpy as np
import pytest
from rnn_maze_solver import RNNMazeSolver


@pytest.fixture
def sample_grid():
    return np.array([
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [1, 1, 0, 1],
        [0, 0, 1, 0]
    ])


@pytest.fixture
def solver_instance():
    return RNNMazeSolver(None)


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
