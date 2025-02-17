import numpy as np
import pytest
from generate_labytinths import find_start


def test_find_start_sets_start_position():
    width, height = 5, 5
    maze = np.zeros((height, width), dtype=int)  # Create an empty maze with all 0's
    updated_maze = find_start(maze, width, height)
    start_positions = np.argwhere(updated_maze == 3)
    assert len(start_positions) == 1, "There should be exactly one starting position marked as '3'."


def test_find_start_no_valid_path():
    width, height = 5, 5
    maze = np.ones((height, width), dtype=int)  # Create a maze filled with 1's (no valid paths)
    updated_maze = find_start(maze, width, height)
    start_positions = np.argwhere(updated_maze == 3)
    assert len(start_positions) == 0, "There should be no starting position when no valid paths are available."


def test_find_start_maintains_other_cells():
    width, height = 5, 5
    maze = np.zeros((height, width), dtype=int)  # Create a maze filled with all 0's
    original_maze = maze.copy()
    updated_maze = find_start(maze, width, height)
    start_positions = np.argwhere(updated_maze == 3)
    if len(start_positions) == 1:
        start_y, start_x = start_positions[0]
        original_maze[start_y, start_x] = 3
    assert np.array_equal(original_maze, updated_maze), "Only the starting position should be updated."
