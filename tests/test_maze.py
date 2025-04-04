import numpy as np

from maze import Maze


def test_initialization():
    grid = np.array([
        [1, 1, 1],
        [1, 3, 0],
        [1, 1, 1]
    ])
    start_position = (1, 1)
    exit_position = (1, 2)
    maze = Maze(grid, start_position=start_position, exit_position=exit_position)

    assert maze.grid.shape == (3, 3), "Maze grid shape is incorrect."
    assert maze.start_position == start_position, "Start position is incorrect."
    assert maze.exit == exit_position, "Exit position is incorrect."
    assert maze.path == [start_position], "Initial path is incorrect."


def test_set_exit():
    grid = np.ones((5, 5))
    grid[1, 1] = Maze.START
    grid[1, 2] = Maze.CORRIDOR
    grid[1, 3] = Maze.CORRIDOR
    maze = Maze(grid, start_position=(1, 1))
    result = maze.set_exit((1, 3))

    assert result is True, "Failed to set a valid exit."
    assert maze.exit == (1, 3), "Exit position is not set correctly."
    assert maze.grid[1, 3] == Maze.CORRIDOR, "Exit position is not set as corridor."


def test_is_within_bounds():
    grid = np.ones((4, 4))
    maze = Maze(grid)
    assert maze.is_within_bounds((2, 2)) is True, "Valid position incorrectly detected as out of bounds."
    assert maze.is_within_bounds((4, 4)) is False, "Out of bounds position incorrectly detected as valid."


def test_is_wall():
    grid = np.ones((4, 4))
    grid[1, 1] = Maze.CORRIDOR
    maze = Maze(grid)
    assert maze.is_wall((0, 0)) is True, "Wall incorrectly detected as not a wall."
    assert maze.is_wall((1, 1)) is False, "Corridor incorrectly detected as a wall."


def test_is_corridor():
    grid = np.ones((4, 4))
    grid[1, 1] = Maze.CORRIDOR
    maze = Maze(grid)
    assert maze.is_corridor((1, 1)) is True, "Corridor incorrectly detected as not a corridor."
    assert maze.is_corridor((0, 0)) is False, "Wall incorrectly detected as a corridor."


def test_is_valid_move():
    grid = np.ones((4, 4))
    grid[1, 1] = Maze.CORRIDOR
    maze = Maze(grid)
    assert maze.is_valid_move((1, 1)) is True, "Valid move incorrectly marked as invalid."
    assert maze.is_valid_move((0, 0)) is False, "Invalid move incorrectly marked as valid."


def test_move():
    grid = np.ones((5, 5))
    grid[2, 2] = Maze.START
    grid[2, 3] = Maze.CORRIDOR
    maze = Maze(grid, start_position=(2, 2))
    result = maze.move((2, 3))

    assert result is True, "Failed to move to a valid position."
    assert maze.current_position == (2, 3), "Current position not updated correctly."
    assert maze.path == [(2, 2), (2, 3)], "Path not updated correctly."


def test_self_test():
    grid = np.array([
        [1, 1, 1, 1],
        [1, 3, 0, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 1]
    ])
    maze = Maze(grid, start_position=(1, 1), exit_position=(1, 2))
    assert maze.self_test() is True, "Valid maze configuration failed verification."


def test_test_solution():
    grid = np.array([
        [1, 1, 1, 1],
        [1, 3, 0, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 1]
    ])
    maze = Maze(grid, start_position=(1, 1), exit_position=(1, 2))
    solution = [(1, 1), (1, 2)]
    maze.set_solution(solution)
    assert maze.test_solution() is True, "Valid solution failed verification."


def test_at_exit():
    grid = np.array([
        [1, 1, 1],
        [1, 3, 0],
        [1, 1, 1]
    ])
    maze = Maze(grid, start_position=(1, 1), exit_position=(1, 2))
    maze.move((1, 2))
    assert maze.at_exit() is True, "Failed to detect when at the exit position."
