import pytest

from maze import Maze
from rnn.rnn2_maze_solver import RNN2MazeSolver


def test_rnn_solver_valid_solution():
    # Create a solvable maze
    solvable_maze_grid = [
        [1, 1, 1, 1],
        [1, 1, 0, 1],
        [1, 2, 0, 1],
        [1, 1, 0, 1]
    ]
    solvable_maze = Maze(solvable_maze_grid)
    solver = RNN2MazeSolver(solvable_maze)
    solution = solver.solve()
    assert solvable_maze.test_solution(solution), "RNN Solver failed to solve a valid maze."


def test_rnn_solver_for_unsolvable_maze():
    # Create a maze with no valid solution
    unsolvable_maze_grid = [
        [1, 1, 1, 0],
        [1, 1, 2, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 1]
    ]
    unsolvable_maze = Maze(unsolvable_maze_grid)
    solver = RNN2MazeSolver(unsolvable_maze)
    with pytest.raises(ValueError, match="Maze is unsolvable"):
        solver.solve()


def test_rnn_solver_invalid_dimensions():
    # Test maze with invalid dimensions
    invalid_maze_grid = [
        [0, 1],
        [1, 0]
    ]
    with pytest.raises(ValueError, match="Maze dimensions are invalid"):
        invalid_maze = Maze(invalid_maze_grid)
        RNN2MazeSolver(invalid_maze)


def test_rnn_solver_respects_max_steps():
    # Create a solvable maze but test the solver with limited steps
    solvable_maze_grid = [
        [1, 1, 1, 1],
        [1, 1, 0, 1],
        [1, 2, 0, 1],
        [1, 1, 0, 1]
    ]
    max_steps = 5
    solvable_maze = Maze(solvable_maze_grid)
    solver = RNN2MazeSolver(solvable_maze, max_steps=max_steps)
    solution = solver.solve()
    assert len(solution) <= max_steps, "RNN Solver exceeded the maximum step limit."


def test_rnn_solver_with_mock_configuration():
    # Mocked configuration for the neural network settings
    mock_config = {
        "hidden_size": 64,
        "num_layers": 2,
        "learning_rate": 0.001
    }
    solvable_maze_grid = [
        [1, 1, 1, 1],
        [1, 1, 0, 1],
        [1, 2, 0, 1],
        [1, 1, 0, 1]
    ]
    solvable_maze = Maze(solvable_maze_grid)
    solver = RNN2MazeSolver(solvable_maze, neural_net_config=mock_config)
    solution = solver.solve()
    assert solvable_maze.test_solution(), "RNN Solver failed with the mocked configuration."
