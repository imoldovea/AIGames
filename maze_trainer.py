#maze_trainer.py
import numpy as np
import torch
import utils
from backtrack_maze_solver import BacktrackingMazeSolver
from torch.utils.data import DataLoader, Dataset
import logging
from maze import Maze

# -------------------------------
# Custom Dataset for Training Samples
# -------------------------------
class MazeTrainingDataset(Dataset):
    def __init__(self, data):
        """
        data: a list of tuples (local_context, target_action)
              local_context: list of 4 scalar values (e.g., [0, 0, 1, 1])
              target_action: integer (0-3)
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        local_context, target_action = self.data[idx]
        # Convert list of scalars to a numpy array then to a tensor of shape [4]
        local_context = torch.from_numpy(np.array(local_context)).float()
        return local_context, torch.tensor(target_action, dtype=torch.long)


# -------------------------------
# Training Utilities (Imitation Learning Setup)
# -------------------------------
class RNN2MazeTrainer:
    def __init__(self, training_file_path="input/training_mazes.pkl"):
        """
        Initializes the trainer with a file path for training mazes.
        Loads and processes the training mazes upon instantiation.
        """
        self.training_file_path = training_file_path
        self.training_mazes = self._load_and_process_training_mazes()

    def _load_and_process_training_mazes(self):
        """
        Loads training mazes from the specified file and processes each maze.
        Returns:
            list: A list of Maze objects with computed solutions.
        """
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
        Constructs a dataset for training a model to navigate mazes.
        Each training sample is a tuple (local_context, target_action), where:
          - local_context: A list of 4 values representing the state of the maze cells
                           in the order [up, down, left, right] relative to the current position.
                           (0 for corridor, 1 for wall; out-of-bound cells are treated as walls.)
          - target_action: An integer (0: up, 1: down, 2: left, 3: right) representing the move
                           from the current position to the next position in the solution.
        Returns:
            list: A list of (local_context, target_action) training samples.
        """
        dataset = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        direction_to_target = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}

        for maze in self.training_mazes:
            solution = maze.get_solution()  # List of coordinates from start to exit
            for i in range(len(solution) - 1):
                current_pos = solution[i]
                next_pos = solution[i + 1]
                local_context = self._compute_local_context(maze, current_pos, directions)
                move_delta = (next_pos[0] - current_pos[0], next_pos[1] - current_pos[1])
                if move_delta not in direction_to_target:
                    raise KeyError(f"Invalid move delta: {move_delta}")
                target_action = direction_to_target[move_delta]
                dataset.append((local_context, target_action))
        return dataset

    def _compute_local_context(self, maze, position, directions):
        """
        Computes the local context around a given position in a maze.
        Returns a list of cell states in the order defined by 'directions'.
        """
        r, c = position
        local_context = []
        for dr, dc in directions:
            neighbor = (r + dr, c + dc)
            cell_state = maze.grid[neighbor] if maze.is_within_bounds(neighbor) else WALL
            local_context.append(cell_state)
        return local_context
