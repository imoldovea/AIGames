#maze_trainer.py
import numpy as np
import utils
from backtrack_maze_solver import BacktrackingMazeSolver
from torch.utils.data import Dataset
import logging
from maze import Maze
from configparser import ConfigParser

# -------------------------------
# Custom Dataset for Training Samples
# -------------------------------
class MazeTrainingDataset(Dataset):
    def __init__(self, data):
        """
        Initializes the MazeTrainingDataset instance.

        Args:
            data (list): List of tuples containing:
                - local_context (list): A list representing the state of maze cells around the current position.
                - target_action (int): The action to take from the current position (0: up, 1: down, 2: left, 3: right).
                - step_number (int): The step number in the maze solution path.
        """
        self.data = data  # Store the dataset data as an instance variable
        max_steps = max(sample[2] for sample in data)  # Determine the maximum step number in any solution path
        self.max_steps = max_steps  # Store the maximum step count as an instance variable

    def __len__(self):
        """
        Returns:
            int: The total number of samples in the dataset.
        """
        return len(self.data)  # Return the total number of training samples

    def __getitem__(self, idx):
        """
        Retrieves the training sample at the specified index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - input_features (numpy.ndarray): Combined input features (local_context and normalized step_number).
                - target_action (int): The action to take from the current position.
                - step_number_normalized (float): The normalized step number (0.0 to 1.0).
        """
        local_context, target_action, step_number = self.data[idx]  # Unpack the sample data at the given index
        step_number_normalized = step_number / self.max_steps  # Normalize the step number to a 0.0-1.0 range

        # Combine the local context with the normalized step number
        #input_features = np.append(local_context, step_number_normalized).astype(np.float32)

        # Return the input features, target action, and normalized step number
        return np.array(local_context, dtype=np.float32), target_action, step_number_normalized




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
        config = ConfigParser()
        config.read("config.properties")
        self.training_mazes = self._load_and_process_training_mazes()[
                              :int(config.getint("DEFAULT", "training_samples"))]


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
        dataset = []  # List to store the training samples
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Possible directions: up, down, left, right
        direction_to_target = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}  # Mapping from directions to actions

        # Iterate through all mazes in the training data
        for maze in self.training_mazes:
            solution = maze.get_solution()  # Get the solution path as a list of coordinates
            for i, (current_pos, next_pos) in enumerate(zip(solution[:-1], solution[1:])):
                steps_number = i  # Step number within the solution path
                local_context = self._compute_local_context(maze, current_pos,
                                                            directions)  # Context around the current position
                move_delta = (next_pos[0] - current_pos[0],
                              next_pos[1] - current_pos[1])  # Difference between current and next position

                # Ensure the move_delta corresponds to a valid direction
                if move_delta not in direction_to_target:
                    raise KeyError(f"Invalid move delta: {move_delta}")

                target_action = direction_to_target[move_delta]  # Get the action corresponding to the move
                dataset.append((local_context, target_action, steps_number))  # Append the training sample
        return dataset  # Return the complete dataset

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
