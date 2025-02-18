import torch
import torch.nn as nn
import torch.optim as optim
from maze_solver import MazeSolver
from bfs_maze_solver import BFSMazeSolver
from backtrack_maze_solver import BacktrackingMazeSolver
from maze import Maze
import numpy as np
import logging
from utils import (
    display_all_mazes,
    save_mazes_as_pdf,
    load_mazes)
from enum import Enum


class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

logging.basicConfig(level=logging.INFO)

PATCH_SIZE = 3
HIDDEN_SIZE = 32
PATCH_SIZE = 3  # Assuming PATCH_SIZE is 3

# -----------------------------
# PyTorch RNN Model for Maze Solving
# -----------------------------
class MazeRNNModel(nn.Module):
    def __init__(self, input_size=PATCH_SIZE*3, hidden_size=32, num_actions=4):
        super(MazeRNNModel, self).__init__()
        self.hidden_size = hidden_size
        # A simple LSTMCell for processing the flattened local patch.
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_actions)

    def forward(self, x, hx, cx):
        hx, cx = self.lstm_cell(x, (hx, cx))
        logits = self.fc(hx)
        return logits, hx, cx


# -----------------------------
# RNNMazeSolver Implementation using Maze exploration methods
# -----------------------------
class RNNMazeSolver(MazeSolver):
    def __init__(self, maze=None, model=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_size = HIDDEN_SIZE
        self.model = model if model is not None else MazeRNNModel(hidden_size=self.hidden_size)
        self.model.to(self.device)
        self.model.train()

    def get_local_patch(self, position):
        grid = self.maze.grid
        half_patch = PATCH_SIZE // 2
        x, y = position

        # Pad the grid to handle edges
        padded_grid = np.pad(grid, pad_width=half_patch, mode='constant', constant_values=0)

        # Adjust coordinates because of padding
        x_padded, y_padded = x + half_patch, y + half_patch

        # Extract the patch
        patch = padded_grid[x_padded - half_patch: x_padded + half_patch + 1,
                y_padded - half_patch: y_padded + half_patch + 1]

        # Assert the shape of the patch
        assert patch.shape == (PATCH_SIZE, PATCH_SIZE), (
            f"Extracted patch has incorrect shape: {patch.shape}, expected ({PATCH_SIZE}, {PATCH_SIZE})"
        )

        if patch.shape != (PATCH_SIZE, PATCH_SIZE):
            raise ValueError(
                f"Extracted patch has incorrect shape: {patch.shape}, expected ({PATCH_SIZE}, {PATCH_SIZE})"
            )

        return patch

    def action_to_offset(self, action):
        """Map an action index to a move offset."""
        if action == 0:
            return (-1, 0)  # up
        elif action == 1:
            return (1, 0)  # down
        elif action == 2:
            return (0, -1)  # left
        elif action == 3:
            return (0, 1)  # right
        else:
            raise ValueError("Invalid action.")

    def train_model(self, training_data, epochs: int = 500, early_stopping: bool = False, patience: int = 10) -> None:
        """
            Train the RNN model with training_data, where each entry is a tuple (position, target_action).
            Implements early stopping if enabled.
        """
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        best_loss = float('inf')
        patience_counter = 0  # Counter for early stopping

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            # Step 1: Initialize the hidden states at the start of each epoch.
            hx = torch.zeros(1, self.hidden_size).to(self.device)
            cx = torch.zeros(1, self.hidden_size).to(self.device)
            for (position, target_action) in training_data:
                # Unpack the position into x and y
                x, y = position

                # Step 2: Extract the local patch around the current position.
                rows, cols = self.maze.grid.shape
                assert 0 <= x < rows and 0 <= y < cols, f"Position {position} is out of the maze grid bounds."

                patch = self.get_local_patch(position)
                patch_flat = torch.tensor(patch.flatten(), dtype=torch.float32).unsqueeze(0).to(self.device)

                # Step 3: Perform a forward pass through the model to compute logits.
                logits, hx, cx = self.model(patch_flat, hx, cx)

                # Step 4: Convert the target_action to a tensor.
                target = torch.tensor([target_action], dtype=torch.long).to(self.device)

                # Step 5: Compute the loss using the CrossEntropyLoss function.
                loss = loss_fn(logits, target)

                # Step 6: Reset gradients before backpropagation.
                optimizer.zero_grad()

                # Step 7: Perform backpropagation to compute gradients.
                loss.backward()

                # Step 8: Update model weights using the optimizer.
                optimizer.step()

                # Step 9: Detach hidden states to truncate the computation graph.
                hx = hx.detach()
                cx = cx.detach()

                # Step 10: Accumulate the total loss for monitoring.
                total_loss += loss.item()

            # Compute average loss for the epoch
            avg_loss = total_loss / len(training_data)

            # Step 11: Check for early stopping if enabled
            if early_stopping:
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0  # Reset counter when improvement is observed
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logging.info(f"Early stopping triggered at epoch {epoch}. Best Loss = {best_loss:.4f}")
                        break

            # Print the average loss every 100 epochs for monitoring progress.
            if epoch % 100 == 0:
                logging.info(f"Epoch {epoch}: Avg Loss = {avg_loss:.4f}")

        # Step 12: Switch the model to evaluation mode after training is complete.
        self.model.eval()

    def solve(self, max_steps=50):
        """
        Use the trained RNN model to predict moves until the maze exit is reached or max_steps is exceeded.
        Each move is made with backtrack=True to show progress. When a solution is found,
        set_solution() is called on the maze.
        """
        # Step 1: Switch the model to evaluation mode.
        self.model.eval()

        # Step 2: Initialize the hidden states for solving.
        hx = torch.zeros(1, self.hidden_size).to(self.device)
        cx = torch.zeros(1, self.hidden_size).to(self.device)

        # Step 3: Get the starting position of the maze.
        current_position = self.maze.get_position()
        solution_path = [current_position]

        # Step 4: Begin iterating through steps to solve the maze.
        for step in range(max_steps):
            # Step 4.1: Check if the exit has been reached.
            if self.maze.at_exit():
                logging.debug("Exit reached!")
                break

            # Step 4.2: Extract the current local patch.
            patch = self.get_local_patch(current_position)
            patch_flat = torch.tensor(patch.flatten(), dtype=torch.float32).unsqueeze(0).to(self.device)

            # Step 4.3: Perform a forward pass through the model to get logits.
            with torch.no_grad():
                logits, hx, cx = self.model(patch_flat, hx, cx)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                candidate_action = np.argmax(probs)

            # Step 4.4: Compute the offset for the predicted action.
            candidate_offset = self.action_to_offset(candidate_action)
            new_position = (current_position[0] + candidate_offset[0], current_position[1] + candidate_offset[1])

            # Step 4.5: Check if the new position is valid.
            if self.maze.is_valid_move(new_position):
                # Step 4.5.1: Move to the new position and append to the solution path.
                self.maze.move(new_position, backtrack=True)
                current_position = new_position
                solution_path.append(current_position)
            else:
                # Step 4.6: If the predicted action is invalid, explore all possible moves.
                valid_moves = []
                for action in range(4):
                    offset = self.action_to_offset(action)
                    pos = (current_position[0] + offset[0], current_position[1] + offset[1])
                    if self.maze.is_within_bounds(pos) and not self.maze.is_wall(pos) and self.maze.is_valid_move(pos):
                        valid_moves.append((action, pos))

                # Step 4.7: Choose the best valid move based on the probabilities.
                if valid_moves:
                    valid_probs = {action: probs[action] for action, pos in valid_moves}
                    best_action = max(valid_probs, key=valid_probs.get)
                    best_offset = self.action_to_offset(best_action)
                    new_position = (current_position[0] + best_offset[0], current_position[1] + best_offset[1])
                    self.maze.move(new_position, backtrack=True)
                    current_position = new_position
                    solution_path.append(current_position)
                else:
                    # Step 4.8: If no valid moves remain, stop exploration.
                    logging.error(f"No valid moves from position {current_position}. Exploration halted.")
                    break

        # Step 5: If the exit is reached, record the solution path.
        if self.maze.at_exit():
            self.maze.set_solution(solution_path)

        # Step 6: Return the solution path.
        return solution_path


# -----------------------------
# Test method for the RNN-based Maze Solver
# -----------------------------
def maze_solver_rnn() -> None:
    """
    This function demonstrates the process of solving mazes using both a traditional
    backtracking approach and a Recurrent Neural Network (RNN) based solver. It initially
    loads mazes for training, solves them using backtracking methods, and then uses the
    solutions to generate training data for the RNN model. After training the RNN, it applies
    the trained neural network to solve additional mazes and save the results into specified
    output formats like PDF or display them visually.

    :return: None
    """
    # Load mazes
    mazes=load_mazes("input/mazes.pkl")
    logging.info(f"Loaded {len(mazes)} mazes.")

    training_mazes = load_mazes("input/training_mazes.pkl")
    logging.info(f"Loaded {len(training_mazes)} training mazes.")

    solved_mazes = []
    all_training_data = []  # Initialize an empty list to collect data from all mazes

    # Iterate through each maze in the array
    for idx, maze in enumerate(training_mazes):
        training_maze = Maze(maze)
        training_maze.set_animate(False)
        training_maze.set_save_movie(False)
        logging.debug(f"Solving maze {idx + 1} with BFS...")

        solver = BacktrackingMazeSolver(training_maze)
        try:
            solution = solver.solve()
        except Exception as e:
            logging.error(f"Error solving maze {idx + 1}: {e}")
            continue

        if solution:
            logging.debug(f"Maze {idx + 1} solution found:")
            logging.debug(solution)
        else:
            logging.error(f"No solution found for maze {idx + 1}.")
        training_maze.set_solution(solution)
        solved_mazes.append(training_maze)

        training_path = training_maze.get_solution()
        if not training_path:
            logging.warning(f"Skipping maze {idx + 1} due to no solution.")

        for i in range(len(training_path) - 1):
            current = training_path[i]
            next_pos = training_path[i + 1]
            delta = (next_pos[0] - current[0], next_pos[1] - current[1])
            if delta == (-1, 0):
                action = action = Action.UP.value  # up
            elif delta == (1, 0):
                action = action = Action.DOWN.value  # down
            elif delta == (0, -1):
                action = action = Action.LEFT.value  # left
            elif delta == (0, 1):
                action = action = Action.RIGHT.value  # right

            all_training_data.append((current, action))
            if not all_training_data:
                logging.error("No training data collected. Aborting RNN training.")
                return

    # Initialize the RNN maze solver and train the model.
    solver_rnn = RNNMazeSolver(maze=None)

    solver_rnn.train_model(all_training_data, epochs=500, early_stopping=True)


    #Solve mazes
    solved_mazes = []
    for i, maze_data in enumerate(mazes):
        logging.info(f"Solving the maze {i} with RNN exploration...")
        maze = Maze(maze_data)
        maze.set_animate(False)
        maze.set_save_movie(False)

        # Initialize RNNMazeSolver for each maze
        solver = RNNMazeSolver(maze=maze)
        solution = solver.solve(max_steps=25)
        maze.set_solution(solution)
        solved_mazes.append(maze)

    save_mazes_as_pdf(solved_mazes, "output/rnn_maze_solver_output.pdf")
    display_all_mazes(solved_mazes,)
    # save_movie(solved_mazes, "rnn_maze_solver_output.mp4")


if __name__ == '__main__':
    maze_solver_rnn()
