import torch
import torch.nn as nn
from maze_solver import MazeSolver
from bfs_maze_solver import BFSMazeSolver
from backtrack_maze_solver import BacktrackingMazeSolver
from maze import Maze
import numpy as np
import torch.optim as optim
import logging
import pickle

logging.basicConfig(level=logging.INFO)

# -----------------------------
# PyTorch RNN Model for Maze Solving
# -----------------------------
class MazeRNNModel(nn.Module):
    def __init__(self, input_size=9, hidden_size=32, num_actions=4):
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
    def __init__(self, maze: Maze, model: MazeRNNModel = None):
        super().__init__(maze)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_size = 32
        self.model = model if model is not None else MazeRNNModel(hidden_size=self.hidden_size)
        self.model.to(self.device)
        self.model.train()

    def get_local_patch(self, position, patch_size=3):
        """Extract a local patch around the given position (with padding)."""
        pad = patch_size // 2
        padded_grid = np.pad(self.maze.grid, pad, constant_values=1)  # pad with walls
        pr, pc = position[0] + pad, position[1] + pad
        patch = padded_grid[pr - pad: pr + pad + 1, pc - pad: pc + pad + 1]
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

    def train_model(self, training_data, epochs=500):
        """
        Train the RNN model with training_data, where each entry is a tuple (position, target_action).
        """
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            # Reset hidden state at the start of each epoch.
            hx = torch.zeros(1, self.hidden_size).to(self.device)
            cx = torch.zeros(1, self.hidden_size).to(self.device)
            for (position, target_action) in training_data:
                patch = self.get_local_patch(position, patch_size=3)
                patch_flat = torch.tensor(patch.flatten(), dtype=torch.float32).unsqueeze(0).to(self.device)
                logits, hx, cx = self.model(patch_flat, hx, cx)
                target = torch.tensor([target_action], dtype=torch.long).to(self.device)
                loss = loss_fn(logits, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Detach hidden states to prevent backpropagating through the entire history
                hx = hx.detach()
                cx = cx.detach()
                total_loss += loss.item()
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Avg Loss = {total_loss / len(training_data):.4f}")
        self.model.eval()  # switch to evaluation mode

    def solve(self, max_steps=50):
        """
        Use the trained RNN model to predict moves until the maze exit is reached or max_steps is exceeded.
        Each move is made with backtrack=True to show progress. When a solution is found,
        set_solution() is called on the maze.
        """
        self.model.eval()
        hx = torch.zeros(1, self.hidden_size).to(self.device)
        cx = torch.zeros(1, self.hidden_size).to(self.device)
        current_position = self.maze.get_position()
        solution_path = [current_position]
        for step in range(max_steps):
            if self.maze.at_exit():
                print("Exit reached!")
                break
            patch = self.get_local_patch(current_position, patch_size=3)
            patch_flat = torch.tensor(patch.flatten(), dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits, hx, cx = self.model(patch_flat, hx, cx)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                candidate_action = np.argmax(probs)

            candidate_offset = self.action_to_offset(candidate_action)
            new_position = (current_position[0] + candidate_offset[0], current_position[1] + candidate_offset[1])
            # Check move validity using Maze methods.
            if self.maze.is_valid_move(new_position):
                self.maze.move(new_position, backtrack=True)
                current_position = new_position
                solution_path.append(current_position)
            else:
                # Fallback: explore all possible moves.
                valid_moves = []
                for action in range(4):
                    offset = self.action_to_offset(action)
                    pos = (current_position[0] + offset[0], current_position[1] + offset[1])
                    if self.maze.is_within_bounds(pos) and not self.maze.is_wall(pos) and self.maze.is_valid_move(pos):
                        valid_moves.append((action, pos))
                if valid_moves:
                    valid_probs = {action: probs[action] for action, pos in valid_moves}
                    best_action = max(valid_probs, key=valid_probs.get)
                    best_offset = self.action_to_offset(best_action)
                    new_position = (current_position[0] + best_offset[0], current_position[1] + best_offset[1])
                    self.maze.move(new_position, backtrack=True)
                    current_position = new_position
                    solution_path.append(current_position)
                else:
                    print(f"No valid moves from position {current_position}. Exploration halted.")
                    break
        # If exit is reached, record the solution path.
        if self.maze.at_exit():
            self.maze.set_solution(solution_path)
        return solution_path


# -----------------------------
# Test method for the RNN-based Maze Solver
# -----------------------------
def test_maze_solver_rnn():
    # Load mazes
    with open('input/mazes.pkl', 'rb') as f:
        mazes = pickle.load(f)
    logging.info(f"Loaded {len(mazes)} mazes.")

    # Iterate through each maze in the array
    for i, maze_matrix in enumerate(mazes):
        maze = Maze(maze_matrix)
        maze.set_animate(False)
        maze.set_save_movie(False)
        logging.debug(f"Solving maze {i + 1} with BFS...")

        solver = BacktrackingMazeSolver(maze)
        solution = solver.solve()

        if solution:
            logging.debug(f"Maze {i + 1} solution found:")
            logging.debug(solution)
        else:
            logging.debug(f"No solution found for maze {i + 1}.")
        maze.set_solution(solution)

    training_path = maze.get_solution()
    training_data = []
    for i in range(len(training_path) - 1):
        current = training_path[i]
        next_pos = training_path[i + 1]
        delta = (next_pos[0] - current[0], next_pos[1] - current[1])
        if delta == (-1, 0):
            action = 0  # up
        elif delta == (1, 0):
            action = 1  # down
        elif delta == (0, -1):
            action = 2  # left
        elif delta == (0, 1):
            action = 3  # right
        training_data.append((current, action))

    # Initialize the RNN maze solver and train the model.
    solver_rnn = RNNMazeSolver(maze)
    print("Training the RNN model...")
    solver_rnn.train_model(training_data, epochs=500)

    # Reset maze state.
    maze.current_position = maze.start_position
    maze.path = [maze.start_position]

    print("Solving the maze with RNN exploration...")
    solution = solver_rnn.solve(max_steps=20)
    print("Solution path:", solution)
    maze.plot_maze(show_path=False, show_solution=True)


if __name__ == '__main__':
    test_maze_solver_rnn()
