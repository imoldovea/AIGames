# neuro_evo_solver.py
# Wraps a genome + model into a MazeSolver
import torch

from maze_solver import MazeSolver
from neuro_net import NeuroNet


class NeuroEvoSolver(MazeSolver):
    """
    A maze solver using a neural network whose weights are evolved.
    Input: [walls (N,E,S,W), dx, dy, step_norm]
    Output: move direction from [N,E,S,W]
    """

    def __init__(self, maze, genome):
        super().__init__(maze)
        self.model = NeuroNet.from_genome(genome)
        self.model.eval()
        self.max_steps = maze.rows * maze.cols

    def get_input_vector(self):
        # Local context: 0 for corridor, 1 for wall
        walls = self.maze._compute_local_context(self.maze.current_position)
        walls = [float(w) for w in walls]

        # Position relative to start, normalized
        dy = self.maze.current_position[0] - self.maze.start_position[0]
        dx = self.maze.current_position[1] - self.maze.start_position[1]
        rel_pos = [dx / self.maze.cols, dy / self.maze.rows]

        # Progress (normalized step count)
        steps = len(self.maze.path) / self.max_steps

        return torch.tensor(walls + rel_pos + [steps], dtype=torch.float32)

    def solve(self):
        steps = 0
        while not self.maze.at_exit() and steps < self.max_steps:
            obs = self.get_input_vector().unsqueeze(0)  # shape: [1, 7]
            with torch.no_grad():
                logits = self.model(obs)
                action = torch.argmax(logits, dim=1).item()

            direction_map = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # N, E, S, W
            move = direction_map[action]

            r, c = self.maze.current_position
            new_pos = (r + move[0], c + move[1])

            if not self.maze.move(new_pos):
                break  # invalid move

            steps += 1

        return self.maze.get_path()
