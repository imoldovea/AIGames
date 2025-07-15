# neat_solver.py

from maze_solver import MazeSolver
from neat_network import NEATNetwork


class NEATSolver(MazeSolver):
    """
    MazeSolver wrapper for a NEAT genome/network.
    """

    def __init__(self, maze, genome, activation='tanh', steps_per_move=5):
        super().__init__(maze)
        self.net = NEATNetwork(genome, activation=activation)
        self.steps_per_move = steps_per_move
        self.reset_state()

    def reset_state(self):
        # For RNN-style memory, store previous activations per episode
        self.state = None

    def get_input(self):
        # Example: [N, E, S, W, dx, dy, steps_normalized]
        # This may depend on your actual maze input encoding!
        walls = self.maze._compute_local_context(self.maze.current_position)
        dy = self.maze.current_position[0] - self.maze.start_position[0]
        dx = self.maze.current_position[1] - self.maze.start_position[1]
        max_steps = self.maze.rows * self.maze.cols
        steps_norm = len(self.maze.path) / max_steps
        return [float(w) for w in walls] + [dx / self.maze.cols, dy / self.maze.rows, steps_norm]

    def solve(self):
        self.reset_state()
        max_steps = self.maze.rows * self.maze.cols * 2  # double area for safety
        steps = 0
        while not self.maze.at_exit() and steps < max_steps:
            inputs = self.get_input()
            outputs = self.net.activate(inputs, steps=self.steps_per_move, initial_state=self.state)
            # Choose action with highest output activation
            action = int(max(range(len(outputs)), key=lambda i: outputs[i]))
            direction_map = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # N, E, S, W
            r, c = self.maze.current_position
            move = direction_map[action]
            new_pos = (r + move[0], c + move[1])

            if not self.maze.move(new_pos):
                break  # Invalid move, stop

            steps += 1
            # Optional: keep the state for next move (if you want persistent RNN-style memory)
            self.state = self.net.get_state()

        return self.maze.get_path()
