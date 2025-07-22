import gymnasium
import numpy as np


class MazePoolEnv(gymnasium.Env):
    def __init__(self, maze_grids, starts, exits):
        super().__init__()
        self.maze_grids = maze_grids
        self.starts = starts
        self.exits = exits
        self.num_mazes = len(maze_grids)
        self.current_maze = 0

        # Set maze parameters by inspecting first maze
        grid_shape = maze_grids[0].shape
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=1, shape=grid_shape, dtype=np.int8
        )
        self.action_space = gymnasium.spaces.Discrete(4)  # e.g., 4 directions

        self.state = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        # Select a random maze for each episode
        self.current_maze = np.random.randint(self.num_mazes)
        self.state = self.maze_grids[self.current_maze].copy()

        info = {}  # Add any reset-specific info if needed

        # If you return more than just the grid, adjust observation_space and this line
        return self.state, info

    def step(self, action):
        # For demo purposes, reward/done are dummy
        # You MUST implement correct maze logic here!
        reward = 0
        terminated = False  # True if agent reaches the exit
        truncated = False  # True if episode was truncated (e.g., by time limit)
        info = {}
        # Update self.state based on action here...
        return self.state, reward, terminated, truncated, info

    def render(self, mode='human'):
        print(self.state)
