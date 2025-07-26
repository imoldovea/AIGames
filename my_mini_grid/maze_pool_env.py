import gymnasium
import numpy as np


class MazePoolEnv(gymnasium.Env):
    def __init__(self, maze_grids, starts, exits):
        super().__init__()  # Initialize the base gym environment class
        self.maze_grids = maze_grids  # List of maze grid arrays representing the environment layouts
        self.starts = starts  # List of start positions in each maze
        self.exits = exits  # List of exit positions in each maze
        self.num_mazes = len(maze_grids)  # Total number of mazes available
        self.current_maze = 0  # Index to track which maze is currently active

        # Determine parameters from the first maze grid to define observation space
        grid_shape = maze_grids[0].shape
        # Define observation space as a Box of 0s and 1s matching the maze grid shape
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=1, shape=grid_shape, dtype=np.int8
        )
        # Define action space as discrete with 4 possible actions (e.g., up, down, left, right)
        self.action_space = gymnasium.spaces.Discrete(4)

        self.state = None  # To hold the current state of the environment (current maze layout)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)  # Reset the base environment with optional seed for reproducibility
        if seed is not None:
            np.random.seed(seed)  # Seed numpy RNG if provided
        # Randomly select an index of maze to start for this new episode
        self.current_maze = np.random.randint(self.num_mazes)
        # Set current state as a copy of the selected maze grid to avoid modifying original
        self.state = self.maze_grids[self.current_maze].copy()

        info = {}  # Placeholder for any additional environment info returned on reset

        # Return the initial observation (full maze grid) and info dictionary
        return self.state, info

    def step(self, action):
        # This method should update the environment state based on the action
        # Currently placeholder values for reward, termination, and truncation
        reward = 0  # No reward logic implemented yet
        terminated = False  # Flag for episode termination (e.g., agent reached exit)
        truncated = False  # Flag for external episode termination (e.g., time limit)
        info = {}  # Dictionary for any additional info about this step
        # Here you would add logic to update self.state based on action, check for goal reached, etc.

        # Return updated state, reward, termination flags, and info after taking action
        return self.state, reward, terminated, truncated, info

    def render(self, mode='human'):
        # Optional method to render the environment, e.g., print maze or draw GUI
        # Currently does nothing, can be implemented later if desired
        pass
