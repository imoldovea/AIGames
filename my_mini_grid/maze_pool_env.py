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

        # Determine parameters from the first maze grid
        grid_shape = maze_grids[0].shape
        self.height, self.width = grid_shape

        # Define observation space as flattened maze grid + agent position + target position
        obs_size = grid_shape[0] * grid_shape[1] + 4  # maze + agent_pos + target_pos
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=1, shape=(obs_size,), dtype=np.float32
        )

        # Define action space: 0=up, 1=right, 2=down, 3=left
        self.action_space = gymnasium.spaces.Discrete(4)

        # Current state variables
        self.agent_pos = None
        self.target_pos = None
        self.maze_grid = None
        self.max_steps = 4 * self.height * self.width
        self.current_step = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        # Randomly select a maze
        self.current_maze = np.random.randint(self.num_mazes)
        self.maze_grid = self.maze_grids[self.current_maze].copy()

        # Set agent and target positions (convert from (row, col) to (row, col))
        start_pos = self.starts[self.current_maze]
        exit_pos = self.exits[self.current_maze]

        self.agent_pos = [int(start_pos[0]), int(start_pos[1])]  # [row, col]
        self.target_pos = [int(exit_pos[0]), int(exit_pos[1])]  # [row, col]
        self.current_step = 0

        obs = self._get_observation()
        info = {'maze_index': self.current_maze}

        return obs, info

    def step(self, action):
        self.current_step += 1

        # Define action mappings: 0=up, 1=right, 2=down, 3=left
        action_map = {
            0: (-1, 0),  # up
            1: (0, 1),  # right
            2: (1, 0),  # down
            3: (0, -1)  # left
        }

        # Calculate new position
        dr, dc = action_map[action]
        new_row = self.agent_pos[0] + dr
        new_col = self.agent_pos[1] + dc

        # Check if move is valid (within bounds and not a wall)
        if (0 <= new_row < self.height and
                0 <= new_col < self.width and
                self.maze_grid[new_row, new_col] == 0):  # 0 = corridor, 1 = wall
            self.agent_pos = [new_row, new_col]
            reward = -0.01  # Small negative reward for each step
        else:
            reward = -0.1  # Penalty for hitting wall or going out of bounds

        # Check if reached target
        terminated = (self.agent_pos[0] == self.target_pos[0] and
                      self.agent_pos[1] == self.target_pos[1])
        if terminated:
            reward = 1.0  # Large reward for reaching target

        # Check if episode should be truncated (max steps reached)
        truncated = self.current_step >= self.max_steps

        obs = self._get_observation()
        info = {'maze_index': self.current_maze}

        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        # Flatten maze grid and concatenate with agent and target positions
        maze_flat = self.maze_grid.flatten().astype(np.float32)

        # Normalize positions to [0, 1]
        agent_pos_norm = np.array([
            self.agent_pos[0] / self.height,
            self.agent_pos[1] / self.width
        ], dtype=np.float32)

        target_pos_norm = np.array([
            self.target_pos[0] / self.height,
            self.target_pos[1] / self.width
        ], dtype=np.float32)

        # Concatenate all parts
        obs = np.concatenate([maze_flat, agent_pos_norm, target_pos_norm])
        return obs

    def render(self, mode='human'):
        if mode == 'human':
            # Create a visual representation
            display_grid = self.maze_grid.copy().astype(str)
            display_grid[display_grid == '0'] = '.'  # corridors
            display_grid[display_grid == '1'] = '#'  # walls

            # Mark agent and target
            display_grid[self.agent_pos[0], self.agent_pos[1]] = 'A'
            display_grid[self.target_pos[0], self.target_pos[1]] = 'T'

            print(f"\nMaze {self.current_maze}, Step {self.current_step}")
            for row in display_grid:
                print(''.join(row))
            print()
