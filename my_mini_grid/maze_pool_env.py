from io import BytesIO

import gymnasium
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class Config:
    """Configuration class containing reward constants for the maze environment."""

    # Reward constants
    STEP_REWARD = -0.01  # Small negative reward for each step
    WALL_PENALTY = -0.1  # Penalty for hitting wall or going out of bounds
    SUCCESS_REWARD = 1.0  # Large reward for reaching target

    # Episode limits
    MAX_STEPS_MULTIPLIER = 4  # Multiplier for calculating max steps based on maze size


class MazePoolEnv(gymnasium.Env):
    def __init__(self, maze_grids, starts, exits, render_mode=None):
        super().__init__()
        self.maze_grids = maze_grids
        self.starts = starts
        self.exits = exits
        self.num_mazes = len(maze_grids)
        self.current_maze = 0
        self.render_mode = render_mode

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
        self.max_steps = Config.MAX_STEPS_MULTIPLIER * self.height * self.width
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

        # Convert action to regular Python int if it's a numpy array
        action = int(action)

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
            self.agent_pos = [int(new_row), int(new_col)]  # Ensure it's a Python list with int values
            reward = Config.STEP_REWARD  # Small negative reward for each step
        else:
            reward = Config.WALL_PENALTY  # Penalty for hitting wall or going out of bounds

        # Check if reached target
        terminated = (self.agent_pos[0] == self.target_pos[0] and
                      self.agent_pos[1] == self.target_pos[1])
        if terminated:
            reward = Config.SUCCESS_REWARD  # Large reward for reaching target

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

    def render(self, mode=None):
        # Use self.render_mode if mode is not specified
        if mode is None:
            mode = self.render_mode

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

        elif mode == 'rgb_array':
            # Create a visual representation as RGB array for video recording
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.set_xlim(0, self.width)
            ax.set_ylim(0, self.height)
            ax.set_aspect('equal')
            ax.invert_yaxis()  # Flip y-axis to match array indexing

            # Draw maze walls
            for row in range(self.height):
                for col in range(self.width):
                    if self.maze_grid[row, col] == 1:  # Wall
                        rect = patches.Rectangle((col, row), 1, 1,
                                                 linewidth=0, facecolor='black')
                        ax.add_patch(rect)
                    else:  # Corridor
                        rect = patches.Rectangle((col, row), 1, 1,
                                                 linewidth=0, facecolor='white')
                        ax.add_patch(rect)

            # Draw agent (blue circle)
            agent_circle = patches.Circle((self.agent_pos[1] + 0.5, self.agent_pos[0] + 0.5),
                                          0.3, facecolor='blue', edgecolor='darkblue')
            ax.add_patch(agent_circle)

            # Draw target (red circle)
            target_circle = patches.Circle((self.target_pos[1] + 0.5, self.target_pos[0] + 0.5),
                                           0.3, facecolor='red', edgecolor='darkred')
            ax.add_patch(target_circle)

            # Remove axes and add title
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"Maze {self.current_maze}, Step {self.current_step}")

            # Convert plot to RGB array
            fig.canvas.draw()
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            img = Image.open(buf)
            rgb_array = np.array(img)
            buf.close()
            plt.close(fig)

            return rgb_array

        else:
            # For other modes, just do nothing or return None
            pass
