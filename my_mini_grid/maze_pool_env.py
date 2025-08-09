import logging
import time
import os

import gymnasium
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend for video generation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from utils import profile_method

class Config:
    """Configuration class containing reward constants for the maze environment."""

    # Reward constants (tuned for better learning)
    STEP_REWARD = -0.001  # Small per-step penalty
    WALL_PENALTY = -0.02  # Slightly stronger penalty for invalid moves
    SUCCESS_REWARD = 10.0  # Strong positive reward for reaching target

    # Dense shaping for moving towards target
    SHAPING_CLOSER_REWARD = 0.01
    SHAPING_FARTHER_PENALTY = -0.005

    # Episode limits
    MAX_STEPS_MULTIPLIER = 2  # Short episodes to curb prolonged negative returns


class MazePoolEnv(gymnasium.Env):
    # Gymnasium metadata to support recording/rendering
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

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

        # Path tracing
        self.path = []  # list of (row, col) visited positions
        self.show_trace = True  # toggle to render the path


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

        # Initialize path with starting position
        self.path = [(self.agent_pos[0], self.agent_pos[1])]

        # Initialize previous manhattan distance for shaping
        self._prev_manhattan = abs(self.agent_pos[0] - self.target_pos[0]) + abs(self.agent_pos[1] - self.target_pos[1])

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
            # Append to path on successful move
            if not self.path or self.path[-1] != (self.agent_pos[0], self.agent_pos[1]):
                self.path.append((self.agent_pos[0], self.agent_pos[1]))

            # Dense reward shaping based on Manhattan distance improvement
            new_dist = abs(self.agent_pos[0] - self.target_pos[0]) + abs(self.agent_pos[1] - self.target_pos[1])
            if new_dist < getattr(self, "_prev_manhattan", new_dist):
                reward += Config.SHAPING_CLOSER_REWARD
            elif new_dist > getattr(self, "_prev_manhattan", new_dist):
                reward += Config.SHAPING_FARTHER_PENALTY
            self._prev_manhattan = new_dist
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

        if self.maze_grid is None:
            raise ValueError("Cannot render: maze_grid is not initialized.")

        if mode == 'human':
            # Create a visual representation
            display_grid = self.maze_grid.copy().astype(str)
            display_grid[display_grid == '0'] = '.'  # corridors
            display_grid[display_grid == '1'] = '#'  # walls

            # Draw path trace if available
            if self.show_trace and getattr(self, 'path', None):
                for r, c in self.path:
                    # Don't overwrite walls
                    if 0 <= r < self.height and 0 <= c < self.width and self.maze_grid[r, c] == 0:
                        display_grid[r, c] = 'o'

            # Mark agent and target (override path symbol at those cells)
            display_grid[self.agent_pos[0], self.agent_pos[1]] = 'A'
            display_grid[self.target_pos[0], self.target_pos[1]] = 'T'

            print(f"\nMaze {self.current_maze}, Step {self.current_step}")
            for row in display_grid:
                print(''.join(row))
            print()

        elif mode == 'rgb_array':
            # Create a visual representation as RGB array for video recording
            fig, ax = plt.subplots(figsize=(6, 6))
            fig.tight_layout(pad=0)
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

            # Draw path trace as a line connecting centers of visited cells
            if self.show_trace and getattr(self, 'path', None) and len(self.path) > 1:
                xs = [c + 0.5 for (_, c) in self.path]
                ys = [r + 0.5 for (r, _) in self.path]
                ax.plot(xs, ys, color='orange', linewidth=2, alpha=0.8)

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
            # Keep a fixed-size canvas for consistent video frames (no tight bbox)
            ax.set_title(f"Maze {self.current_maze}, Step {self.current_step}")

            # Convert the current canvas to a fixed-size RGB array
            fig.canvas.draw()
            
            # More robust method that works with all matplotlib versions
            renderer = fig.canvas.get_renderer()
            width, height = int(renderer.width), int(renderer.height)
            
            # Get the RGB buffer
            if hasattr(fig.canvas, 'buffer_rgba'):
                # Newer matplotlib versions
                buf = fig.canvas.buffer_rgba()
                rgb_array = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 4))
                rgb_array = rgb_array[:, :, :3]  # Remove alpha channel
            elif hasattr(fig.canvas, 'tostring_rgb'):
                # Older matplotlib versions
                buf = fig.canvas.tostring_rgb()
                rgb_array = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 3))
            else:
                # Fallback method
                fig.savefig('temp_frame.png', dpi=fig.dpi, bbox_inches='tight', pad_inches=0)
                from PIL import Image
                img = Image.open('temp_frame.png')
                rgb_array = np.array(img)[:, :, :3]
                os.remove('temp_frame.png')
            
            plt.close(fig)
            return rgb_array

        else:
            # For other modes, just do nothing or return None
            pass

def run_test_episode_with_human_render(env, model, max_steps):
    """Run a test episode with human console rendering for debugging"""
    obs, info = env.reset()
    step_count = 0
    done = False
    solution = []

    # Create a separate env for human visualization with proper initialization
    human_env = MazePoolEnv(
        env.unwrapped.maze_grids,
        env.unwrapped.starts,
        env.unwrapped.exits,
        render_mode="human"
    )
    # Properly initialize the human environment
    human_env.reset()

    # Now copy the current state
    human_env.current_maze = env.unwrapped.current_maze
    human_env.agent_pos = env.unwrapped.agent_pos.copy()
    human_env.target_pos = env.unwrapped.target_pos.copy()
    human_env.maze_grid = env.unwrapped.maze_grid.copy()
    human_env.path = env.unwrapped.path.copy() if hasattr(env.unwrapped, 'path') else []

    while not done and step_count < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        # Update and render human visualization
        human_env.current_maze = env.unwrapped.current_maze
        human_env.agent_pos = env.unwrapped.agent_pos.copy()
        human_env.target_pos = env.unwrapped.target_pos.copy()
        human_env.current_step = env.unwrapped.current_step
        human_env.maze_grid = env.unwrapped.maze_grid.copy()
        if hasattr(env.unwrapped, 'path'):
            human_env.path = env.unwrapped.path.copy()
        
        human_env.render()  # This will show the human visualization
        time.sleep(0.3)  # Add delay for better visualization

        agent_pos = list(env.unwrapped.agent_pos)
        solution.append(tuple(agent_pos))
        done = terminated or truncated
        step_count += 1

        if step_count % 20 == 0:
            logging.info(f"  Step {step_count}: Action={action}, Reward={reward:.3f}")

    if hasattr(human_env, 'close'):
        human_env.close()
    return terminated, step_count, solution
