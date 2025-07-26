import logging
import subprocess

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

from maze_loader import load_mazes_h5
from maze_mini_grid_env import MazeMiniGridEnv
from maze_pool_env import MazePoolEnv
from utils import setup_logging, clean_outupt_folder


def main():
    log_dir = "output/"

    # Configure gym logger to save multiple formats (log, csv, json, tensorboard)
    gym_logger = configure(folder=log_dir, format_strings=["log", "csv", "json", "tensorboard"])
    gym_logger.set_level(logging.INFO)
    gym_logger.info("Starting training...")

    # Load a batch of mazes, sampling 10 for training
    maze_grids, starts, exits = load_mazes_h5("input/training_mazes.h5", samples=10)

    # Create a pool environment with randomized maze selection for training
    env = MazePoolEnv(maze_grids, starts, exits)

    # Initialize PPO model with Multilayer Perceptron policy
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    model.set_logger(gym_logger)  # Attach the custom logger to model

    # Train the model for a total of 1000 timesteps with progress bar
    model.learn(total_timesteps=1_000, progress_bar=True)

    # Save the trained model to disk
    model.save("output/ppo_minigrid_maze")

    # Select the last maze from the batch for demonstration/testing
    last_idx = -1
    maze_grid = maze_grids[last_idx]
    start_pos = starts[last_idx]
    exit_pos = exits[last_idx]

    # In your HDF5, positions are (row, col)
    start_row, start_col = map(int, start_pos)
    exit_row, exit_col = map(int, exit_pos)

    h, w = maze_grid.shape

    # Sanity check: Are these within bounds?
    assert 0 <= start_row < h and 0 <= start_col < w, "start_pos out of bounds"
    assert 0 <= exit_row < h and 0 <= exit_col < w, "exit_pos out of bounds"

    # Make sure start/exit are on a corridor (0)
    assert maze_grid[start_row, start_col] == 0, f"start_pos ({start_row},{start_col}) is not in a corridor!"
    assert maze_grid[exit_row, exit_col] == 0, f"exit_pos ({exit_row},{exit_col}) is not in a corridor!"

    # MiniGrid expects (x, y) = (col, row)!
    mg_start_pos = (start_col, start_row)
    mg_exit_pos = (exit_col, exit_row)

    # Extra checks before creating the environment
    h, w = maze_grid.shape
    assert 0 <= mg_start_pos[0] < w and 0 <= mg_start_pos[1] < h, f"MiniGrid start pos {mg_start_pos} out of bounds"
    assert 0 <= mg_exit_pos[0] < w and 0 <= mg_exit_pos[1] < h, f"MiniGrid exit pos {mg_exit_pos} out of bounds"
    assert maze_grid[mg_start_pos[1], mg_start_pos[0]] == 0, f"MiniGrid agent_pos {mg_start_pos} not on corridor"
    assert maze_grid[mg_exit_pos[1], mg_exit_pos[0]] == 0, f"MiniGrid exit_pos {mg_exit_pos} not on corridor"

    # Now safe to create the environment
    minigrid_env = MazeMiniGridEnv(maze_grid, mg_start_pos, mg_exit_pos)

    obs, info = minigrid_env.reset()  # Reset environment to get initial observation

    # Initialize path list to store agent's positions during the run
    solution = []
    done = False

    # Run inference loop until completion of episode
    while not done:
        minigrid_env.render()  # Render the environment visually in human mode

        # Get current agent position as (x, y)
        agent_x, agent_y = minigrid_env.agent_pos

        # Append agent position as (row, col) to solution path
        solution.append((agent_y, agent_x))

        # Predict next action based on current observation
        action, _ = model.predict(obs)

        # Take action in the environment and observe results
        obs, reward, terminated, truncated, info = minigrid_env.step(action)

        # Determine if episode finished via termination or truncation
        done = terminated or truncated

    # Optionally, render and animate the full solution path (commented out)
    # import time
    # for row, col in solution:
    #     minigrid_env.agent_pos = (col, row)
    #     minigrid_env.render()
    #     time.sleep(0.1)

    # Show the final state where the agent ended up
    minigrid_env.agent_pos = (solution[-1][1], solution[-1][0])
    minigrid_env.render(mode='human')


def start_tensorboard(logdir):
    """Start TensorBoard server for training visualization"""
    tensorboard = subprocess.Popen(
        ['tensorboard', '--logdir', logdir, '--port', '6006'],  # Run tensorboard process with log directory and port
        stdout=subprocess.PIPE,  # Pipe stdout
        stderr=subprocess.PIPE  # Pipe stderr
    )
    print("TensorBoard started. Open http://localhost:6006 in your browser")  # Notify user

    return tensorboard  # Return handle for possible later termination


if __name__ == "__main__":
    clean_outupt_folder()  # Clean output directory before starting
    setup_logging()  # Setup basic logging configuration
    tensoresboard = start_tensorboard("output/")  # Start tensorboard on output logs
    main()  # Run the main training and evaluation function
