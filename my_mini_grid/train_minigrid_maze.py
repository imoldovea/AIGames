import logging
import subprocess

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

from maze_loader import load_mazes_h5
from maze_mini_grid_env import MazeMiniGridEnv
from maze_pool_env import MazePoolEnv
from utils import setup_logging, clean_outupt_folder


def main():
    # Logging
    log_dir = "output/"
    gym_logger = configure(folder=log_dir, format_strings=["log", "csv", "json", "tensorboard"])
    gym_logger.set_level(logging.INFO)
    gym_logger.info("Starting training...")

    # Load a batch of mazes (choose a large enough number)
    maze_grids, starts, exits = load_mazes_h5("input/training_mazes.h5", samples=10)

    # Create a randomized environment pool
    env = MazePoolEnv(maze_grids, starts, exits)

    # Train RL agent
    # MlpPolicy, MlpLstmPolicy (LSTM), MlpLnLstmPolicy (LSTM normalized)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    model.set_logger(gym_logger)
    # 1_000_000
    model.learn(total_timesteps=1_000, progress_bar=True)
    model.save("output/ppo_minigrid_maze")

    # Pick the last maze (or any index you want)
    last_idx = -1
    maze_grid = maze_grids[last_idx]
    start_pos = starts[last_idx]
    exit_pos = exits[last_idx]

    # Convert to tuple of ints!
    start_pos = tuple(int(x) for x in start_pos)
    exit_pos = tuple(int(x) for x in exit_pos)
    h, w = maze_grid.shape

    assert 0 <= start_pos[1] < h and 0 <= start_pos[0] < w, "start_pos out of bounds"
    assert 0 <= exit_pos[1] < h and 0 <= exit_pos[0] < w, "exit_pos out of bounds"
    assert maze_grid[start_pos[0], start_pos[1]] == 0, f"start_pos {start_pos} is not in a corridor!"
    assert maze_grid[exit_pos[0], exit_pos[1]] == 0, f"exit_pos {exit_pos} is not in a corridor!"

    # Create a single-maze Minigrid environment
    minigrid_env = MazeMiniGridEnv(maze_grid, start_pos, exit_pos)
    obs, info = minigrid_env.reset()

    # Collect agent's path by using the trained model
    solution = []
    done = False
    while not done:
        # Optionally render each step
        minigrid_env.render(mode='human')
        # Save current agent position (row, col)
        agent_x, agent_y = minigrid_env.agent_pos
        solution.append((agent_y, agent_x))  # (row, col)
        # Predict next action
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = minigrid_env.step(action)
        done = terminated or truncated

    # Optionally, animate the full path again (not needed if you rendered above)
    # import time
    # for row, col in solution:
    #     minigrid_env.agent_pos = (col, row)
    #     minigrid_env.render()
    #     time.sleep(0.1)

    # Or just show the final solution state:
    minigrid_env.agent_pos = (solution[-1][1], solution[-1][0])
    minigrid_env.render(mode='human')


def start_tensorboard(logdir):
    """Start TensorBoard server for training visualization"""
    tensorboard = subprocess.Popen(
        ['tensorboard', '--logdir', logdir, '--port', '6006'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    print("TensorBoard started. Open http://localhost:6006 in your browser")

    return tensorboard


if __name__ == "__main__":
    clean_outupt_folder()
    setup_logging()
    tensoresboard = start_tensorboard("output/")
    main()
