import logging
import subprocess
import time

import numpy as np  # Import numpy
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

from maze_loader import load_mazes_h5
from maze_pool_env import MazePoolEnv
from utils import setup_logging, clean_outupt_folder


class Config:
    TRAINING_TIMESTEPS = 10_000
    TRAINING_SAMPLES = 10
    TEST_SAMPLES = 5
    MAX_TEST_STEPS = 200


def validate_environments(train_env, test_env):
    """Ensure training and testing environments are compatible"""
    if train_env.observation_space.shape != test_env.observation_space.shape:
        raise ValueError(
            f"Training and test environments have different observation spaces: "
            f"Training shape: {train_env.observation_space.shape}, "
            f"Test shape: {test_env.observation_space.shape}"
        )
    if train_env.action_space.n != test_env.action_space.n:
        raise ValueError(
            f"Training and test environments have different action spaces: "
            f"Training actions: {train_env.action_space.n}, "
            f"Test actions: {test_env.action_space.n}"
        )


def train_model(env, log_dir):
    """Train the PPO model on the maze environment"""
    # Configure gym logger
    gym_logger = configure(folder=log_dir, format_strings=["log", "csv", "json", "tensorboard"])
    gym_logger.set_level(logging.INFO)
    gym_logger.info("Starting training...")

    # Use MlpPolicy for simple Box observation space
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    model.set_logger(gym_logger)

    # Train the model
    model.learn(total_timesteps=Config.TRAINING_TIMESTEPS, progress_bar=True)
    model.save("output/ppo_minigrid_maze")
    
    return model

def test_model(model, base_test_env):
    """Test the trained model on different maze set"""
    logging.info("\n" + "=" * 50)
    logging.info("TESTING TRAINED MODEL ON DIFFERENT MAZE SET")
    logging.info("=" * 50)

    # Wrap test environment with RecordVideo to record all test episodes
    test_env = RecordVideo(
        base_test_env,
        video_folder="output/videos",
        episode_trigger=lambda episode_id: True,  # Record all test episodes
        name_prefix="testing"
    )

    logging.info(f"Successfully loaded {len(base_test_env.maze_grids)} test mazes")

    # Test the model on multiple mazes
    total_success = 0
    total_tests = min(5, len(base_test_env.maze_grids))  # Test up to 5 mazes

    # In the testing section, use the run_test_episode function
    for test_idx in range(total_tests):
        logging.info(f"\n--- Testing on Maze {test_idx + 1}/{total_tests} ---")
        terminated, step_count, solution = run_test_episode(test_env, model, Config.MAX_TEST_STEPS)

        maze_index = test_env.unwrapped.current_maze_index

        logging.info(f"Testing on maze index {maze_index}")

        # Evaluate results
        if terminated:
            logging.info(f"  ✅ SUCCESS! Reached target in {step_count} steps")
            total_success += 1
        elif step_count >= Config.MAX_TEST_STEPS:
            logging.info(f"  ⏰ TIMEOUT after {step_count} steps")
        else:
            logging.info(f"  ❌ FAILED after {step_count} steps")

        logging.info(f"  Final position: {test_env.unwrapped.agent_pos}")
        logging.info(f"  Target position: {test_env.unwrapped.target_pos}")
        logging.info(f"  Path length: {len(solution)}")

    # Print overall results
    success_rate = (total_success / total_tests) * 100
    logging.info(f"\n{'=' * 50}")
    logging.info(f"OVERALL TEST RESULTS:")
    logging.info(f"Successful: {total_success}/{total_tests} ({success_rate:.1f}%)")
    logging.info(f"{'=' * 50}")

    test_env.close()
    return success_rate


def run_test_episode(env, model, max_steps):
    obs, info = env.reset()
    step_count = 0
    done = False
    solution = []

    # Create a separate env for human visualization
    human_env = MazePoolEnv(
        env.unwrapped.maze_grids,
        env.unwrapped.starts,
        env.unwrapped.exits,
        render_mode="human"
    )
    human_env.current_maze = env.unwrapped.current_maze
    human_env.agent_pos = env.unwrapped.agent_pos.copy()
    human_env.target_pos = env.unwrapped.target_pos.copy()

    while not done and step_count < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        # Update and render human visualization
        human_env.current_maze = env.unwrapped.current_maze
        human_env.agent_pos = env.unwrapped.agent_pos.copy()
        human_env.target_pos = env.unwrapped.target_pos.copy()
        human_env.current_step = env.unwrapped.current_step
        human_env.render()  # This will show the human visualization
        time.sleep(0.1)  # Add delay for better visualization

        agent_pos = list(env.unwrapped.agent_pos)
        solution.append(tuple(agent_pos))
        done = terminated or truncated
        step_count += 1

        if step_count % 20 == 0:
            logging.info(f"  Step {step_count}: Action={action}, Reward={reward:.3f}")

    human_env.close()
    return terminated, step_count, solution

def main():
    log_dir = "output/"

    # Load and create environments with uniform padding
    base_env, base_test_env = load_and_create_environments(
        training_file="input/training_mazes.h5",
        test_file="input/mazes.h5",
        training_samples=Config.TRAINING_SAMPLES,
        test_samples=Config.TEST_SAMPLES
    )

    # First wrap with RecordVideo
    env = RecordVideo(
        base_env,
        video_folder="output/videos",
        episode_trigger=lambda episode_id: episode_id % 50 == 0,
        name_prefix="training"
    )

    # Validate compatibility (check base environments)
    validate_environments(base_env, base_test_env)

    # Train the model
    model = train_model(env, log_dir)
    
    # Test the model
    test_model(model, base_test_env)

    # Close environments to ensure videos are saved
    env.close()
    
    logging.info(f"\nVideos saved to: output/videos/")


def start_tensorboard(logdir):
    """Start TensorBoard server for training visualization"""
    tensorboard = subprocess.Popen(
        ['tensorboard', '--logdir', logdir, '--port', '6006'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    logging.info("TensorBoard started. Open http://localhost:6006 in your browser")
    return tensorboard


def load_and_create_environments(training_file, test_file, training_samples, test_samples):
    """
    Load training and test mazes, pad them to uniform size, and create environments.
    
    Args:
        training_file (str): Path to training mazes HDF5 file
        test_file (str): Path to test mazes HDF5 file
        training_samples (int): Number of training mazes to load
        test_samples (int): Number of test mazes to load
        
    Returns:
        tuple: (base_env, base_test_env) - Training and test environments with uniformly padded mazes
    """
    # Load training and test mazes
    maze_grids, starts, exits = load_mazes_h5(training_file, samples=training_samples)
    test_maze_grids, test_starts, test_exits = load_mazes_h5(test_file, samples=test_samples)

    # Find maximum dimensions across both sets
    max_h = max(max(grid.shape[0] for grid in maze_grids),
                max(grid.shape[0] for grid in test_maze_grids))
    max_w = max(max(grid.shape[1] for grid in maze_grids),
                max(grid.shape[1] for grid in test_maze_grids))

    # Pad both sets to uniform size
    def pad_to_size(grids, starts, target_h, target_w):
        padded_grids = []
        updated_starts = []
        for grid, start in zip(grids, starts):
            h, w = grid.shape
            padded = np.full((target_h, target_w), 2, dtype=grid.dtype)  # Use 2 for padding
            padded[:h, :w] = grid
            padded_grids.append(padded)
            start_row, start_col = start
            updated_starts.append((start_row, start_col))
        return padded_grids, updated_starts

    # Pad both sets to the same dimensions
    maze_grids, starts = pad_to_size(maze_grids, starts, max_h, max_w)
    test_maze_grids, test_starts = pad_to_size(test_maze_grids, test_starts, max_h, max_w)

    # Create environments with padded mazes and support both render modes
    base_env = MazePoolEnv(maze_grids, starts, exits, render_mode="rgb_array")  # Changed back to rgb_array
    base_test_env = MazePoolEnv(test_maze_grids, test_starts, test_exits, render_mode="rgb_array")  # Changed back to rgb_array

    return base_env, base_test_env

if __name__ == "__main__":
    clean_outupt_folder()  # Clean output directory before starting
    setup_logging()  # Setup basic logging configuration
    tensorboard_process = start_tensorboard("output/")  # Start tensorboard on output logs
    main()  # Run the main training and evaluation function