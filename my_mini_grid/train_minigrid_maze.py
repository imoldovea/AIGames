import logging
import subprocess

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


def main():
    log_dir = "output/"

    # Configure gym logger
    gym_logger = configure(folder=log_dir, format_strings=["log", "csv", "json", "tensorboard"])
    gym_logger.set_level(logging.INFO)
    gym_logger.info("Starting training...")

    # Load and create environments with uniform padding
    base_env, base_test_env = load_and_create_environments(
        training_file="input/training_mazes.h5",
        test_file="input/mazes.h5",
        training_samples=Config.TRAINING_SAMPLES,
        test_samples=Config.TEST_SAMPLES
    )

    # Wrap with RecordVideo to record training episodes
    env = RecordVideo(
        base_env,
        video_folder="output/videos",
        episode_trigger=lambda episode_id: episode_id % 50 == 0,
        name_prefix="training"
    )

    # Use MlpPolicy for simple Box observation space
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    model.set_logger(gym_logger)

    # Train the model
    model.learn(total_timesteps=Config.TRAINING_TIMESTEPS, progress_bar=True)
    model.save("output/ppo_minigrid_maze")

    # TESTING PHASE
    logging.info("\n" + "=" * 50)
    logging.info("TESTING TRAINED MODEL ON DIFFERENT MAZE SET")
    logging.info
    # Wrap test environment with RecordVideo
    test_env = RecordVideo(
        base_test_env,
        video_folder="output/videos",
        episode_trigger=lambda episode_id: True,
        name_prefix="testing"
    )

    # TESTING PHASE - Load different maze set from output/mazes.h5
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

    # Validate compatibility (check base environments)
    validate_environments(base_env, base_test_env)

    logging.info(f"Successfully loaded {len(base_test_env.maze_grids)} test mazes")

    # Test the model on multiple mazes
    total_success = 0
    total_tests = min(5, len(base_test_env.maze_grids))  # Test up to 5 mazes

    for test_idx in range(total_tests):
        logging.info(f"\n--- Testing on Maze {test_idx + 1}/{total_tests} ---")

        # Reset environment for testing
        obs, info = test_env.reset()
        maze_index = info['maze_index']

        logging.info(f"Testing on maze index {maze_index}")
        # Remove or comment out the complexity line since it's not available in this context
        # logging.info(f"Maze complexity: {test_mazes[maze_index].complexity}")

        # Initialize tracking variables
        solution = []
        done = False
        step_count = 0
        max_test_steps = Config.MAX_TEST_STEPS

        # Run inference loop
        while not done and step_count < max_test_steps:
            # Store current agent position - ensure it's a regular Python list
            agent_pos = list(test_env.unwrapped.agent_pos)  # Use unwrapped to access original env
            solution.append(tuple(agent_pos))

            # Predict next action (deterministic for testing)
            action, _ = model.predict(obs, deterministic=True)

            # Take step in environment
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            step_count += 1

            # Optional: print progress every 20 steps
            if step_count % 20 == 0:
                logging.info(f"  Step {step_count}: Action={action}, Reward={reward:.3f}")

        # Evaluate results
        if terminated:
            logging.info(f"  ✅ SUCCESS! Reached target in {step_count} steps")
            total_success += 1

            # Remove maze visualization since test_mazes is not defined
            # test_mazes[maze_index].set_solution(solution)
            # test_mazes[maze_index].plot_maze(show_solution=True, show_path=False)

        elif truncated:
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

    # Close environments to ensure videos are saved
    env.close()
    test_env.close()

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

    # Create environments with padded mazes and both render modes
    base_env = MazePoolEnv(maze_grids, starts, exits, render_mode="human")  # Changed to "human"
    base_test_env = MazePoolEnv(test_maze_grids, test_starts, test_exits, render_mode="human")  # Changed to "human"

    return base_env, base_test_env

if __name__ == "__main__":
    clean_outupt_folder()  # Clean output directory before starting
    setup_logging()  # Setup basic logging configuration
    tensorboard_process = start_tensorboard("output/")  # Start tensorboard on output logs
    main()  # Run the main training and evaluation function