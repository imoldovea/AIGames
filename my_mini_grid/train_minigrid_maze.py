import logging
import subprocess

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np  # Import numpy
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

from maze_loader import load_mazes_h5
from maze_pool_env import MazePoolEnv
from utils import setup_logging, clean_outupt_folder


class Config:
    TRAINING_TIMESTEPS = 10_1000  # 10_000
    TRAINING_SAMPLES = 10  # 10000
    TEST_SAMPLES = 5
    VALIDATION_SAMPLES = 5
    MAX_TEST_STEPS = 50  # 200


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


def train_model(env, log_dir, eval_env=None):
    """Train the PPO model on the maze environment"""
    import os

    os.makedirs(log_dir, exist_ok=True)
    logging.info("Starting training...")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,  # PPO will create subdirectories here
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='output/best_model',
        log_path='output/validation_logs',
        eval_freq=100,  # Evaluate every 1000 steps
        deterministic=True,
        render=False
    )

    model.learn(
        total_timesteps=Config.TRAINING_TIMESTEPS,
        progress_bar=True,
        tb_log_name="PPO_MazePool",
        log_interval=1,
        callback=eval_callback
    )

    os.makedirs("output", exist_ok=True)
    model.save("output/ppo_minigrid_maze")

    return model


def run_test_episode(env, model, max_steps, enable_visual=False):
    """Run a test episode with optional visual display"""
    obs, info = env.reset()
    step_count = 0
    done = False
    solution = []

    # Initialize visual display if enabled
    visual_handler = None
    if enable_visual:
        visual_handler = initialize_visual_display()

    while not done and step_count < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        # Update visual display if enabled
        if enable_visual and visual_handler:
            update_visual_display(env, visual_handler, step_count, action, reward, solution)

        agent_pos = env.unwrapped.agent_pos
        agent_pos_tuple = tuple(agent_pos)
        solution.append(agent_pos_tuple)
        done = terminated or truncated
        step_count += 1

        if step_count % 20 == 0:
            logging.info(f"  Step {step_count}: Action={action}, Reward={reward:.3f}")

    # Cleanup visual display if it was used
    if enable_visual and visual_handler:
        cleanup_visual_display(visual_handler)

    return terminated, step_count, solution


def initialize_visual_display():
    """Initialize matplotlib display for visual maze rendering"""
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.show()
    return {'fig': fig, 'ax': ax}


def update_visual_display(env, visual_handler, step_count, action, reward, solution):
    """Update the visual display with current maze state"""
    ax = visual_handler['ax']

    # Clear the previous plot
    ax.clear()

    # Get current state from environment
    maze_grid = env.unwrapped.maze_grid
    agent_pos = env.unwrapped.agent_pos
    target_pos = env.unwrapped.target_pos
    current_maze = env.unwrapped.current_maze

    # Set up the plot
    ax.set_xlim(-0.5, maze_grid.shape[1] - 0.5)
    ax.set_ylim(maze_grid.shape[0] - 0.5, -0.5)
    ax.set_aspect('equal')

    # Draw maze walls and corridors
    for row in range(maze_grid.shape[0]):
        for col in range(maze_grid.shape[1]):
            if maze_grid[row, col] == 1:  # Wall
                rect = patches.Rectangle((col - 0.5, row - 0.5), 1, 1,
                                         facecolor='black', edgecolor='gray')
                ax.add_patch(rect)
            elif maze_grid[row, col] == 0:  # Corridor
                rect = patches.Rectangle((col - 0.5, row - 0.5), 1, 1,
                                         facecolor='white', edgecolor='lightgray')
                ax.add_patch(rect)
            else:  # Padding
                rect = patches.Rectangle((col - 0.5, row - 0.5), 1, 1,
                                         facecolor='gray', edgecolor='darkgray')
                ax.add_patch(rect)

    # Draw agent (blue circle)
    agent_circle = patches.Circle((agent_pos[1], agent_pos[0]), 0.3,
                                  facecolor='blue', edgecolor='darkblue', linewidth=2)
    ax.add_patch(agent_circle)

    # Draw target (red star)
    target_circle = patches.Circle((target_pos[1], target_pos[0]), 0.3,
                                   facecolor='red', edgecolor='darkred', linewidth=2)
    ax.add_patch(target_circle)

    # Draw solution path so far
    if len(solution) > 0:
        path_array = np.array(solution)
        ax.plot(path_array[:, 1], path_array[:, 0],
                color='green', linewidth=2, alpha=0.7, linestyle='--')

    # Set title and remove axes
    ax.set_title(f"Maze {current_maze} - Step {step_count}\nAction: {action}, Reward: {reward:.3f}")
    ax.set_xticks([])
    ax.set_yticks([])

    # Update the display
    plt.draw()
    plt.pause(0.1)  # Pause for animation effect


def cleanup_visual_display(visual_handler):
    """Clean up the visual display resources"""
    plt.ioff()  # Turn off interactive mode
    plt.close(visual_handler['fig'])


def test_model(model, base_test_env, enable_visual=True):
    """Test the trained model on different maze set with optional visual display"""
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

    # Test each maze with optional visual display
    for test_idx in range(total_tests):
        logging.info(f"\n--- Testing on Maze {test_idx + 1}/{total_tests} ---")
        terminated, step_count, solution = run_test_episode(
            test_env, model, Config.MAX_TEST_STEPS, enable_visual=enable_visual
        )

        maze_index = test_env.unwrapped.current_maze
        logging.info(f"Testing on maze index {maze_index}")

        # Evaluate results
        if terminated:
            logging.info(f"SUCCESS! Reached target in {step_count} steps")
            total_success += 1
        elif step_count >= Config.MAX_TEST_STEPS:
            logging.info(f" TIMEOUT after {step_count} steps")
        else:
            logging.info(f"FAILED after {step_count} steps")

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


def main():
    log_dir = "output/"

    # Ensure output directory exists
    import os
    os.makedirs(log_dir, exist_ok=True)

    # Load and create environments with uniform padding
    base_env, base_test_env, validation_env = load_and_create_environments(
        training_file="input/training_mazes.h5",
        test_file="input/mazes.h5",
        validation_file="input/validation_mazes.h5",
        training_samples=Config.TRAINING_SAMPLES,
        test_samples=Config.TEST_SAMPLES,
        validation_samples=Config.VALIDATION_SAMPLES
    )

    # First wrap with RecordVideo for the training environment
    env = RecordVideo(
        base_env,
        video_folder="output/videos",
        episode_trigger=lambda episode_id: episode_id % 50 == 0,
        name_prefix="training"
    )

    # For evaluation environment (do not wrap with RecordVideo)
    eval_env = validation_env

    # Validate compatibility (check base environments)
    validate_environments(base_env, validation_env)
    validate_environments(base_env, base_test_env)

    # Train the model and pass in eval_env (now validation)
    model = train_model(env, log_dir, eval_env=eval_env)

    # Close environments to ensure videos are saved
    env.close()

    logging.info(f"\nVideos saved to: output/videos/")
    logging.info(f"TensorBoard logs saved to: {log_dir}tensorboard/")
    logging.info("To view TensorBoard, run: tensorboard --logdir output/tensorboard --port 6006")


def start_tensorboard(logdir):
    """Start TensorBoard server for PPO's training logs"""
    import os

    os.makedirs(logdir, exist_ok=True)

    try:
        tensorboard = subprocess.Popen(
            ['tensorboard', '--logdir', logdir, '--port', '6006'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logging.info(f"TensorBoard started with logdir: {logdir}")
        logging.info("Open http://localhost:6006 in your browser")
        return tensorboard
    except FileNotFoundError:
        logging.error("TensorBoard not found. Install it with: pip install tensorboard")
        return None
    except Exception as e:
        logging.error(f"Failed to start TensorBoard: {e}")
        return None


def load_and_create_environments(training_file, test_file, validation_file, training_samples, test_samples,
                                 validation_samples):
    """
    Load training and test mazes, pad them to uniform size, and create environments.
    
    Args:
        training_file (str): Path to training mazes HDF5 file
        test_file (str): Path to test mazes HDF5 file
        training_samples (int): Number of training mazes to load
        test_samples (int): Number of test mazes to load
        validation_samples (int): Number of validation mazes to load
        
    Returns:
        tuple: (base_env, base_test_env) - Training and test environments with uniformly padded mazes
    """
    # Load training and test mazes
    maze_grids, starts, exits = load_mazes_h5(training_file, samples=training_samples)
    test_maze_grids, test_starts, test_exits = load_mazes_h5(test_file, samples=test_samples)
    val_maze_grids, val_starts, val_exits = load_mazes_h5(validation_file, samples=validation_samples)

    # Find maximum dimensions across both sets
    max_h = max(
        max(grid.shape[0] for grid in maze_grids),
        max(grid.shape[0] for grid in test_maze_grids),
        max(grid.shape[0] for grid in val_maze_grids)
    )
    max_w = max(
        max(grid.shape[1] for grid in maze_grids),
        max(grid.shape[1] for grid in test_maze_grids),
        max(grid.shape[1] for grid in val_maze_grids)
    )

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
    val_maze_grids, val_starts = pad_to_size(val_maze_grids, val_starts, max_h, max_w)

    # Create environments with padded mazes and support both render modes
    base_env = MazePoolEnv(maze_grids, starts, exits, render_mode="rgb_array")  # Changed back to rgb_array
    base_test_env = MazePoolEnv(test_maze_grids, test_starts, test_exits,
                                render_mode="rgb_array")  # Changed back to rgb_array
    validation_env = MazePoolEnv(val_maze_grids, val_starts, val_exits, render_mode="rgb_array")

    return base_env, base_test_env, validation_env


if __name__ == "__main__":
    clean_outupt_folder()  # Clean output directory before starting
    setup_logging()  # Setup basic logging configuration
    tensorboard_process = start_tensorboard("output/")  # Start tensorboard on output logs
    main()  # Run the main training and evaluation function
