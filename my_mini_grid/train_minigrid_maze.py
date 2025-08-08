import subprocess
import torch
import matplotlib
import subprocess

import matplotlib
import torch

matplotlib.use('Agg')  # Use non-interactive backend for video generation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np  # Import numpy
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from stable_baselines3.common.logger import configure as sb3_configure
from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm

from maze_loader import load_mazes_h5
from maze_pool_env import MazePoolEnv
import sys
import os
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import setup_logging, clean_outupt_folder


class Config:
    TRAINING_TIMESTEPS = 10_000  # 10_000
    TRAINING_SAMPLES = 1000  # 10000
    TEST_SAMPLES = 2
    VALIDATION_SAMPLES = 5
    MAX_TEST_STEPS = TRAINING_SAMPLES / 10  # 200
    ENABLE_TRAINING_VIDEO = False  # If False, disable video generation during training
    ENABLE_VALIDATION_VIDEO = False  # If True, record validation episodes during EvalCallback
    ENABLE_TEST_VIDEO = True  # If True, record videos during testing ("flagf")


class ProgressLoggingCallback(BaseCallback):
    """Shows a progress bar for training progress using tqdm."""
    def __init__(self, total_timesteps: int, min_log_interval_sec: float = 5.0, min_step_delta: int = 1000):
        super().__init__()
        self.total_timesteps = int(total_timesteps)
        self.pbar = None

    def _on_training_start(self) -> None:
        """Initialize progress bar at start of training"""
        # Disable the default progress bar from stable-baselines3
        self.model.progress_bar = True
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress",
                         unit="steps", ncols=100,
                         bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    def _on_step(self) -> bool:
        """Update progress bar on each step"""
        if self.pbar is not None:
            current_steps = self.num_timesteps
            self.pbar.n = current_steps
            self.pbar.update(0)  # Force refresh without incrementing counter
        return True

    def _on_training_end(self) -> None:
        """Close progress bar at end of training"""
        if self.pbar is not None:
            self.pbar.close()


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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    os.makedirs(log_dir, exist_ok=True)
    logging.info("Starting training...")

    # Calculate learning rate schedule
    initial_learning_rate = 3e-4
    end_learning_rate = 1e-4

    def lr_schedule(progress_remaining):
        """
        Linear learning rate schedule.
        :param progress_remaining: float between 0 and 1
        :return: current learning rate
        """
        return end_learning_rate + progress_remaining * (initial_learning_rate - end_learning_rate)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log=os.path.join(log_dir, "tensorboard"),
        learning_rate=3e-4,  # Pass the schedule function
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device=device
    )

    # Configure evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='output/best_model',
        log_path='output/validation_logs',
        eval_freq=100,  # Evaluate every 100 steps
        deterministic=True,
        render=False,
        verbose=0
    )

    # Configure SB3 logger to avoid stdout and only log to files/tensorboard
    tensorboard_dir = os.path.join(log_dir, "tensorboard")
    sb3_logger = sb3_configure(tensorboard_dir, ["csv", "tensorboard"])  # no "stdout" output format
    model.set_logger(sb3_logger)

    # Progress logging callback to emulate progress bar via logs
    progress_cb = ProgressLoggingCallback(total_timesteps=Config.TRAINING_TIMESTEPS,
                                          min_log_interval_sec=5.0,
                                          min_step_delta=5_000)
    callbacks = CallbackList([eval_callback, progress_cb])

    model.learn(
        total_timesteps=Config.TRAINING_TIMESTEPS,
        progress_bar=False,  # Disable default progress bar
        tb_log_name="PPO_MazePool",
        log_interval=1,
        callback=callbacks,
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

    # Create video directory if needed
    if Config.ENABLE_TEST_VIDEO:
        video_dir = "output/videos"
        if not os.path.exists(video_dir):
            logging.info(f"Creating video directory: {video_dir}")
            os.makedirs(video_dir, exist_ok=True)

    # Create a new environment with the correct render mode
    if Config.ENABLE_TEST_VIDEO:
        logging.info("Testing video recording is ENABLED (Config.ENABLE_TEST_VIDEO=True)")
        # Get the underlying environment's data
        unwrapped_env = base_test_env.unwrapped
        # Create a new MazePoolEnv with the correct render mode
        test_env = MazePoolEnv(
            unwrapped_env.maze_grids,
            unwrapped_env.starts,
            unwrapped_env.exits,
            render_mode="rgb_array"
        )
        # Wrap with Monitor first
        test_env = Monitor(test_env)
        # Then wrap with RecordVideo
        test_env = RecordVideo(
            test_env,
            video_folder="output/videos",
            episode_trigger=lambda episode_id: True,  # Record all test episodes
            name_prefix="testing",
            video_length=0  # Record entire episode
        )
        logging.info("Testing videos will be saved to: output/videos/")
    else:
        test_env = base_test_env

    logging.info(f"Successfully loaded {len(unwrapped_env.maze_grids)} test mazes")

    # Test the model on multiple mazes
    total_success = 0
    total_tests = min(5, len(unwrapped_env.maze_grids))  # Test up to 5 mazes

    # Test each maze with optional visual display
    for test_idx in tqdm(range(total_tests), desc="Testing mazes", unit="maze"):
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
    logging.info("Test environment closed - videos should be saved.")
    return success_rate


def main():
    log_dir = "output/"
    tensorboard_log = os.path.join(log_dir, "tensorboard")
    
    # Ensure output directory exists
    os.makedirs(tensorboard_log, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Start tensorboard process
    tensorboard_process = start_tensorboard(tensorboard_log)

    if tensorboard_process is None:
        logging.error("Failed to start TensorBoard. Please check if tensorboard is installed.")
        logging.info("You can install it with: pip install tensorboard")

    # Load and create environments with uniform padding
    base_env, base_test_env, validation_env = load_and_create_environments(
        training_file="input/training_mazes.h5",
        test_file="input/mazes.h5",
        validation_file="input/validation_mazes.h5",
        training_samples=Config.TRAINING_SAMPLES,
        test_samples=Config.TEST_SAMPLES,
        validation_samples=Config.VALIDATION_SAMPLES
    )

    # Conditionally wrap with RecordVideo for the training environment
    if Config.ENABLE_TRAINING_VIDEO:
        env = RecordVideo(
            base_env,
            video_folder="output/videos",
            episode_trigger=lambda episode_id: episode_id % 50 == 0,
            name_prefix="training",
            video_length=0  # Record entire episode
        )
    else:
        env = base_env

    # For evaluation environment, optionally wrap with RecordVideo based on config
    if Config.ENABLE_VALIDATION_VIDEO:
        eval_env = RecordVideo(
            validation_env,
            video_folder="output/validation_videos",
            episode_trigger=lambda episode_id: True,  # Record all validation episodes
            name_prefix="validation",
            video_length=0  # Record entire episode
        )
    else:
        eval_env = validation_env

    # Validate compatibility (check base environments)
    validate_environments(base_env, validation_env)
    validate_environments(base_env, base_test_env)

    try:
        # Train the model and pass in eval_env (now validation)
        model = train_model(env, log_dir, eval_env=eval_env)
        logging.info(f"Training complete. TensorBoard logs saved in {tensorboard_log}")
        logging.info("To view training progress, open http://localhost:6006 in your browser")

        # Close environments to ensure videos are saved
        env.close()

        if Config.ENABLE_TRAINING_VIDEO:
            logging.info(f"\nTraining videos saved to: output/videos/")
        else:
            logging.info("\nTraining video recording is disabled (Config.ENABLE_TRAINING_VIDEO=False)")

        if Config.ENABLE_VALIDATION_VIDEO:
            logging.info(f"Validation videos saved to: output/validation_videos/")
        else:
            logging.info("Validation video recording is disabled (Config.ENABLE_VALIDATION_VIDEO=False)")

        logging.info(f"TensorBoard logs saved to: {log_dir}tensorboard/")
        logging.info("To view TensorBoard, run: tensorboard --logdir output/tensorboard --port 6006")

        # Run tests to generate videos if enabled
        try:
            test_success = test_model(model, base_test_env, enable_visual=False)
            if Config.ENABLE_TEST_VIDEO:
                logging.info(f"Test videos (if any) saved to: output/videos/")
        except Exception as e:
            logging.error(f"Error during testing/video generation: {e}")
    finally:
        if tensorboard_process:
            tensorboard_process.terminate()


def start_tensorboard(logdir):
    """Start TensorBoard server for PPO's training logs"""
    import os

    # Remove any duplicate 'tensorboard' in the path
    if logdir.endswith('tensorboard'):
        logdir = os.path.dirname(logdir)

    # Create the tensorboard directory if it doesn't exist
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
            padded = np.full((target_h, target_w), 1, dtype=grid.dtype)  # Use 1 for padding (treat as wall)
            padded[:h, :w] = grid
            padded_grids.append(padded)
            start_row, start_col = start
            updated_starts.append((start_row, start_col))
        return padded_grids, updated_starts

    # Pad both sets to the same dimensions
    maze_grids, starts = pad_to_size(maze_grids, starts, max_h, max_w)
    test_maze_grids, test_starts = pad_to_size(test_maze_grids, test_starts, max_h, max_w)
    val_maze_grids, val_starts = pad_to_size(val_maze_grids, val_starts, max_h, max_w)

    # Create environments with padded mazes and always use rgb_array render mode
    base_env = MazePoolEnv(maze_grids, starts, exits, render_mode="rgb_array")
    base_test_env = MazePoolEnv(test_maze_grids, test_starts, test_exits, render_mode="rgb_array")
    validation_env = MazePoolEnv(val_maze_grids, val_starts, val_exits, render_mode="rgb_array")

    # Wrap environments with Monitor
    base_env = Monitor(base_env)
    base_test_env = Monitor(base_test_env)
    validation_env = Monitor(validation_env)

    return base_env, base_test_env, validation_env


if __name__ == "__main__":
    clean_outupt_folder()  # Clean output directory before starting
    setup_logging()  # Setup basic logging configuration

    # Optional environment flag to enable testing video generation ("flagf")
    try:
        import os as _os

        _flagf = _os.environ.get("FLAGF", "") or _os.environ.get("ENABLE_TEST_VIDEO", "")
        if str(_flagf).lower() in ("1", "true", "yes", "y", "on"):  # simple truthy check
            Config.ENABLE_TEST_VIDEO = True
            logging.info("FLAGF detected via environment: ENABLE_TEST_VIDEO set to True")
    except Exception as _e:
        logging.warning(f"Could not parse FLAGF environment variable: {_e}")

    tensorboard_process = start_tensorboard("output/")  # Start tensorboard on output logs
    main()  # Run the main training and evaluation function
