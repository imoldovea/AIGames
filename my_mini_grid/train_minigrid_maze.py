import logging
import subprocess

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
        raise ValueError("Training and test environments have different observation spaces")
    if train_env.action_space.n != test_env.action_space.n:
        raise ValueError("Training and test environments have different action spaces")


def main():
    log_dir = "output/"

    # Configure gym logger to save multiple formats (log, csv, json, tensorboard)
    gym_logger = configure(folder=log_dir, format_strings=["log", "csv", "json", "tensorboard"])
    gym_logger.set_level(logging.INFO)
    gym_logger.info("Starting training...")

    # Load a batch of mazes, sampling 10 for training
    maze_grids, starts, exits = load_mazes_h5("input/training_mazes.h5", samples=Config.TRAINING_SAMPLES)

    # Create a pool environment with randomized maze selection for training
    base_env = MazePoolEnv(maze_grids, starts, exits, render_mode="rgb_array")

    # Wrap with RecordVideo to record training episodes
    env = RecordVideo(
        base_env,
        video_folder="output/videos",
        episode_trigger=lambda episode_id: episode_id % 50 == 0,  # Record every 50th episode
        name_prefix="training"
    )

    # Use MlpPolicy for simple Box observation space
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    model.set_logger(gym_logger)

    # Train the model for a total timesteps with progress bar
    model.learn(total_timesteps=Config.TRAINING_TIMESTEPS, progress_bar=True)  # Increased timesteps

    # Save the trained model to disk
    model.save("output/ppo_minigrid_maze")

    # TESTING PHASE - Load different maze set from output/mazes.h5
    print("\n" + "=" * 50)
    print("TESTING TRAINED MODEL ON DIFFERENT MAZE SET")
    print("=" * 50)

    # Load test mazes from output directory using utils.load_mazes
    print("Loading test mazes from output/mazes.h5...")
    test_maze_grids, test_starts, test_exits = load_mazes_h5("input/mazes.h5", samples=Config.TEST_SAMPLES)

    # Create new environment with test mazes
    base_test_env = MazePoolEnv(test_maze_grids, test_starts, test_exits, render_mode="rgb_array")

    # Wrap test environment with RecordVideo to record all test episodes
    test_env = RecordVideo(
        base_test_env,
        video_folder="output/videos",
        episode_trigger=lambda episode_id: True,  # Record all test episodes
        name_prefix="testing"
    )

    # Validate compatibility (check base environments)
    validate_environments(base_env, base_test_env)

    print(f"Successfully loaded {len(test_maze_grids)} test mazes")

    # Test the model on multiple mazes
    total_success = 0
    total_tests = min(5, len(test_maze_grids))  # Test up to 5 mazes

    for test_idx in range(total_tests):
        print(f"\n--- Testing on Maze {test_idx + 1}/{total_tests} ---")

        # Reset environment for testing
        obs, info = test_env.reset()
        maze_index = info['maze_index']

        print(f"Testing on maze index {maze_index}")
        # Remove or comment out the complexity line since it's not available in this context
        # print(f"Maze complexity: {test_mazes[maze_index].complexity}")

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
                print(f"  Step {step_count}: Action={action}, Reward={reward:.3f}")

        # Evaluate results
        if terminated:
            print(f"  ✅ SUCCESS! Reached target in {step_count} steps")
            total_success += 1

            # Remove maze visualization since test_mazes is not defined
            # test_mazes[maze_index].set_solution(solution)
            # test_mazes[maze_index].plot_maze(show_solution=True, show_path=False)

        elif truncated:
            print(f"  ⏰ TIMEOUT after {step_count} steps")
        else:
            print(f"  ❌ FAILED after {step_count} steps")

        print(f"  Final position: {test_env.unwrapped.agent_pos}")
        print(f"  Target position: {test_env.unwrapped.target_pos}")
        print(f"  Path length: {len(solution)}")

    # Print overall results
    success_rate = (total_success / total_tests) * 100
    print(f"\n{'=' * 50}")
    print(f"OVERALL TEST RESULTS:")
    print(f"Successful: {total_success}/{total_tests} ({success_rate:.1f}%)")
    print(f"{'=' * 50}")

    # Close environments to ensure videos are saved
    env.close()
    test_env.close()

    print(f"\nVideos saved to: output/videos/")


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
    clean_outupt_folder()  # Clean output directory before starting
    setup_logging()  # Setup basic logging configuration
    tensorboard_process = start_tensorboard("output/")  # Start tensorboard on output logs
    main()  # Run the main training and evaluation function
