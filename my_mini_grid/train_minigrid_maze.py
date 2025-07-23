from stable_baselines3 import PPO

from maze_loader import load_mazes_h5
from maze_pool_env import MazePoolEnv
from utils import setup_logging, clean_outupt_folder


def main():
    log_dir = "output/"

    # Load a batch of mazes (choose a large enough number)
    maze_grids, starts, exits = load_mazes_h5("input/mazes.h5", samples=5000)

    # Create a randomized environment pool
    env = MazePoolEnv(maze_grids, starts, exits)

    # Train RL agent
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    model.learn(total_timesteps=1_000_000)

    # Visualize a solution on a new maze
    obs = env.reset()
    env.render()

    obs, info = env.reset()
    done = False
    steps = 0
    frames = []
    while not done:
        frame = env.render(mode='rgb_array')
        frames.append(frame)
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()  # Show the current state (agent, goal, walls, etc.)
        done = terminated or truncated

    imageio.mimsave('maze_solution.gif', frames, duration=0.05)


if __name__ == "__main__":
    clean_outupt_folder()
    setup_logging()
    main()
