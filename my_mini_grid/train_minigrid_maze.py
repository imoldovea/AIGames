from stable_baselines3 import PPO

from maze_loader import load_mazes_h5
from maze_pool_env import MazePoolEnv

# Load a batch of mazes (choose a large enough number)
maze_grids, starts, exits = load_mazes_h5("input/mazes.h5", samples=5000)

# Create a randomized environment pool
env = MazePoolEnv(maze_grids, starts, exits)

# Train RL agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)

# Visualize a solution on a new maze
obs = env.reset()
env.render()
