import pickle
import random

from maze_visualizer import MazeVisualizer
from neat_solver import NEATSolver
from utils import load_mazes

# Load the saved best genome
with open("output/best_neat_genome.pkl", "rb") as f:
    best_genome = pickle.load(f)

# Load mazes
mazes = load_mazes("input/mazes.h5")

# Visualizer setup
visualizer = MazeVisualizer(renderer_type="matplotlib", output_dir="output")

# Pick a maze to test
maze = random.choice(mazes)
maze.reset()

# Solve with NEAT agent
solver = NEATSolver(maze, best_genome)
solver.solve()

# Visualize solution
visualizer.create_live_matplotlib_animation(maze, solver, algorithm_name="Best NEAT", fps=10, step_delay=0.1)
