import torch
import torch.nn as nn
import torch.optim as optim
from maze_solver import MazeSolver
from backtrack_maze_solver import BacktrackingMazeSolver
from maze import Maze
import numpy as np
import logging
from utils import (
    display_all_mazes,
    save_mazes_as_pdf,
    load_mazes)
from enum import Enum



logging.basicConfig(level=logging.INFO)

# -----------------------------
# PyTorch RNN Model for Maze Solving
# -----------------------------
class MazeRNNModel():


    def train_model(self, training_data, epochs: int = 500, early_stopping: bool = False, patience: int = 10):
        """
        Train the RNN model with training_data, where each entry is a tuple (position, target_action).
        Implements early stopping if enabled.
        """
        break


    def solve(Maze=maze, max_steps=50):
        """
        Use the trained RNN model to predict moves until the maze exit is reached or max_steps is exceeded.
        Each move is made with backtrack=True to show progress. When a solution is found,
        set_solution() is called on the maze.
        """
        solution_path = []
        if not maze.self.exit:
            maze.set_solution(Null)
            return solution_path
        solution_path.append(maze.start_position)


        # Return the solution path.
        return solution_path


# -----------------------------
# Test method for the RNN-based Maze Solver
# -----------------------------
def maze_solver_rnn() -> None:
    """
    This function demonstrates the process of solving mazes using both a traditional
    backtracking approach and a Recurrent Neural Network (RNN) based solver. It initially
    loads mazes for training, solves them using backtracking methods, and then uses the
    solutions to generate training data for the RNN model. After training the RNN, it applies
    the trained neural network to solve additional mazes and save the results into specified
    output formats like PDF or display them visually.

    :return: None
    """
    # Load mazes
    mazes=load_mazes("input/mazes.pkl")
    logging.info(f"Loaded {len(mazes)} mazes.")

    training_mazes = load_mazes("input/training_mazes.pkl")
    logging.info(f"Loaded {len(training_mazes)} training mazes.")

    solved__training_mazes = []
    all_training_data = []  # Initialize an empty list to collect data from all mazes

    # Iterate through each maze in the array
    for idx, maze in enumerate(training_mazes):
        training_maze = Maze(maze)
        training_maze.set_animate(False)
        training_maze.set_save_movie(False)
        logging.debug(f"Training on  maze {idx + 1} with Backtrack...")

        solver = BacktrackingMazeSolver(training_maze)
        try:
            solution = solver.solve()
        except Exception as e:
            logging.error(f"Error solving maze {idx + 1}: {e}")
            continue

        if solution:
            logging.debug(f"Maze {idx + 1} solution found:")
            logging.debug(solution)
        else:
            logging.error(f"No solution found for maze {idx + 1}.")
        training_maze.set_solution(solution)
        solved__training_mazes.append(training_maze)

    #Solve mazes
    solved_mazes = []
    for i, maze_data in enumerate(mazes):
        logging.info(f"Solving the maze {i} with RNN exploration...")
        maze = Maze(maze_data)
        maze.set_animate(False)
        maze.set_save_movie(False)
        #insert code here to solve the maze using the trained model MazeRNNModel

    save_mazes_as_pdf(solved_mazes, "output/rnn_maze_solver_output.pdf")
    display_all_mazes(solved_mazes,)
    # save_movie(solved_mazes, "rnn_maze_solver_output.mp4")


if __name__ == '__main__':
    maze_solver_rnn()
