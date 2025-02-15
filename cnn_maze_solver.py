from numpy.f2py.auxfuncs import throw_error
import numpy as np
from maze_solver import MazeSolver
from maze import Maze
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense
from tensorflow.keras.models import Model
import logging

logging.basicConfig(level=logging.INFO)

class CNNMazeSolver(MazeSolver):
    def __init__(self, maze):
        """
        Initializes the CNNMazeSolver with a Maze object.

        Args:
            maze (Maze): The maze to solve.
        """
        super().__init__(maze)
        self.maze = maze
        maze.set_algorithm(self.__class__.__name__)

        # Load or initialize the CNN model
        self.model = self.load_cnn_model()

    def load_cnn_model(self):
        """
        Loads a pre-trained CNN model or initializes a new one.

        Returns:
            model: The CNN model.
        """
        try:
            model = load_model('models/cnn_maze_solver.h5')
            print("CNN model loaded successfully.")
        except FileNotFoundError:
            print("Pre-trained CNN model not found. Initializing a new model.")
            model = self.build_cnn_model()
            # Here you would typically train the model with maze data
            # model.fit(...)
        return model

    def build_cnn_model(self):
        """
        Builds a CNN model architecture for maze solving.

        Returns:
            model: The CNN model.
        """
        input_layer = Input(shape=(self.maze.rows, self.maze.cols, 1))
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        output_layer = Dense(self.maze.rows * self.maze.cols, activation='sigmoid')(x)

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model


    def run_cnn_model(self, maze_input):
        """
        Runs the CNN model to predict the solution path.

        Args:
            maze_input (np.array): The preprocessed maze input.

        Returns:
            np.array: The CNN's predicted output.
        """
        prediction = self.model.predict(maze_input)
        return prediction

    def decode_solution(self, predicted_path):
        """
        Decodes the CNN's output into a sequence of coordinates.

        Args:
            predicted_path (np.array): The CNN's predicted output.

        Returns:
            list: A list of (row, col) tuples representing the solution path.
        """
        # Threshold the prediction to get binary values
        binary_output = (predicted_path > 0.5).astype(int).reshape(self.maze.rows, self.maze.cols)

        path = []
        for row in range(self.maze.rows):
            for col in range(self.maze.cols):
                if binary_output[row, col] == 1:
                    path.append((row, col))

        # Optionally, sort the path based on proximity to the start
        path_sorted = self.sort_path(path)
        return path_sorted

    def sort_path(self, path):
        """
        Sorts the path coordinates from start to exit.

        Args:
            path (list): Unsorted list of (row, col) tuples.

        Returns:
            list: Sorted list of (row, col) tuples.
        """
        start = self.maze.start_position
        exit = self.maze.exit
        # Simple nearest-neighbor sorting; can be replaced with a more sophisticated algorithm
        sorted_path = [start]
        remaining = set(path)
        remaining.remove(start)

        current = start
        while remaining:
            next_step = min(remaining, key=lambda x: abs(x[0] - current[0]) + abs(x[1] - current[1]))
            sorted_path.append(next_step)
            remaining.remove(next_step)
            current = next_step
            if current == exit:
                break

        return sorted_path

    # language: Python

    def prepare_maze_input(self):
        """
        Converts the maze into a format suitable for CNN input.

        Returns:
            np.array: Preprocessed maze input.
        """
        grid = self.maze.grid
        # Assuming WALL=1, CORRIDOR=0
        maze_array = np.array([[1 if cell == Maze.WALL else 0 for cell in row] for row in grid])
        maze_array = maze_array.reshape(1, self.maze.rows, self.maze.cols, 1)
        return maze_array.astype('float32')

    def solve(self):
        """
        Solves the maze using a Convolutional Neural Network (CNN) approach.

        This method preprocesses the maze input, applies a trained CNN to predict the solution path,
        and then translates it back into a sequence of coordinates.

        Returns:
            A list of (row, col) coordinates representing the path from the start to the exit,
            or None if no solution is found.
        """
        if self.maze.exit is None:
            raise ValueError("Maze exit is not set.")

        maze_array = self.prepare_maze_input()
        # Use the CNN model to predict the solution path.
        predicted_path = self.run_cnn_model(maze_array)

        # Convert the CNN output into a sequence of coordinates.
        solution_path = self.decode_solution(predicted_path)

        if not solution_path:
            return None

        self.maze.path = solution_path
        return solution_path

# Example test function for the BFS solver
def test_cnn_solver():
    """
    Test function that loads an array of mazes from 'input/mazes.npy',
    creates a Maze object using the first maze in the array, sets an exit,
    solves the maze using the CNNMazeSolver, and displays the solution.
    """
    try:
        # Load mazes from a NumPy file
        maze_array = np.load("input/mazes.npy", allow_pickle=True)

        # Iterate through each maze in the array
        for i, maze_matrix in enumerate(maze_array):
            logging.debug(f"Solving maze {i + 1} with BFS...")
            maze_obj = Maze(maze_matrix)
            maze_obj.set_animate(True)
            maze_obj.set_save_movie(True)
            solver = CNNMazeSolver(maze_obj)
            solution = solver.solve()

            if solution:
                logging.debug(f"Maze {i + 1} solution found:")
                logging.debug(solution)
            else:
                logging.debug(f"No solution found for maze {i + 1}.")

            # Visualize the solved maze (with the solution path highlighted)
            maze_obj.set_solution(solution)
            maze_obj.plot_maze(show_path=False, show_solution=False)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        throw_error(e)

if __name__ == '__main__':
    test_cnn_solver()