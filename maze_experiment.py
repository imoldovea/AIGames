import os

import numpy as np
from backtrack_maze_solver import BacktrackingMazeSolver
from maze import Maze
from fpdf import FPDF
import tempfile
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)

# Define a custom PDF class (optional, for adding a header)
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Collection of maze solution', ln=True, align='C')
        self.ln(10)

def load_mazes(file_path):
    """
    Loads mazes from a numpy file.

    Args:
        file_path (str): Path to the numpy file containing mazes.

    Returns:
        list: A list of maze matrices.
    """
    try:
        return np.load(file_path, allow_pickle=True)
    except Exception as e:
        raise FileNotFoundError(f"Could not load mazes from {file_path}: {e}")


def solve_all_mazes(mazes, solver_class):
    """
    Solves all the mazes using the provided solver class.

    Args:
        mazes (list): List of maze matrices.
        solver_class (class): Solver class to solve the mazes.

    Returns:
        list: List of tuples with the maze and its solution path.
    """
    solved_mazes = []
    for i, maze_matrix in enumerate(mazes):
        maze_obj = Maze(maze_matrix)
        solver = solver_class(maze_obj)

        try:
            solution = solver.solve()
            maze_obj.set_solution(solution)
            solved_mazes.append((maze_obj, solution))
            logging.debug(f"Maze {i + 1} solved successfully.")
        except Exception as e:
            solved_mazes.append((maze_obj, None))
            logging.error(f"Failed to solve maze {i + 1}: {e}")

    return solved_mazes


def save_mazes_as_pdf(solved_mazes, output_filename="maze_solutions.pdf"):
    """
      Save a collection of maze strings to a PDF file.

      Parameters:
          mazes (list of str): The list of maze representations.
          filename (str): The filename for the output PDF.
      """
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", size=12)

        # Set a base font for the document
        pdf.set_font("Arial", size=12)

        # Iterate over each maze in the list
        for index, (maze_obj, solution) in enumerate(solved_mazes, start=1):
            try:
                # Get the maze image as a numpy array
                image_array = maze_obj.get_maze_as_png(show_path=True, show_solution=True)

                # Save the numpy array as a temporary PNG file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                    temp_image_path = tmp_file.name
                Image.fromarray(image_array).save(temp_image_path)

                pdf.add_page()
                # Optionally add a title for each maze
                pdf.cell(0, 10, f"Maze {index}", ln=True, align='C')
                pdf.ln(5)
                # Use multi_cell to allow for multi-line maze text
                pdf.image(temp_image_path, x=10, y=30, w=pdf.w - 20)
                os.remove(temp_image_path)
            except Exception as e:
                logging.error(f"Error processing maze {index}: {e}")

        # Save the PDF to the specified file
        pdf.output(output_filename)
        logging.info(f"Mazes saved as PDF: {output_filename}")
    except Exception as e:
        logging.error(f"Failed to save mazes to PDF: {e}")


def display_all_mazes(solved_mazes):
    """
    Displays all the mazes and their solutions.

    Args:
        solved_mazes (list): List of tuples with the maze and its solution.
    """
    for i, (maze, solution) in enumerate(solved_mazes):
        try:
            logging.debug(f"Displaying maze {i + 1}...")
            maze.plot_maze(show_path=True, show_solution=True)
        except Exception as e:
            logging.warning(f"Could not display maze {i + 1}: {e}")


def main():
    """
    Main function to load, solve, and save all mazes into a PDF.
    """
    input_file = "input/mazes.npy"
    output_file = "output/solved_mazes.pdf"

    try:
        # Step 1: Load mazes
        mazes = load_mazes(input_file)
        logging.debug(f"Loaded {len(mazes)} mazes from {input_file}.")

        # Step 2: Solve mazes
        solved_mazes = solve_all_mazes(mazes, BacktrackingMazeSolver)

        # Step 3: Save mazes to PDF
        save_mazes_as_pdf(solved_mazes, output_file)
        display_all_mazes(solved_mazes)

    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
